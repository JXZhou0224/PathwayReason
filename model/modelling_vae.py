import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import LoraConfig, get_peft_model
from .preceiver import PerceiverVAE
from .configs import VAEConfig

def kl_loss(mu, logvar):
    return -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(), dim=-1
    ).mean()
# ------------------------------------------------------------
# VAE Model (Perceiver-style)
# ------------------------------------------------------------
class VAEModel(nn.Module):
    def __init__(
        self,
        cfg: VAEConfig
    ):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.lm_model_name)
        self.lm_model = AutoModelForCausalLM.from_pretrained(cfg.lm_model_name, device_map=cfg.device)
        self.device = cfg.device
        for p in self.lm_model.parameters():
            p.requires_grad = False
        self.hidden_size = self.lm_model.config.hidden_size
        self.vae = PerceiverVAE(
            dim_lm=self.hidden_size,
            dim_latent=cfg.dim_latent,
            dim_ae=cfg.dim_ae,
            num_encoder_latents=cfg.latent_token_n,
            depth=cfg.depth,
            latent_seed=cfg.latent_seed
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            assert self.tokenizer.pad_token_id is not None
            print("No default pad_token detected, using eos token as padding token")
        self.pad_id = self.tokenizer.pad_token_id
    
    def dump_ckpt(self, path: str):
        torch.save(self.vae.state_dict(), path)

    def load_ckpt(self, path: str, strict: bool = True):
        state = torch.load(path, map_location="cpu")
        self.vae.load_state_dict(state, strict=strict)

    def preprocess_text(self,inputs):
        """
        inputs: {"system","content"}
        """
        encoder_inputs = []
        systems = [ins + "\n" for ins in inputs["system"]]
        contents = inputs["content"]
        batch_size = len(systems)
        texts = [systems[i] + contents[i] for i in range(batch_size)]
        enc = self.tokenizer(texts, 
                                return_tensors="pt", 
                                return_offsets_mapping=True,
                                add_special_tokens=True,
                                padding=True, 
                                truncation=True)
        content_start_tokens = []
        for i in range(batch_size):
            offsets = enc["offset_mapping"][i]
            content_start_char = len(systems[i])
            content_start_token = next(
                i for i, (s, e) in enumerate(offsets)
                if s >= content_start_char
            )
            content_start_tokens.append(content_start_token)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        
        return input_ids, attention_mask, content_start_tokens
    
    def encode_text(self, inputs):
        input_ids, attention_mask, content_start_tokens = self.preprocess_text(inputs)
        return self.encode(input_ids,attention_mask,content_start_tokens)
    
    def encode(self,input_ids,attention_mask, content_start_tokens):
        outputs = self.lm_model.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        batch_size = last_hidden_state.size(0)
        content_attention_mask = attention_mask.clone()
        for i in range(batch_size):
            content_start_token = content_start_tokens[i]
            content_attention_mask[i, :content_start_token] = 0
        mu, logvar = self.vae.encode(last_hidden_state, content_attention_mask)
        return mu, logvar
    
    def decode(
        self,
        ae_latents,          # [B, D]
    ):
        """
        Returns:
            logits: [B, T, V]
            tokens: [B, T]
        """
        decoder_outputs = self.vae.decode(ae_latents)
        device = decoder_outputs.device
        batch_size = decoder_outputs.size(0)
        max_len = self.cfg.max_output_token

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        tokens = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        logits_all = []
        past_kv = None
        inputs_embeds = decoder_outputs
        for t in range(max_len):
            token_embeds = self.lm_model.get_input_embeddings()(tokens[:, -1:]) if tokens is not None else None

            inputs_embeds = torch.cat([inputs_embeds, token_embeds], dim=1) if token_embeds is not None else decoder_outputs

            out = self.lm_model(
                inputs_embeds=inputs_embeds,
                past_key_values=past_kv,
                use_cache=True,
            )

            step_logits = out.logits[:, -1]    # [B, V]
            past_kv = out.past_key_values
            logits_all.append(step_logits)
            next_token = step_logits.argmax(dim=-1)

            # once EOS, always PAD
            next_token = torch.where(
                finished,
                torch.full_like(next_token, pad_id),
                next_token,
            )

            finished |= next_token.eq(eos_id)
            tokens = torch.cat([tokens, next_token[:, None]], dim=1) if tokens is not None else next_token[:, None]

        logits = torch.stack(logits_all, dim=1)     # [B, T, V]
        tokens = tokens[:, 1:]                      # drop BOS → [B, T]

        return logits, tokens
    
    def decode_text(self, ae_latents):
        _, tokens = self.decode(ae_latents)   # tokens: [B, T], padded
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        texts = []
        for seq in tokens:
            seq = seq.tolist()
            # cut at EOS if exists
            if eos_id is not None and eos_id in seq:
                seq = seq[: seq.index(eos_id)]
            # remove PAD
            seq = [t for t in seq if t != pad_id]
            texts.append(self.tokenizer.decode(seq, skip_special_tokens=True))

        return texts


    def forward(self, inputs):
        contents = inputs["content"]
        input_ids,attention_mask,content_start_tokens = self.preprocess_text(inputs)
        mu, logvar = self.encode(input_ids,attention_mask,content_start_tokens)

        z = self.vae.reparametrize(mu, logvar)

        decoder_outputs = self.vae.decode(z)
        max_len = 0
        batch_ids = []
        max_len = 0
        for text in contents:
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            ids.append(self.tokenizer.eos_token_id)
            max_len = max(max_len, len(ids))
            batch_ids.append(ids)

        for i in range(len(batch_ids)):
            batch_ids[i] += [self.pad_id] * (max_len - len(batch_ids[i]))

        target_tokens = torch.tensor(batch_ids, device=decoder_outputs.device)

        # embeddings
        embeds = self.lm_model.get_input_embeddings()(target_tokens)

        # concat latent + text
        inputs_embeds = torch.cat([decoder_outputs, embeds], dim=1)

        # attention mask
        latent_mask = torch.ones(
            target_tokens.size(0),
            decoder_outputs.size(1),
            device=decoder_outputs.device
        )
        text_mask = (target_tokens != self.pad_id).long()
        attention_mask = torch.cat([latent_mask, text_mask], dim=1)

        # forward
        out = self.lm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )

        latent_len = decoder_outputs.size(1)
        logits = out.logits[:, latent_len-1:-1, :]

        recon = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_tokens.reshape(-1),
            ignore_index=self.pad_id
        )

        kl = kl_loss(mu, logvar)
        return recon, kl