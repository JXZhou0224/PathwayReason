from model.modelling_vae import VAEModel
from model.configs import VAEConfig
from transformers import AutoTokenizer
def test_vae_model():
    cfg = VAEConfig()
    model = VAEModel(cfg=VAEConfig()).to(cfg.device)
    test_input = [
        {
            "system": "You are a helpful assistant.",
            "content": "The quick brown fox jumps over the lazy dog."
        }
    ]
    res, _ , _ = model(test_input)
    logits = res
    token_ids = logits.argmax(dim=-1)
    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_model_name)
    text = tokenizer.batch_decode(
        token_ids,
        skip_special_tokens=True
    )
    print(text)
test_vae_model()