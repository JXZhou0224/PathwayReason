import math
import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F

"""
1. base model size
2. latent size
3. question
"""
from einops import rearrange, reduce, repeat


def exists(x):
    return x is not None

def divisible_by(numer, denom):
    return (numer % denom) == 0


# NN components
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.gamma


def FeedForward(dim, mult=4):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, dim)
    )

# Standard attention
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        qk_norm=True,
    ):
        super().__init__()
        hidden_dim = dim
        heads = dim // dim_head
        assert divisible_by(dim, heads), 'dimension must be divisible by number of heads'


        self.scale = dim_head ** -0.5
        self.heads = heads

        self.norm = nn.LayerNorm(dim) 

        self.query_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.key_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()

        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        self.to_k = nn.Linear(dim, hidden_dim, bias = False)
        self.to_v = nn.Linear(dim, hidden_dim, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(
        self,
        x,
    ):
        h = self.heads

        x = self.norm(x)


        qkv = (self.to_q(x), self.to_k(x), self.to_v(x))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        sim = einsum('b h i d, b h j d -> b h i j', self.query_norm(q)* self.scale, self.key_norm(k))

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# attention pooling

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_latent,
        dim_head=64,
        qk_norm=True,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        inner_dim = max(dim_latent, dim)
        self.heads = inner_dim // dim_head

        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim_latent)

        self.query_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.key_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()

        self.to_q = nn.Linear(dim_latent, inner_dim, bias=False)
        if dim_latent != dim:
            self.latent_to_kv = nn.Linear(dim_latent, inner_dim * 2, bias=False)
        else:
            self.latent_to_kv = None
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_latent),
        )

    def forward(self, x, latents, mask=None):
        x = self.norm(x)
        latents = self.norm_latents(latents)

        b, h = x.shape[0], self.heads

        q = self.to_q(latents)
        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        if exists(self.latent_to_kv):
            kv_input = torch.cat([self.to_kv(x), self.latent_to_kv(latents)], dim=1)
        else:
            kv_input = torch.cat([self.to_kv(x), self.to_kv(latents)], dim=1)
        k, v = rearrange(kv_input, 'b n (split d) -> split b n d', split=2)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # similarities and masking

        sim = einsum('... i d, ... j d  -> ... i j',
                     self.query_norm(q) * self.scale, self.key_norm(k))

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.to(sim.dtype)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_latent,
        depth,
        dim_head=64,
        num_latents=16,
        ff_mult=4,
        l2_normalize_latents=False,
        latent_seed=42,
    ):
        super().__init__()
        self.generator = torch.Generator()
        self.generator.manual_seed(latent_seed)
        self.latents = nn.Parameter(torch.randn(num_latents, dim_latent,generator=self.generator))
        nn.init.normal_(self.latents, std = 0.02,generator=self.generator)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(
                    dim=dim, dim_latent=dim_latent, dim_head=dim_head),
                FeedForward(dim=dim_latent, mult=ff_mult)
            ]))

        self.l2_normalize_latents = l2_normalize_latents

        self.final_norm = nn.LayerNorm(dim_latent)

    def forward(self, x, mask=None):

        latents = repeat(self.latents, 'n d -> b n d', b=x.shape[0])

        for attn, ff in self.layers:
            latents = attn(x, latents, mask=mask) + latents
            latents = ff(latents) + latents
        # Normalize latents to norm sqrt(d_latent)
        if self.l2_normalize_latents:
            latents = F.normalize(latents, dim=-1) * math.sqrt(latents.shape[-1])
        return latents
        
class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim_tx,
        depth,
        dim_head=64,
        ff_mult=4,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(
                    dim=dim_tx, dim_head=dim_head),
                FeedForward(dim=dim_tx, mult=ff_mult)
            ]))

        self.final_norm = nn.LayerNorm(dim_tx)

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.final_norm(x)


class PerceiverVAE(nn.Module):
    def __init__(
        self,
        *,
        dim_lm,
        dim_latent,
        dim_ae,
        depth,
        dim_head=64,
        num_encoder_latents=8,
        ff_mult=4,
        l2_normalize_latents=False,
        latent_seed=42,
    ):
        super().__init__()
        self.perceiver_encoder = PerceiverResampler(dim=dim_lm, dim_latent=dim_latent, depth=depth, dim_head=dim_head,
                                                    num_latents=num_encoder_latents, ff_mult=ff_mult, l2_normalize_latents=l2_normalize_latents, latent_seed=latent_seed)
        self.mean = nn.Linear(dim_latent*num_encoder_latents, dim_ae)
        self.logvar = nn.Linear(dim_latent*num_encoder_latents, dim_ae)
        self.dim_ae = dim_ae
        self.dim_latent = dim_latent
        self.dim_lm = dim_lm
        self.re_project = nn.Linear(dim_ae, dim_lm*num_encoder_latents)
        self.perceiver_decoder = Transformer(dim_tx=dim_lm, depth=depth, dim_head=dim_head, ff_mult=ff_mult)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, ae_latents):
        ae_latents = self.re_project(ae_latents)
        ae_latents = rearrange(ae_latents, 'b (n d) -> b n d', d=self.dim_lm)
        return self.perceiver_decoder(ae_latents)

    def encode(self, encoder_outputs, attention_mask):
        encoder_latents = self.perceiver_encoder(encoder_outputs, mask=attention_mask.bool())
        mean = self.mean(encoder_latents.flatten(start_dim=1))
        logvar = self.logvar(encoder_latents.flatten(start_dim=1))
        return mean, logvar

    def forward(self, encoder_outputs, attention_mask):
        mu, logvar = self.encode(encoder_outputs, attention_mask)
        z = self.reparametrize(mu, logvar)
        decoder_outputs = self.decode(z)
        return decoder_outputs, mu, logvar