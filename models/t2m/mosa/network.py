import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from .embed_rope import apply_rotary_emb

class SwiGLU(nn.Module):
    def __init__(self, latent_dim, ff_size) -> None:
        super().__init__()
        
        self.c_fc1 = nn.Linear(latent_dim, ff_size, bias=False)
        self.c_fc2 = nn.Linear(latent_dim, ff_size, bias=False)
        self.c_proj = nn.Linear(ff_size, latent_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


class SelfAttention(nn.Module):

    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)

        self.n_head = n_head

        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None
        
    
    def kv_caching(self, enable: bool): self.caching, self.cached_k, self.cached_v = enable, None, None

    def forward(self, x, freqs_cis, attn_mask=None, padding_mask=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)

        if freqs_cis is not None:
            q, k= apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        if self.caching:
            if self.cached_k is None: self.cached_k = k; self.cached_v = v
            else: k = self.cached_k = torch.cat((self.cached_k, k), dim=2); v = self.cached_v = torch.cat((self.cached_v, v), dim=2)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # or cross-attention; (B, nh, T, hs) x (B, nh, hs, M) -> (B, nh, T, M)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if attn_mask is not None:
            att = att.masked_fill(attn_mask == 0, -1e30)
        if padding_mask is not None:
            att = att.masked_fill(padding_mask[:, None, None, :] == 0, -1e30)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v # (B, nh, T, M) x (B, nh, M, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    

class CrossAttention(nn.Module):

    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)

        self.n_head = n_head

        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None
    
    
    def kv_caching(self, enable: bool): self.caching, self.cached_k, self.cached_v = enable, None, None

    def forward(self, x, y, attn_mask=None):
        B, T, C = x.size()
        _, M, _ = y.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.caching:
            if self.cached_k is None: 
                self.cached_k = k = self.key(y).view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
                self.cached_v = v = self.value(y).view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
            else: 
                k = self.cached_k; v = self.cached_v
        else:
            k = self.key(y).view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
            v = self.value(y).view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # or cross-attention; (B, nh, T, hs) x (B, nh, hs, M) -> (B, nh, T, M)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if attn_mask is not None:
            att = att.masked_fill(attn_mask[:, None, None, :] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, M) x (B, nh, M, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Block(nn.Module):

    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.1, ff_size=1024, norm_first=True):
        super().__init__()
        self.ln0 = RMSNorm(embed_dim)
        self.ln0_t = RMSNorm(embed_dim)
        self.ln1 = RMSNorm(embed_dim)
        self.ln2 = RMSNorm(embed_dim)
        self.attn0 = CrossAttention(embed_dim, n_head, drop_out_rate)
        self.attn = SelfAttention(embed_dim, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            SwiGLU(embed_dim, ff_size),
            nn.Dropout(drop_out_rate),
        )
        self.norm_first = norm_first

    def kv_caching(self, enable: bool):
        self.attn0.kv_caching(enable)
        self.attn.kv_caching(enable)

    def forward(self, x, y, freqs_cis, attn_mask=None, padding_mask=None):
        if self.norm_first:
            x = x + self.attn0(self.ln0(x), self.ln0_t(y))
            x = x + self.attn(self.ln1(x), freqs_cis, attn_mask, padding_mask)
            x = x + self.mlp(self.ln2(x))
        else:
            x = self.ln0(x + self.attn0(x, self.ln0_t(y)))
            x = self.ln1(x + self.attn(x, freqs_cis, attn_mask, padding_mask))
            x = self.ln2(x + self.mlp(x))
        return x