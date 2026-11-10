"""A small video diffusion transformer (rectified flow, adaLN-zero conditioning).

Tokens are 8x8 grayscale patches.  Every token carries an additive spatial and an
additive temporal position embedding.  Attention accepts an additive per-key bias,
which is how proportional attention (+log m on a merged key) and token removal
(-inf on a dropped key) are expressed.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data as D


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=t.dtype) / half)
    args = t[..., None] * 1000.0 * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Block(nn.Module):
    def __init__(self, d, heads, mlp_ratio):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(approximate="tanh"),
                                 nn.Linear(mlp_ratio * d, d))
        self.mod = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, c, key_bias=None):
        s1, g1, a1, s2, g2, a2 = self.mod(c).chunk(6, dim=-1)
        h = self.n1(x) * (1 + g1) + s1
        B, N, d = h.shape
        qkv = self.qkv(h).view(B, N, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=key_bias)
        x = x + a1 * self.proj(att.transpose(1, 2).reshape(B, N, d))
        x = x + a2 * self.mlp(self.n2(x) * (1 + g2) + s2)
        return x


class VideoDiT(nn.Module):
    def __init__(self, d=128, depth=4, heads=4, mlp_ratio=4, temporal_pe=True):
        super().__init__()
        self.d, self.depth, self.heads, self.mlp_ratio = d, depth, heads, mlp_ratio
        self.embed = nn.Linear(D.D_TOK, d)
        self.pos_s = nn.Parameter(torch.randn(D.P_POS, d) * 0.02)
        self.pos_t = nn.Parameter(torch.randn(D.F_FRAMES, d) * 0.02)
        self.temporal_pe = temporal_pe
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads, mlp_ratio) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.final_mod = nn.Linear(d, 2 * d)
        self.final = nn.Linear(d, D.D_TOK)
        for m in (self.final_mod, self.final):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, t, key_bias=None, frame_idx=None, pos_idx=None):
        """x: [B,F,P,D] (full grid) or [B,N',D] with explicit frame_idx/pos_idx [B,N'].

        t: [B].  key_bias: None or [B,N] additive bias on attention keys.
        Returns the velocity (noise minus data) with the same shape as x.
        """
        full = x.dim() == 4
        if full:
            B = x.shape[0]
            x = x.reshape(B, D.N_TOK, D.D_TOK)
            frame_idx = torch.arange(D.F_FRAMES).repeat_interleave(D.P_POS)[None].expand(B, -1)
            pos_idx = torch.arange(D.P_POS).repeat(D.F_FRAMES)[None].expand(B, -1)
        h = self.embed(x) + self.pos_s[pos_idx]
        if self.temporal_pe:
            h = h + self.pos_t[frame_idx]
        c = self.t_mlp(timestep_embedding(t, self.d))[:, None, :]
        kb = None if key_bias is None else key_bias[:, None, None, :].to(h.dtype)
        for blk in self.blocks:
            h = blk(h, c, kb)
        shift, scale = self.final_mod(c).chunk(2, dim=-1)
        out = self.final(self.final_norm(h) * (1 + scale) + shift)
        return out.view(-1, D.F_FRAMES, D.P_POS, D.D_TOK) if full else out


def n_params(model):
    return sum(p.numel() for p in model.parameters())
