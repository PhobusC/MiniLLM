"""Causal multi-head self-attention."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, max_seq_len, max_seq_len), persistent=False)

    def forward(self, x: torch.Tensor, return_weights: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """(B, T, C) -> (B, T, C); optionally also attention weights (B, H, T, T)."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = self.attn_drop(weights) @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.proj(out))
        return (out, weights) if return_weights else out
