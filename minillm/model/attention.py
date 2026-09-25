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
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.use_manual = False
        self.last_weights: torch.Tensor | None = None

    def _manual_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Explicit softmax(QKᵀ/√d)V; slower than SDPA but exposes the attention weights."""
        T = q.size(-2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return self.attn_drop(weights) @ v, weights

    def forward(self, x: torch.Tensor, return_weights: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """(B, T, C) -> (B, T, C); optionally also attention weights (B, H, T, T)."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        weights = None
        if return_weights or self.use_manual:
            out, weights = self._manual_attention(q, k, v)
            if self.use_manual:
                self.last_weights = weights.detach()
        else:
            dropout_p = self.dropout if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.proj(out))
        return (out, weights) if return_weights else out
