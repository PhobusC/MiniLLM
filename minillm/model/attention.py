"""Causal multi-head self-attention with optional KV cache."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

KVCache = tuple[torch.Tensor, torch.Tensor]


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

    def _manual_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Explicit softmax(QKᵀ/√d)V; slower than SDPA but exposes the attention weights."""
        if attn_mask is None:
            attn_mask = torch.ones(q.size(-2), k.size(-2), dtype=torch.bool, device=q.device).tril()
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~attn_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return self.attn_drop(weights) @ v, weights

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
        attn_mask: torch.Tensor | None = None,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, KVCache]:
        """(B, T, C) -> (B, T, C).

        attn_mask: bool (B, 1, T, T_total), True = may attend. None means plain causal (no cache).
        Returns (out, weights) if return_weights, (out, (k, v)) if use_cache, else out.
        """
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        weights = None
        if return_weights or self.use_manual:
            out, weights = self._manual_attention(q, k, v, attn_mask)
            if self.use_manual:
                self.last_weights = weights.detach()
        else:
            dropout_p = self.dropout if self.training else 0.0
            is_causal = attn_mask is None and past_kv is None
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.proj(out))
        if return_weights:
            return out, weights
        if use_cache:
            return out, (k, v)
        return out
