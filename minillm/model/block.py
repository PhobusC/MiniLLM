"""Pre-LN transformer block: attention + feed-forward with residuals."""

from __future__ import annotations

import torch
from torch import nn

from minillm.model.attention import CausalSelfAttention, KVCache


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        attn_out = self.attn(self.ln1(x), attn_mask=attn_mask, past_kv=past_kv, use_cache=use_cache)
        present = None
        if use_cache:
            attn_out, present = attn_out
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return (x, present) if use_cache else x
