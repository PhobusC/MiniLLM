"""Token and learned positional embeddings."""

from __future__ import annotations

import torch
from torch import nn


class TokenPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """(B, T) int64 -> (B, T, C)."""
        T = input_ids.size(1)
        if T > self.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.max_seq_len}")
        positions = torch.arange(T, device=input_ids.device)
        return self.drop(self.tok(input_ids) + self.pos(positions))
