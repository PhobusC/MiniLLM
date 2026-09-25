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

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """(B, T) int64 -> (B, T, C). position_ids defaults to 0..T-1."""
        if position_ids is None:
            T = input_ids.size(1)
            if T > self.max_seq_len:
                raise ValueError(f"sequence length {T} exceeds max_seq_len {self.max_seq_len}")
            position_ids = torch.arange(T, device=input_ids.device)
        elif int(position_ids.max()) >= self.max_seq_len:
            raise ValueError(f"position {int(position_ids.max())} exceeds max_seq_len {self.max_seq_len}")
        return self.drop(self.tok(input_ids) + self.pos(position_ids))
