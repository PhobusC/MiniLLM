"""Rotary position embeddings (RoPE)."""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables; rotates each (even, odd-half) pair of q/k dims by angle pos * freq."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE needs an even head_dim")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(torch.arange(max_seq_len).float(), inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """x: (B, H, T, head_dim); position_ids: (B or 1, T) -> rotated x, same shape."""
        cos = self.cos[position_ids].unsqueeze(1).to(x.dtype)
        sin = self.sin[position_ids].unsqueeze(1).to(x.dtype)
        half = x.size(-1) // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
