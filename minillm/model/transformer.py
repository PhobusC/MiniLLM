"""Decoder-only transformer stack with LM head."""

from __future__ import annotations

import torch
from torch import nn

from minillm.config import Config
from minillm.model.block import TransformerBlock
from minillm.model.embeddings import TokenPositionalEmbedding


class MiniGPT(nn.Module):
    def __init__(self, cfg: Config, tie_weights: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = TokenPositionalEmbedding(cfg.vocab_size, cfg.d_model, cfg.max_seq_len, cfg.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.max_seq_len, cfg.dropout)
            for _ in range(cfg.n_layers)
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        if tie_weights:
            self.lm_head.weight = self.embed.tok.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """(B, T) int64 -> logits (B, T, vocab_size)."""
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
