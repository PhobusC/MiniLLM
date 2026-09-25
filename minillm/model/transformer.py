"""Decoder-only transformer stack with LM head."""

from __future__ import annotations

import torch
from torch import nn

from minillm.config import Config
from minillm.model.attention import KVCache
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

    @staticmethod
    def build_attn_mask(
        attention_mask: torch.Tensor | None, past_len: int, T: int, device: torch.device
    ) -> torch.Tensor:
        """Bool (B or 1, 1, T, past_len + T): causal over absolute positions, minus padded keys.

        Each query may always attend to itself so fully padded rows never produce NaN.
        """
        total = past_len + T
        mask = torch.ones(T, total, dtype=torch.bool, device=device).tril(diagonal=past_len)[None]
        if attention_mask is not None:
            mask = mask & attention_mask.bool()[:, None, :]
        idx = torch.arange(T, device=device)
        mask[:, idx, past_len + idx] = True
        return mask[:, None]

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        kv_cache: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[KVCache]]:
        """(B, T) int64 -> logits (B, T, vocab_size); with use_cache also returns per-layer (k, v).

        attention_mask: (B, past_len + T) with 1 for real tokens, 0 for padding.
        """
        T = input_ids.size(1)
        past_len = kv_cache[0][0].size(2) if kv_cache else 0
        if position_ids is None and past_len:
            position_ids = torch.arange(past_len, past_len + T, device=input_ids.device)[None]
        mask = None
        if attention_mask is not None or (past_len and T > 1):
            mask = self.build_attn_mask(attention_mask, past_len, T, input_ids.device)

        x = self.embed(input_ids, position_ids)
        presents: list[KVCache] = []
        for i, block in enumerate(self.blocks):
            past = kv_cache[i] if kv_cache else None
            x = block(x, attn_mask=mask, past_kv=past, use_cache=use_cache)
            if use_cache:
                x, present = x
                presents.append(present)
        logits = self.lm_head(self.ln_f(x))
        return (logits, presents) if use_cache else logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_report(self, target: tuple[int, int] = (5_000_000, 20_000_000)) -> str:
        n = self.num_params()
        lo, hi = target
        status = "within" if lo <= n <= hi else "OUTSIDE"
        return f"params={n:,} ({n / 1e6:.2f}M, {status} target {lo / 1e6:.0f}-{hi / 1e6:.0f}M)"
