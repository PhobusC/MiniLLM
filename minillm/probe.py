"""Forward hooks that record per-layer activation statistics, plus gradient-norm helpers."""

from __future__ import annotations

from types import TracebackType

import torch
from torch import nn

from minillm.model.transformer import MiniGPT

STAT_KEYS = ("mean", "std", "abs_max", "rms", "frac_inactive", "attn_entropy")


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float()
    return {
        "mean": x.mean().item(),
        "std": x.std().item(),
        "abs_max": x.abs().max().item(),
        "rms": x.pow(2).mean().sqrt().item(),
    }


def attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Mean entropy (nats) per head of (B, H, T, T) attention weights -> (H,)."""
    w = weights.float().clamp_min(1e-12)
    return -(weights.float() * w.log()).sum(-1).mean(dim=(0, 2))


class ActivationProbe:
    """Context manager: records stats for the latest forward pass in `self.stats[layer_name]`.

    Layers: embed, block{i}.attn, block{i}.gelu, block{i}.ff, block{i}.resid, ln_f.
    While active, attention uses the manual path so its weights (and entropy) are visible.
    """

    def __init__(self, model: MiniGPT, attention_entropy: bool = True) -> None:
        self.model = model
        self.track_entropy = attention_entropy
        self.stats: dict[str, dict[str, float]] = {}
        self.head_entropy: dict[str, list[float]] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, name: str, kind: str = "plain"):
        def hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
            out = output[0] if isinstance(output, tuple) else output
            row = tensor_stats(out)
            if kind == "gelu":
                row["frac_inactive"] = (out.detach() <= 0).float().mean().item()
            if kind == "attn" and self.track_entropy and module.last_weights is not None:
                per_head = attention_entropy(module.last_weights)
                row["attn_entropy"] = per_head.mean().item()
                self.head_entropy[name] = per_head.tolist()
            self.stats[name] = row

        return hook

    def __enter__(self) -> ActivationProbe:
        m = self.model
        self._register(m.embed, "embed")
        for i, block in enumerate(m.blocks):
            if self.track_entropy:
                block.attn.use_manual = True
            self._register(block.attn, f"block{i}.attn", "attn")
            self._register(block.ff.net[1], f"block{i}.gelu", "gelu")
            self._register(block.ff, f"block{i}.ff")
            self._register(block, f"block{i}.resid")
        self._register(m.ln_f, "ln_f")
        return self

    def _register(self, module: nn.Module, name: str, kind: str = "plain") -> None:
        self._handles.append(module.register_forward_hook(self._hook(name, kind)))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for block in self.model.blocks:
            block.attn.use_manual = False
            block.attn.last_weights = None


def grad_norms(model: MiniGPT) -> dict[str, float]:
    """L2 norm of gradients grouped by top-level component (embed, block{i}, ln_f). Call after backward()."""
    groups: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        parts = name.split(".")
        key = f"block{parts[1]}" if parts[0] == "blocks" else parts[0]
        groups[key] = groups.get(key, 0.0) + p.grad.detach().float().pow(2).sum().item()
    return {k: v**0.5 for k, v in groups.items()}


def format_table(stats: dict[str, dict[str, float]]) -> str:
    header = f"{'layer':<14}" + "".join(f"{k:>14}" for k in STAT_KEYS)
    lines = [header, "-" * len(header)]
    for name, row in stats.items():
        cells = "".join(f"{row[k]:>14.4f}" if k in row else f"{'':>14}" for k in STAT_KEYS)
        lines.append(f"{name:<14}{cells}")
    return "\n".join(lines)
