"""Print per-layer activation statistics for a prompt run through a checkpoint.

Usage:
    python inspect_model.py --ckpt checkpoints/v1 --prompt "Once upon a time"
    python inspect_model.py --ckpt checkpoints/v1 --prompt "Once upon a time" --heads   # per-head attention entropy

Columns: mean/std/abs_max/rms of each layer's output; frac_inactive = share of FFN units with GELU output <= 0;
attn_entropy = average attention entropy in nats (0 = attends to one token, ln(T) = uniform).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from generate import load_model
from minillm.probe import ActivationProbe, format_table
from minillm.utils import get_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect MiniGPT layer activations")
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/v1"))
    p.add_argument("--prompt", type=str, default="Once upon a time, there was a little girl named Lily.")
    p.add_argument("--heads", action="store_true", help="also print per-head attention entropy")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


@torch.no_grad()
def main(argv: list[str] | None = None) -> ActivationProbe:
    args = parse_args(argv)
    device = torch.device(args.device) if args.device else get_device()
    model, tok = load_model(args.ckpt, device)
    ids = torch.tensor([[tok.bos_id] + tok.encode(args.prompt)], device=device)
    with ActivationProbe(model) as probe:
        model(ids)
    print(f"{model.param_report()} | prompt tokens={ids.size(1)}")
    print(format_table(probe.stats))
    if args.heads:
        print("\nper-head attention entropy (nats):")
        for name, heads in probe.head_entropy.items():
            print(f"{name:<14}" + " ".join(f"{h:6.3f}" for h in heads))
    return probe


if __name__ == "__main__":
    main()
