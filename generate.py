"""Sample text from a trained MiniGPT checkpoint.

Usage:
    python generate.py --ckpt checkpoints/v1 --prompt "Once upon a time"
    python generate.py --ckpt checkpoints/v1 --prompt "Tom saw a dog" --greedy
    python generate.py --ckpt checkpoints/v1 --prompt "Lily" --temperature 0.8 --top-k 40 --num-samples 3
    python generate.py --ckpt checkpoints/v1 --prompt "Once upon a time" --compare   # greedy vs sampled
    python generate.py --prompt "Tom went" --prompt "The sun was"                    # several prompts in one batch
    python generate.py --prompt "Once upon a time" --no-cache                        # disable the KV cache

Greedy decoding tends to loop on small models; sampling (temperature ~0.8, top-k ~50) is usually more varied.
All prompts and samples of one run are generated together as a single left-padded batch.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from minillm import generation
from minillm.config import Config
from minillm.model import MiniGPT
from minillm.tokenizer import MiniTokenizer
from minillm.utils import get_device, load_checkpoint, set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text with MiniGPT")
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/v1"), help="checkpoint dir or .pt file")
    p.add_argument("--prompt", type=str, action="append", help="repeat to batch several prompts")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50, help="0 disables top-k filtering")
    p.add_argument("--greedy", action="store_true", help="always pick the most likely token")
    p.add_argument("--num-samples", type=int, default=1, help="sampled outputs per prompt")
    p.add_argument("--compare", action="store_true", help="print greedy output, then --num-samples sampled outputs")
    p.add_argument("--no-cache", action="store_true", help="recompute the full context every step")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args(argv)
    args.prompt = args.prompt or ["Once upon a time"]
    return args


def load_model(ckpt: Path, device: torch.device) -> tuple[MiniGPT, MiniTokenizer]:
    path = ckpt / "ckpt.pt" if ckpt.is_dir() else ckpt
    payload = load_checkpoint(path, map_location=device)
    cfg = Config(**payload["config"])
    model = MiniGPT(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, MiniTokenizer.load(cfg.tokenizer_path)


def generate(
    model: MiniGPT,
    input_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    greedy: bool = False,
    eos_id: int | None = None,
    use_cache: bool = True,
) -> list[int]:
    """Single-prompt convenience wrapper around minillm.generation.generate."""
    return generation.generate(model, [input_ids], max_new_tokens, temperature, top_k, greedy,
                               eos_id, use_cache=use_cache)[0]


def main(argv: list[str] | None = None) -> list[str]:
    args = parse_args(argv)
    if args.seed is not None:
        set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()
    model, tok = load_model(args.ckpt, device)
    prompts = [[tok.bos_id] + tok.encode(p) for p in args.prompt]
    print(model.param_report())

    runs: list[tuple[str, bool, int]] = []
    if args.compare or args.greedy:
        runs.append(("greedy", True, 1))
    if args.compare or not args.greedy:
        runs.append((f"sampled (T={args.temperature}, top-k={args.top_k})", False, args.num_samples))

    outputs = []
    for label, greedy, copies in runs:
        batch = [p for p in prompts for _ in range(copies)]
        start = time.perf_counter()
        results = generation.generate(
            model, batch, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
            greedy=greedy, eos_id=tok.eos_id, pad_id=tok.pad_id, use_cache=not args.no_cache,
        )
        elapsed = time.perf_counter() - start
        new_tokens = sum(len(r) - len(p) for r, p in zip(results, batch))
        for i, ids in enumerate(results):
            text = tok.decode(ids)
            outputs.append(text)
            suffix = f" #{i % copies + 1}" if copies > 1 else ""
            print(f"--- {label}{suffix} ---\n{text}\n")
        print(f"[{label}: {new_tokens} tokens in {elapsed:.2f}s = {new_tokens / elapsed:,.0f} tok/s, "
              f"batch={len(batch)}, cache={'off' if args.no_cache else 'on'}]\n")
    return outputs


if __name__ == "__main__":
    main()
