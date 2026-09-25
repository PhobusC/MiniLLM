"""Sample text from a trained MiniGPT checkpoint.

Usage:
    python generate.py --ckpt checkpoints/v1 --prompt "Once upon a time"
    python generate.py --ckpt checkpoints/v1 --prompt "Tom saw a dog" --greedy
    python generate.py --ckpt checkpoints/v1 --prompt "Lily" --temperature 0.8 --top-k 40 --num-samples 3
    python generate.py --ckpt checkpoints/v1 --prompt "Once upon a time" --compare   # greedy vs sampled

Greedy decoding tends to loop on small models; sampling (temperature ~0.8, top-k ~50) is usually more varied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minillm.config import Config
from minillm.model import MiniGPT
from minillm.tokenizer import MiniTokenizer
from minillm.utils import get_device, load_checkpoint, set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text with MiniGPT")
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/v1"), help="checkpoint dir or .pt file")
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50, help="0 disables top-k filtering")
    p.add_argument("--greedy", action="store_true", help="always pick the most likely token")
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--compare", action="store_true", help="print one greedy output, then --num-samples sampled outputs")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def load_model(ckpt: Path, device: torch.device) -> tuple[MiniGPT, MiniTokenizer]:
    path = ckpt / "ckpt.pt" if ckpt.is_dir() else ckpt
    payload = load_checkpoint(path, map_location=device)
    cfg = Config(**payload["config"])
    model = MiniGPT(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, MiniTokenizer.load(cfg.tokenizer_path)


@torch.no_grad()
def generate(
    model: MiniGPT,
    input_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    greedy: bool = False,
    eos_id: int | None = None,
) -> list[int]:
    """Autoregressively extend input_ids; re-runs the last max_seq_len tokens each step (no KV cache)."""
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(ids[:, -model.cfg.max_seq_len :])[0, -1]
        if greedy:
            next_id = int(logits.argmax())
        else:
            logits = logits / max(temperature, 1e-5)
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            next_id = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))
        if next_id == eos_id:
            break
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    return ids[0].tolist()


def main(argv: list[str] | None = None) -> list[str]:
    args = parse_args(argv)
    if args.seed is not None:
        set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()
    model, tok = load_model(args.ckpt, device)
    prompt_ids = [tok.bos_id] + tok.encode(args.prompt)
    print(model.param_report())

    runs = [("greedy", True)] if args.compare or args.greedy else []
    if not args.greedy or args.compare:
        label = f"sampled (T={args.temperature}, top-k={args.top_k})"
        runs += [(f"{label} #{i + 1}", False) for i in range(args.num_samples)]

    outputs = []
    for label, greedy in runs:
        ids = generate(
            model, prompt_ids, args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, greedy=greedy, eos_id=tok.eos_id,
        )
        text = tok.decode(ids)
        outputs.append(text)
        print(f"--- {label} ---\n{text}\n")
    return outputs


if __name__ == "__main__":
    main()
