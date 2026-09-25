"""Train MiniGPT on a text corpus.

Usage:
    python train.py --data data/raw/TinyStories-train.txt --val-data data/raw/TinyStories-valid.txt --out checkpoints/v1
    python train.py --data some_small.txt --out checkpoints/overfit --max-steps 300   # overfit sanity check
    python train.py --out checkpoints/v1 --resume                                   # continue from ckpt.pt

The tokenizer is trained once on --data and cached at --tokenizer; each text file is tokenized once
into data/processed/<name>.bin. Logs go to <out>/log.csv, checkpoints to <out>/ckpt.pt.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import math
import time
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from minillm.config import Config
from minillm.dataset import IGNORE_INDEX, TokenBinDataset, list_text_files, load_or_tokenize, make_dataloader
from minillm.model import MiniGPT
from minillm.probe import STAT_KEYS, ActivationProbe, grad_norms
from minillm.tokenizer import train_or_load
from minillm.utils import get_device, load_checkpoint, save_checkpoint, set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    d = Config()
    p = argparse.ArgumentParser(description="Train MiniGPT")
    p.add_argument("--data", type=Path, default=d.data_raw_dir / "TinyStories-train.txt", help="text file or directory")
    p.add_argument("--val-data", type=Path, default=None, help="optional validation text file")
    p.add_argument("--out", type=Path, default=d.checkpoint_dir / "v1")
    p.add_argument("--tokenizer", type=Path, default=d.tokenizer_path)
    p.add_argument("--processed-dir", type=Path, default=d.data_processed_dir)
    p.add_argument("--vocab-size", type=int, default=d.vocab_size)
    p.add_argument("--max-seq-len", type=int, default=d.max_seq_len)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--max-steps", type=int, default=d.max_steps)
    p.add_argument("--warmup-steps", type=int, default=d.warmup_steps)
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--probe-interval", type=int, default=0, help="log activation/grad stats to probe.csv every N steps (0=off)")
    p.add_argument("--resume", action="store_true", help="continue from <out>/ckpt.pt")
    p.add_argument("--device", type=str, default=None, help="override device (cpu, mps, cuda)")
    return p.parse_args(argv)


def lr_at(step: int, cfg: Config) -> float:
    """Linear warmup, then cosine decay to 10% of peak."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def make_optimizer(model: MiniGPT, cfg: Config) -> torch.optim.AdamW:
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def infinite(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yield from loader


def compute_loss(model: MiniGPT, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    x = batch["input_ids"].to(device)
    y = batch["labels"].to(device)
    logits = model(x)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=IGNORE_INDEX)


def write_probe_rows(
    writer: csv.writer, step: int, stats: dict[str, dict[str, float]], grads: dict[str, float]
) -> None:
    for layer, row in stats.items():
        writer.writerow([step, layer, *(f"{row[k]:.6g}" if k in row else "" for k in STAT_KEYS), ""])
    for group, norm in grads.items():
        writer.writerow([step, f"grad/{group}", *([""] * len(STAT_KEYS)), f"{norm:.6g}"])


@torch.no_grad()
def evaluate(model: MiniGPT, loader: DataLoader, device: torch.device, max_batches: int) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        losses.append(compute_loss(model, batch, device).item())
    model.train()
    return sum(losses) / len(losses)


def main(argv: list[str] | None = None) -> dict[str, float]:
    args = parse_args(argv)
    device = torch.device(args.device) if args.device else get_device()
    ckpt_path = args.out / "ckpt.pt"

    if args.resume:
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        cfg = Config(**ckpt["config"])
        cfg.max_steps = args.max_steps
    else:
        ckpt = None
        cfg = Config(
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            lr=args.lr,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            dropout=args.dropout,
            tokenizer_path=args.tokenizer,
            data_processed_dir=args.processed_dir,
        )
    set_seed(cfg.seed)

    train_files = [args.data] if args.data.is_file() else list_text_files(args.data)
    tokenizer = train_or_load(train_files, cfg.vocab_size, cfg.tokenizer_path)
    cfg.vocab_size = tokenizer.vocab_size

    train_ds = TokenBinDataset(load_or_tokenize(tokenizer, args.data, cfg.data_processed_dir), cfg.max_seq_len)
    train_loader = make_dataloader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = None
    if args.val_data is not None:
        val_ds = TokenBinDataset(load_or_tokenize(tokenizer, args.val_data, cfg.data_processed_dir), cfg.max_seq_len)
        val_loader = make_dataloader(val_ds, cfg.batch_size, shuffle=False)

    model = MiniGPT(cfg).to(device)
    optimizer = make_optimizer(model, cfg)
    start_step = 0
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]

    print(f"device={device} {model.param_report()} train_windows={len(train_ds):,} start_step={start_step}")

    def save(step: int) -> None:
        save_checkpoint(
            ckpt_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": dataclasses.asdict(cfg),
                "step": step,
            },
        )

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "log.csv"
    new_log = not (args.resume and log_path.is_file())
    log_file = log_path.open("w" if new_log else "a", newline="")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["step", "lr", "train_loss", "val_loss", "tokens_per_sec"])

    probe_file = None
    probe_log = None
    if args.probe_interval > 0:
        probe_path = args.out / "probe.csv"
        new_probe = not (args.resume and probe_path.is_file())
        probe_file = probe_path.open("w" if new_probe else "a", newline="")
        probe_log = csv.writer(probe_file)
        if new_probe:
            probe_log.writerow(["step", "layer", *STAT_KEYS, "grad_norm"])

    batches = infinite(train_loader)
    model.train()
    running, count = 0.0, 0
    tokens_per_step = cfg.batch_size * cfg.max_seq_len
    t0 = time.perf_counter()
    last = {"train_loss": float("nan"), "val_loss": float("nan")}

    try:
        for step in range(start_step, cfg.max_steps):
            lr = lr_at(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            probing = args.probe_interval > 0 and (step + 1) % args.probe_interval == 0
            with ActivationProbe(model) if probing else contextlib.nullcontext() as probe:
                loss = compute_loss(model, next(batches), device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if probing:
                write_probe_rows(probe_log, step + 1, probe.stats, grad_norms(model))
                probe_file.flush()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            running += loss.item()
            count += 1

            done = step + 1
            is_last = done == cfg.max_steps
            if done % args.log_interval == 0 or is_last:
                elapsed = time.perf_counter() - t0
                tps = count * tokens_per_step / elapsed
                last["train_loss"] = running / count
                val_loss = float("nan")
                if val_loader is not None and (done % args.eval_interval == 0 or is_last):
                    val_loss = evaluate(model, val_loader, device, args.eval_batches)
                    last["val_loss"] = val_loss
                print(f"step {done:>6} | lr {lr:.2e} | loss {last['train_loss']:.4f} | val {val_loss:.4f} | {tps:,.0f} tok/s")
                log.writerow([done, f"{lr:.6e}", f"{last['train_loss']:.4f}", f"{val_loss:.4f}", f"{tps:.0f}"])
                log_file.flush()
                running, count = 0.0, 0
                t0 = time.perf_counter()
            if done % args.save_interval == 0 or is_last:
                save(done)
    finally:
        log_file.close()
        if probe_file is not None:
            probe_file.close()

    print(f"saved checkpoint to {ckpt_path}; loss curve in {log_path}")
    return last


if __name__ == "__main__":
    main()
