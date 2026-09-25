"""Training entrypoint. Loop is phase 3; tokenizer + data are ready now.

Usage (after you add text under data/raw):
    python train.py
"""

from __future__ import annotations

from config import Config
from minillm import get_device, set_seed
from minillm.dataset import build_dataset, make_dataloader


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)
    device = get_device()
    print(f"device={device}")
    dataset = build_dataset(cfg)
    loader = make_dataloader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    batch = next(iter(loader))
    print(
        "phase 1 ready:",
        f"windows={len(dataset)}",
        f"input_ids={tuple(batch['input_ids'].shape)}",
        f"labels={tuple(batch['labels'].shape)}",
        f"vocab_size={dataset.tokenizer.vocab_size}",
    )
    print("training loop is phase 3; model modules are phase 2.")


if __name__ == "__main__":
    main()
