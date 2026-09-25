"""Shared hyperparameters and default paths for MiniLLM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Config:
    # Model (v1 decoder-only transformer)
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    vocab_size: int = 8000
    max_seq_len: int = 256
    dropout: float = 0.1
    pos_encoding: str = "rope"  # "rope" or "learned" (v1)

    # Training (used from phase 3; kept here so sizes live in one place)
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 5000
    grad_clip: float = 1.0
    seed: int = 42
    num_workers: int = 0

    # Paths (relative to the process working directory, typically repo root)
    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    tokenizer_path: Path = Path("data/processed/tokenizer.json")
    checkpoint_dir: Path = Path("checkpoints")

    def __post_init__(self) -> None:
        self.data_raw_dir = Path(self.data_raw_dir)
        self.data_processed_dir = Path(self.data_processed_dir)
        self.tokenizer_path = Path(self.tokenizer_path)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.pos_encoding not in ("rope", "learned"):
            raise ValueError(f"unknown pos_encoding {self.pos_encoding!r}")

    @classmethod
    def from_checkpoint(cls, saved: dict[str, Any]) -> Config:
        """Rebuild a saved config; checkpoints from before pos_encoding existed used learned positions."""
        return cls(**{"pos_encoding": "learned", **saved})


def default_config() -> Config:
    return Config()
