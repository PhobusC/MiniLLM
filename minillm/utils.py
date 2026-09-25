"""Device selection, seeding, and checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def get_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path | str, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path | str, map_location: str | torch.device | None = None) -> dict[str, Any]:
    location = map_location if map_location is not None else get_device()
    return torch.load(path, map_location=location, weights_only=False)
