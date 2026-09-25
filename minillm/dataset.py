"""Load raw text, tokenize, and pack next-token training windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from minillm.config import Config
from minillm.tokenizer import MiniTokenizer, train_or_load

IGNORE_INDEX = -100
TEXT_SUFFIXES = {".txt", ".md"}


def list_text_files(raw_dir: Path | str) -> list[Path]:
    root = Path(raw_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"raw data directory not found: {root}")
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(
            f"no .txt or .md files under {root}. Put a corpus there (see plan.md) "
            "before training the tokenizer or building batches."
        )
    return files


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TextDataset(Dataset):
    """Packed token windows: inputs `x[:-1]`, labels `x[1:]` of length `max_seq_len`."""

    def __init__(
        self,
        tokenizer: MiniTokenizer,
        files: list[Path],
        max_seq_len: int,
    ) -> None:
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        ids = self._concat_documents(files)
        self.windows = self._pack(ids, tokenizer.pad_id, max_seq_len + 1)
        if not self.windows:
            raise ValueError("corpus produced no training windows")

    def _concat_documents(self, files: list[Path]) -> list[int]:
        ids: list[int] = []
        for path in files:
            text = _read_text(path).strip()
            if not text:
                continue
            ids.extend(self.tokenizer.encode(text, add_special_tokens=True))
        if len(ids) < 2:
            raise ValueError("need at least two tokens after encoding the corpus")
        return ids

    def _pack(self, ids: list[int], pad_id: int, window: int) -> list[list[int]]:
        packed: list[list[int]] = []
        for start in range(0, len(ids), window):
            chunk = ids[start : start + window]
            if len(chunk) < 2:
                continue
            if len(chunk) < window:
                chunk = chunk + [pad_id] * (window - len(chunk))
            packed.append(chunk)
        return packed

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.windows[index]
        input_ids = torch.tensor(window[:-1], dtype=torch.long)
        labels = torch.tensor(window[1:], dtype=torch.long)
        labels = labels.masked_fill(labels == self.tokenizer.pad_id, IGNORE_INDEX)
        return {"input_ids": input_ids, "labels": labels}


def make_dataloader(
    dataset: TextDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def build_dataset(config: Config, tokenizer: MiniTokenizer | None = None) -> TextDataset:
    files = list_text_files(config.data_raw_dir)
    if tokenizer is None:
        tokenizer = train_or_load(files, config.vocab_size, config.tokenizer_path)
    return TextDataset(tokenizer, files, config.max_seq_len)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a packed TextDataset and print one batch")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/processed/tokenizer.json"))
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    files = list_text_files(args.raw_dir)
    tok = train_or_load(files, args.vocab_size, args.tokenizer)
    dataset = TextDataset(tok, files, args.max_seq_len)
    loader = make_dataloader(dataset, batch_size=args.batch_size, shuffle=False)
    batch = next(iter(loader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    print(f"windows={len(dataset)} input_ids={tuple(input_ids.shape)} labels={tuple(labels.shape)}")
    print(f"dtypes: {input_ids.dtype} {labels.dtype}")
    sample = tok.decode(input_ids[0].tolist())
    print(f"decode(input_ids[0][:64 ids]): {sample[:200]!r}")


if __name__ == "__main__":
    main()
