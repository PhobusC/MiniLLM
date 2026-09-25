"""Load raw text, tokenize, and pack next-token training windows."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
    dataset: Dataset,
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


DOC_SEPARATOR = "<|endoftext|>"


def _iter_documents(path: Path, docs_per_batch: int, max_batch_chars: int = 50_000_000) -> Iterator[list[str]]:
    """Yield batches of documents split on DOC_SEPARATOR, streaming the file line by line.

    A batch ends at docs_per_batch documents or max_batch_chars characters, so book-length documents
    don't pile up in memory.
    """
    batch: list[str] = []
    batch_chars = 0
    lines: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip() == DOC_SEPARATOR:
                doc = "".join(lines).strip()
                lines = []
                if doc:
                    batch.append(doc)
                    batch_chars += len(doc)
                if len(batch) >= docs_per_batch or batch_chars >= max_batch_chars:
                    yield batch
                    batch = []
                    batch_chars = 0
            else:
                lines.append(line)
    doc = "".join(lines).strip()
    if doc:
        batch.append(doc)
    if batch:
        yield batch


def tokenize_to_bin(
    tokenizer: MiniTokenizer,
    files: list[Path],
    out_path: Path | str,
    docs_per_batch: int = 10_000,
) -> Path:
    """Encode documents (each wrapped in <bos>/<eos>) into a flat uint16 token file."""
    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("vocab too large for uint16 token storage")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")
    n_tokens = 0
    with tmp_path.open("wb") as out:
        for path in files:
            for docs in tqdm(_iter_documents(path, docs_per_batch), desc=f"tokenize {path.name}", unit="batch"):
                ids = np.fromiter(
                    (i for doc in tokenizer.encode_batch(docs, add_special_tokens=True) for i in doc),
                    dtype=np.uint16,
                )
                ids.tofile(out)
                n_tokens += ids.size
    tmp_path.replace(out_path)
    print(f"wrote {n_tokens:,} tokens to {out_path}")
    return out_path


class TokenBinDataset(Dataset):
    """Non-overlapping windows over a memory-mapped uint16 token file."""

    def __init__(self, bin_path: Path | str, max_seq_len: int) -> None:
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.window = max_seq_len + 1
        self.n_windows = len(self.tokens) // self.window
        if self.n_windows == 0:
            raise ValueError(f"{bin_path} has fewer than {self.window} tokens")

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.window
        chunk = torch.from_numpy(self.tokens[start : start + self.window].astype(np.int64))
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


def load_or_tokenize(tokenizer: MiniTokenizer, text_path: Path, processed_dir: Path) -> Path:
    """Return the path of the cached .bin for a text file or directory, tokenizing on first use."""
    files = [text_path] if text_path.is_file() else list_text_files(text_path)
    bin_path = Path(processed_dir) / f"{text_path.stem}.bin"
    if not bin_path.is_file():
        tokenize_to_bin(tokenizer, files, bin_path)
    return bin_path


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
