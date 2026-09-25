"""Build a plain-text Project Gutenberg corpus from the sedthh/gutenberg_english parquet shards."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from minillm.dataset import DOC_SEPARATOR

REPO_ID = "sedthh/gutenberg_english"
_NEWLINE_RUNS = re.compile(r"\n+")
_TRAILING_SPACES = re.compile(r"[ \t]+\n")
_SPACES = re.compile(r"[ \t]{2,}")


@dataclass
class CorpusStats:
    kept: int = 0
    skipped_short: int = 0
    skipped_noisy: int = 0
    train_books: int = 0
    valid_books: int = 0
    train_chars: int = 0
    valid_chars: int = 0
    valid_ids: list[str] = field(default_factory=list)


def clean_text(text: str) -> str:
    """Unwrap hard-wrapped lines into paragraphs and normalize whitespace.

    Most books in this dump are "double-spaced": wrapped lines end in two newlines and paragraphs
    in three. Others use the usual one newline per wrap and a blank line per paragraph. Both become
    one line per paragraph, with paragraphs separated by a blank line.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(DOC_SEPARATOR, "")
    text = _TRAILING_SPACES.sub("\n", text).strip()
    runs = [len(m) for m in _NEWLINE_RUNS.findall(text)]
    double_spaced = runs.count(2) > runs.count(1)
    min_para = 3 if double_spaced else 2
    paragraphs = re.split(rf"\n{{{min_para},}}", text)
    unwrapped = (_SPACES.sub(" ", re.sub(r"\n+", " ", p)).strip() for p in paragraphs)
    return "\n\n".join(p for p in unwrapped if p)


def prose_ratio(text: str) -> float:
    """Share of characters that are letters, whitespace, or common punctuation."""
    if not text:
        return 0.0
    ok = sum(c.isalpha() or c.isspace() or c in ".,;:!?'\"-()" for c in text)
    return ok / len(text)


def is_valid_book(book_id: str, valid_frac: float) -> bool:
    """Deterministic book-level split: the same book always lands on the same side."""
    bucket = int(hashlib.md5(book_id.encode()).hexdigest(), 16) % 10_000
    return bucket < valid_frac * 10_000


def iter_books(parquet_path: Path) -> Iterator[tuple[str, str]]:
    """Yield (book_id, text) from one shard, reading row groups to limit memory."""
    shard = pq.ParquetFile(parquet_path)
    for group in range(shard.num_row_groups):
        table = shard.read_row_group(group, columns=["TEXT", "METADATA"])
        for text, meta in zip(table.column("TEXT").to_pylist(), table.column("METADATA").to_pylist()):
            try:
                book_id = str(json.loads(meta).get("text_id", ""))
            except (TypeError, json.JSONDecodeError):
                book_id = ""
            yield book_id or hashlib.md5(text[:1000].encode()).hexdigest(), text


def build_corpus(
    shards: Iterable[Path],
    out_dir: Path,
    target_chars: int,
    valid_frac: float = 0.005,
    min_chars: int = 20_000,
    min_prose_ratio: float = 0.97,
) -> CorpusStats:
    """Stream books into <out_dir>/Gutenberg-train.txt and Gutenberg-valid.txt until target_chars of training text."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = CorpusStats()
    with (out_dir / "Gutenberg-train.txt").open("w", encoding="utf-8") as train, \
            (out_dir / "Gutenberg-valid.txt").open("w", encoding="utf-8") as valid:
        for shard in shards:
            for book_id, raw in iter_books(shard):
                text = clean_text(raw or "")
                if len(text) < min_chars:
                    stats.skipped_short += 1
                    continue
                if prose_ratio(text) < min_prose_ratio:
                    stats.skipped_noisy += 1
                    continue
                stats.kept += 1
                if is_valid_book(book_id, valid_frac):
                    valid.write(f"{text}\n{DOC_SEPARATOR}\n")
                    stats.valid_books += 1
                    stats.valid_chars += len(text)
                    stats.valid_ids.append(book_id)
                else:
                    train.write(f"{text}\n{DOC_SEPARATOR}\n")
                    stats.train_books += 1
                    stats.train_chars += len(text)
                if stats.train_chars >= target_chars:
                    return stats
    return stats


def download_shards(cache_dir: Path | None = None) -> Iterator[Path]:
    """Download shards one at a time, in order, only as build_corpus asks for them."""
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(f for f in HfApi().list_repo_files(REPO_ID, repo_type="dataset") if f.endswith(".parquet"))
    for name in files:
        yield Path(hf_hub_download(REPO_ID, name, repo_type="dataset", cache_dir=cache_dir))
