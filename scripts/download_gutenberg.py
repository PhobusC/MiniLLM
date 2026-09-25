"""Download English Project Gutenberg books and write a train/valid text corpus.

Usage (from the repo root):
    .venv/bin/python scripts/download_gutenberg.py                  # ~5GB of training text
    .venv/bin/python scripts/download_gutenberg.py --target-gb 0.2  # quick small corpus

Source: huggingface.co/datasets/sedthh/gutenberg_english (MIT-licensed collection of public-domain books).
Output: data/raw/gutenberg/Gutenberg-{train,valid}.txt, books separated by <|endoftext|>.
The validation split is chosen per book, so no book appears in both files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minillm.gutenberg import build_corpus, download_shards  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=Path("data/raw/gutenberg"))
    p.add_argument("--target-gb", type=float, default=5.0, help="stop after this much training text")
    p.add_argument("--valid-frac", type=float, default=0.005)
    p.add_argument("--min-chars", type=int, default=20_000, help="skip books shorter than this")
    args = p.parse_args()

    stats = build_corpus(download_shards(), args.out_dir, int(args.target_gb * 1e9), args.valid_frac, args.min_chars)
    print(f"kept {stats.kept:,} books (skipped {stats.skipped_short:,} short, {stats.skipped_noisy:,} noisy)")
    print(f"train: {stats.train_books:,} books, {stats.train_chars / 1e9:.2f}G chars")
    print(f"valid: {stats.valid_books:,} books, {stats.valid_chars / 1e6:.1f}M chars")


if __name__ == "__main__":
    main()
