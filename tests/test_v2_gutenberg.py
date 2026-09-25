from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from minillm.dataset import DOC_SEPARATOR, tokenize_to_bin
from minillm.gutenberg import build_corpus, clean_text, is_valid_book, prose_ratio
from minillm.tokenizer import MiniTokenizer

PARA = "It was a bright cold day in April, and the clocks were striking thirteen. " * 20


def write_shard(path: Path, books: list[tuple[str, str]]) -> Path:
    table = pa.table({
        "TEXT": [text for _, text in books],
        "SOURCE": ["test"] * len(books),
        "METADATA": [json.dumps({"text_id": bid, "language": "en"}) for bid, _ in books],
    })
    pq.write_table(table, path)
    return path


class GutenbergTests(unittest.TestCase):
    def test_clean_text_unwraps_double_spaced_books(self) -> None:
        raw = "The first line\r\n\r\ncontinues here.\r\n\r\n\r\nA new paragraph\r\n\r\nstarts."
        self.assertEqual(clean_text(raw), "The first line continues here.\n\nA new paragraph starts.")

    def test_clean_text_unwraps_single_spaced_books(self) -> None:
        raw = "The first line\ncontinues here.\n\nA new\nparagraph."
        self.assertEqual(clean_text(raw), "The first line continues here.\n\nA new paragraph.")

    def test_clean_text_strips_separator_and_spaces(self) -> None:
        self.assertEqual(clean_text(f"a  b {DOC_SEPARATOR} c   \n"), "a b c")

    def test_prose_ratio_flags_noisy_text(self) -> None:
        self.assertGreater(prose_ratio(PARA), 0.97)
        self.assertLess(prose_ratio("01:001:003 | 42 | 17 | {x} = [3, 4]"), 0.8)

    def test_book_split_is_deterministic_and_proportional(self) -> None:
        ids = [str(i) for i in range(20_000)]
        valid = [i for i in ids if is_valid_book(i, 0.01)]
        self.assertEqual(valid, [i for i in ids if is_valid_book(i, 0.01)])
        self.assertTrue(150 < len(valid) < 250, len(valid))

    def test_build_corpus_filters_and_splits_by_book(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            books = [(str(i), PARA) for i in range(300)] + [("short", "Too short."), ("noisy", "1|2|3|{}[]" * 500)]
            shard = write_shard(root / "s.parquet", books)
            stats = build_corpus([shard], root / "out", target_chars=10**9, valid_frac=0.05, min_chars=500)
            self.assertEqual((stats.skipped_short, stats.skipped_noisy, stats.kept), (1, 1, 300))
            self.assertEqual(stats.train_books + stats.valid_books, 300)
            self.assertGreater(stats.valid_books, 0)
            train = (root / "out" / "Gutenberg-train.txt").read_text()
            valid = (root / "out" / "Gutenberg-valid.txt").read_text()
            self.assertEqual(train.count(DOC_SEPARATOR), stats.train_books)
            self.assertEqual(valid.count(DOC_SEPARATOR), stats.valid_books)
            self.assertTrue(all(is_valid_book(b, 0.05) for b in stats.valid_ids))

    def test_build_corpus_stops_at_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = write_shard(root / "s.parquet", [(str(i), PARA) for i in range(100)])
            stats = build_corpus([shard], root / "out", target_chars=5 * len(clean_text(PARA)), valid_frac=0.0, min_chars=500)
            self.assertEqual(stats.train_books, 5)

    def test_long_documents_tokenize_in_char_limited_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "books.txt"
            corpus.write_text("".join(f"{PARA}\n{DOC_SEPARATOR}\n" for _ in range(6)), encoding="utf-8")
            tok = MiniTokenizer.train([corpus], vocab_size=300, min_frequency=1)
            ids = np.fromfile(tokenize_to_bin(tok, [corpus], root / "b.bin"), dtype=np.uint16)
            self.assertEqual(int((ids == tok.bos_id).sum()), 6)


if __name__ == "__main__":
    unittest.main()
