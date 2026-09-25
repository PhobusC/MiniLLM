from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

import train
from minillm.dataset import TokenBinDataset, tokenize_to_bin
from minillm.tokenizer import MiniTokenizer
from minillm.utils import load_checkpoint

STORY = "Once upon a time, a small cat sat on a red mat. The cat liked milk and naps.\n"
CORPUS = (STORY + "<|endoftext|>\n") * 30


class Phase3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.corpus = self.root / "tiny.txt"
        self.corpus.write_text(CORPUS, encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def common_args(self, out: Path) -> list[str]:
        return [
            "--data", str(self.corpus), "--out", str(out),
            "--tokenizer", str(self.root / "tok.json"), "--processed-dir", str(self.root / "proc"),
            "--vocab-size", "300", "--max-seq-len", "32", "--batch-size", "4",
            "--warmup-steps", "10", "--lr", "1e-3", "--dropout", "0.0",
            "--log-interval", "10", "--save-interval", "1000", "--device", "cpu",
        ]

    def test_tokenize_to_bin_splits_documents(self) -> None:
        tok = MiniTokenizer.train([self.corpus], vocab_size=300, min_frequency=1)
        bin_path = tokenize_to_bin(tok, [self.corpus], self.root / "tiny.bin")
        ids = np.fromfile(bin_path, dtype=np.uint16)
        self.assertEqual(int((ids == tok.bos_id).sum()), 30)
        self.assertEqual(int((ids == tok.eos_id).sum()), 30)
        self.assertNotIn("<|endoftext|>", tok.decode(ids[:50].tolist()))

    def test_bin_dataset_shift(self) -> None:
        path = self.root / "seq.bin"
        np.arange(100, dtype=np.uint16).tofile(path)
        ds = TokenBinDataset(path, max_seq_len=9)
        self.assertEqual(len(ds), 10)
        item = ds[1]
        self.assertEqual(item["input_ids"].tolist(), list(range(10, 19)))
        self.assertEqual(item["labels"].tolist(), list(range(11, 20)))

    def test_lr_schedule(self) -> None:
        cfg = train.Config(lr=1.0, warmup_steps=10, max_steps=110)
        self.assertAlmostEqual(train.lr_at(0, cfg), 0.1)
        self.assertAlmostEqual(train.lr_at(9, cfg), 1.0)
        self.assertAlmostEqual(train.lr_at(110, cfg), 0.1)

    def test_overfit_loss_drops_and_checkpoint_resumes(self) -> None:
        out = self.root / "ckpt"
        result = train.main(self.common_args(out) + ["--max-steps", "150"])
        with (out / "log.csv").open() as f:
            losses = [float(r["train_loss"]) for r in csv.DictReader(f)]
        self.assertLess(losses[-1], 0.5 * losses[0])
        self.assertLess(result["train_loss"], 1.0)

        ckpt = load_checkpoint(out / "ckpt.pt", map_location="cpu")
        self.assertEqual(ckpt["step"], 150)
        self.assertEqual(set(ckpt), {"model", "optimizer", "config", "step"})

        train.main(self.common_args(out) + ["--max-steps", "160", "--resume"])
        self.assertEqual(load_checkpoint(out / "ckpt.pt", map_location="cpu")["step"], 160)


if __name__ == "__main__":
    unittest.main()
