from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import train
from minillm.config import Config
from minillm.model import MiniGPT
from minillm.utils import load_checkpoint

V2 = dict(n_layers=12, vocab_size=16_384, max_seq_len=512, pos_encoding="rope")


class ScaleTests(unittest.TestCase):
    def test_v2_param_count_is_about_double_v1(self) -> None:
        v1 = MiniGPT(Config(pos_encoding="learned")).num_params()
        v2 = MiniGPT(Config(**V2)).num_params()
        self.assertAlmostEqual(v2 / 1e6, 13.6, delta=0.1)
        self.assertTrue(1.9 < v2 / v1 < 2.1, v2 / v1)

    def train_args(self, root: Path, extra: list[str]) -> list[str]:
        corpus = root / "c.txt"
        corpus.write_text("The cat sat on the mat and looked at the moon.\n<|endoftext|>\n" * 60, encoding="utf-8")
        return [
            "--data", str(corpus), "--out", str(root / "run"), "--tokenizer", str(root / "tok.json"),
            "--processed-dir", str(root / "proc"), "--vocab-size", "300", "--max-seq-len", "16",
            "--batch-size", "2", "--max-steps", "4", "--warmup-steps", "1", "--log-interval", "2",
            "--device", "cpu", *extra,
        ]

    def test_size_flags_reach_saved_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train.main(self.train_args(root, ["--n-layers", "3", "--d-model", "48", "--n-heads", "6", "--d-ff", "96"]))
            cfg = load_checkpoint(root / "run" / "ckpt.pt", map_location="cpu")["config"]
            self.assertEqual((cfg["n_layers"], cfg["d_model"], cfg["n_heads"], cfg["d_ff"]), (3, 48, 6, 96))

    def test_bf16_autocast_training_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = train.main(self.train_args(Path(tmp), ["--amp", "--d-model", "32", "--n-heads", "4"]))
            self.assertTrue(math.isfinite(result["train_loss"]))


if __name__ == "__main__":
    unittest.main()
