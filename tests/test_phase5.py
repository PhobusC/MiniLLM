from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

import generate
from minillm.config import Config
from minillm.model import MiniGPT
from minillm.tokenizer import MiniTokenizer
from minillm.utils import save_checkpoint

CORPUS = "Once upon a time, a small cat sat on a red mat. The cat liked milk and naps.\n" * 30


class Phase5Tests(unittest.TestCase):
    def test_param_report_flags_target(self) -> None:
        self.assertIn("within target", MiniGPT(Config()).param_report())
        tiny = MiniGPT(Config(d_model=32, n_heads=4, n_layers=1, d_ff=64, vocab_size=50, max_seq_len=16))
        self.assertIn("OUTSIDE target", tiny.param_report())

    def test_compare_prints_greedy_then_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "c.txt"
            corpus.write_text(CORPUS, encoding="utf-8")
            tok = MiniTokenizer.train([corpus], vocab_size=200, min_frequency=1, save_path=root / "tok.json")
            cfg = Config(d_model=32, n_heads=4, n_layers=2, d_ff=64, vocab_size=tok.vocab_size,
                         max_seq_len=32, dropout=0.0, tokenizer_path=root / "tok.json")
            save_checkpoint(root / "ckpt.pt", {"model": MiniGPT(cfg).state_dict(), "optimizer": {},
                                               "config": dataclasses.asdict(cfg), "step": 0})
            base = ["--ckpt", str(root / "ckpt.pt"), "--prompt", "Once", "--max-new-tokens", "5",
                    "--seed", "0", "--device", "cpu"]
            compared = generate.main(base + ["--compare", "--num-samples", "2"])
            self.assertEqual(len(compared), 3)
            greedy_only = generate.main(base + ["--greedy"])
            self.assertEqual(greedy_only, compared[:1])


if __name__ == "__main__":
    unittest.main()
