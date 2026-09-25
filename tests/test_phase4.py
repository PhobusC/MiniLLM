from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

import torch

import generate
from minillm.config import Config
from minillm.model import MiniGPT
from minillm.tokenizer import MiniTokenizer
from minillm.utils import save_checkpoint

CORPUS = "Once upon a time, a small cat sat on a red mat. The cat liked milk and naps.\n" * 30


def tiny_model(max_seq_len: int = 16) -> MiniGPT:
    torch.manual_seed(0)
    cfg = Config(d_model=32, n_heads=4, n_layers=2, d_ff=64, vocab_size=50, max_seq_len=max_seq_len, dropout=0.0)
    return MiniGPT(cfg).eval()


class Phase4Tests(unittest.TestCase):
    def test_generates_requested_length(self) -> None:
        out = generate.generate(tiny_model(), [1, 2, 3], max_new_tokens=10)
        self.assertEqual(len(out), 13)
        self.assertEqual(out[:3], [1, 2, 3])

    def test_crops_context_beyond_max_seq_len(self) -> None:
        out = generate.generate(tiny_model(max_seq_len=8), [1] * 6, max_new_tokens=20)
        self.assertEqual(len(out), 26)

    def test_greedy_is_deterministic(self) -> None:
        model = tiny_model()
        a = generate.generate(model, [5, 6], 15, greedy=True)
        b = generate.generate(model, [5, 6], 15, greedy=True)
        self.assertEqual(a, b)

    def test_top_k_one_matches_greedy(self) -> None:
        model = tiny_model()
        greedy = generate.generate(model, [5, 6], 15, greedy=True)
        topk1 = generate.generate(model, [5, 6], 15, temperature=1.0, top_k=1)
        self.assertEqual(greedy, topk1)

    def test_stops_at_eos(self) -> None:
        model = tiny_model()
        first = generate.generate(model, [5, 6], 1, greedy=True)[-1]
        out = generate.generate(model, [5, 6], 20, greedy=True, eos_id=first)
        self.assertEqual(out, [5, 6])

    def test_main_loads_checkpoint_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "c.txt"
            corpus.write_text(CORPUS, encoding="utf-8")
            tok = MiniTokenizer.train([corpus], vocab_size=200, min_frequency=1, save_path=root / "tok.json")
            cfg = Config(d_model=32, n_heads=4, n_layers=2, d_ff=64, vocab_size=tok.vocab_size,
                         max_seq_len=32, dropout=0.0, tokenizer_path=root / "tok.json")
            model = MiniGPT(cfg)
            save_checkpoint(root / "ckpt" / "ckpt.pt",
                            {"model": model.state_dict(), "optimizer": {}, "config": dataclasses.asdict(cfg), "step": 0})
            outputs = generate.main(["--ckpt", str(root / "ckpt"), "--prompt", "Once upon",
                                     "--max-new-tokens", "5", "--num-samples", "2", "--seed", "0", "--device", "cpu"])
            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(o.startswith("Once upon") for o in outputs))


if __name__ == "__main__":
    unittest.main()
