from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

import torch

import generate
from minillm import generation
from minillm.config import Config
from minillm.model import MiniGPT
from minillm.model.rope import RotaryEmbedding
from minillm.utils import save_checkpoint


def tiny(pos: str = "rope", max_seq_len: int = 32) -> MiniGPT:
    torch.manual_seed(0)
    cfg = Config(d_model=32, n_heads=4, n_layers=2, d_ff=64, vocab_size=100, max_seq_len=max_seq_len,
                 dropout=0.0, pos_encoding=pos)
    return MiniGPT(cfg).eval()


class RoPETests(unittest.TestCase):
    def test_rotation_preserves_norm(self) -> None:
        rope = RotaryEmbedding(8, 32)
        x = torch.randn(2, 4, 16, 8)
        rotated = rope(x, torch.arange(16)[None])
        torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1))

    def test_position_zero_is_identity(self) -> None:
        rope = RotaryEmbedding(8, 32)
        x = torch.randn(1, 1, 1, 8)
        torch.testing.assert_close(rope(x, torch.zeros(1, 1, dtype=torch.long)), x)

    def test_scores_depend_only_on_relative_position(self) -> None:
        rope = RotaryEmbedding(8, 64)
        q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)

        def score(pq: int, pk: int) -> float:
            rq = rope(q, torch.tensor([[pq]]))
            rk = rope(k, torch.tensor([[pk]]))
            return (rq * rk).sum().item()

        self.assertAlmostEqual(score(5, 2), score(20, 17), places=4)
        self.assertAlmostEqual(score(3, 3), score(40, 40), places=4)

    def test_rope_model_has_no_position_table(self) -> None:
        rope_model, learned_model = tiny("rope"), tiny("learned")
        self.assertIsNone(rope_model.embed.pos)
        self.assertEqual(learned_model.num_params() - rope_model.num_params(), 32 * 32)

    def test_rope_model_is_causal(self) -> None:
        model = tiny("rope")
        ids = torch.randint(0, 100, (1, 16))
        altered = ids.clone()
        altered[0, 10:] = (altered[0, 10:] + 1) % 100
        torch.testing.assert_close(model(ids)[:, :10], model(altered)[:, :10])

    def test_rope_cache_and_batch_equivalence(self) -> None:
        model = tiny("rope")
        prompts = [[5], [1, 2, 3, 4, 5, 6], [9, 8, 7]]
        cached = generation.generate(model, prompts, 12, greedy=True, use_cache=True)
        uncached = generation.generate(model, prompts, 12, greedy=True, use_cache=False)
        single = [generation.generate(model, [p], 12, greedy=True)[0] for p in prompts]
        self.assertEqual(cached, uncached)
        self.assertEqual(cached, single)

    def test_old_checkpoint_without_pos_encoding_loads_as_learned(self) -> None:
        cfg = Config(d_model=32, n_heads=4, n_layers=1, d_ff=64, vocab_size=50, max_seq_len=16,
                     pos_encoding="learned")
        saved = dataclasses.asdict(cfg)
        del saved["pos_encoding"]
        restored = Config.from_checkpoint(saved)
        self.assertEqual(restored.pos_encoding, "learned")
        MiniGPT(restored).load_state_dict(MiniGPT(cfg).state_dict())

    def test_load_model_handles_both_encodings(self) -> None:
        from minillm.tokenizer import MiniTokenizer

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "c.txt"
            corpus.write_text("The cat sat on the mat. " * 50, encoding="utf-8")
            MiniTokenizer.train([corpus], vocab_size=100, min_frequency=1, save_path=root / "tok.json")
            for pos in ("rope", "learned"):
                cfg = Config(d_model=32, n_heads=4, n_layers=1, d_ff=64, vocab_size=100, max_seq_len=16,
                             pos_encoding=pos, tokenizer_path=root / "tok.json")
                path = root / f"{pos}.pt"
                save_checkpoint(path, {"model": MiniGPT(cfg).state_dict(), "optimizer": {},
                                       "config": dataclasses.asdict(cfg), "step": 0})
                model, _ = generate.load_model(path, torch.device("cpu"))
                self.assertEqual(model.cfg.pos_encoding, pos)


if __name__ == "__main__":
    unittest.main()
