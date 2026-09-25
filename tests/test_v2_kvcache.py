from __future__ import annotations

import unittest

import torch

from minillm import generation
from minillm.config import Config
from minillm.model import MiniGPT

EOS = 99


def tiny(max_seq_len: int = 32) -> MiniGPT:
    torch.manual_seed(0)
    cfg = Config(d_model=32, n_heads=4, n_layers=2, d_ff=64, vocab_size=100, max_seq_len=max_seq_len, dropout=0.0)
    return MiniGPT(cfg).eval()


class KVCacheTests(unittest.TestCase):
    def test_cached_logits_match_full_forward(self) -> None:
        model = tiny()
        ids = torch.randint(0, 100, (2, 12))
        full = model(ids)
        logits, cache = model(ids[:, :8], use_cache=True)
        steps = [logits]
        for t in range(8, 12):
            step_logits, cache = model(ids[:, t : t + 1], kv_cache=cache, use_cache=True)
            steps.append(step_logits)
        torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-5, rtol=1e-5)

    def test_cache_grows_one_position_per_step(self) -> None:
        model = tiny()
        _, cache = model(torch.randint(0, 100, (1, 5)), use_cache=True)
        _, cache = model(torch.randint(0, 100, (1, 1)), kv_cache=cache, use_cache=True)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache[0][0].shape, (1, 4, 6, 8))

    def test_cached_generation_matches_uncached(self) -> None:
        model = tiny()
        prompt = [[3, 7, 11, 2]]
        cached = generation.generate(model, prompt, 20, greedy=True, use_cache=True)
        uncached = generation.generate(model, prompt, 20, greedy=True, use_cache=False)
        self.assertEqual(cached, uncached)

    def test_batch_matches_individual_prompts(self) -> None:
        model = tiny()
        prompts = [[5], [1, 2, 3, 4, 5, 6], [9, 8, 7]]
        batched = generation.generate(model, prompts, 12, greedy=True, pad_id=0)
        single = [generation.generate(model, [p], 12, greedy=True)[0] for p in prompts]
        self.assertEqual(batched, single)

    def test_padding_mask_blocks_pad_tokens(self) -> None:
        model = tiny()
        short = [4, 5, 6]
        with_pad0 = generation.generate(model, [short, [1] * 8], 6, greedy=True, pad_id=0)[0]
        with_pad50 = generation.generate(model, [short, [1] * 8], 6, greedy=True, pad_id=50)[0]
        self.assertEqual(with_pad0, with_pad50)

    def test_eos_stops_rows_independently(self) -> None:
        model = tiny()
        first = generation.generate(model, [[5, 6]], 1, greedy=True)[0][-1]
        out = generation.generate(model, [[5, 6], [8, 9, 10]], 5, greedy=True, eos_id=first)
        self.assertEqual(out[0], [5, 6])
        self.assertGreater(len(out[1]), 3)

    def test_generation_past_context_window(self) -> None:
        model = tiny(max_seq_len=8)
        out = generation.generate(model, [[1, 2, 3], [4]], 20, greedy=True)
        self.assertEqual([len(o) for o in out], [23, 21])

    def test_stream_matches_generate(self) -> None:
        model = tiny()
        streamed = list(generation.stream(model, [3, 4], 10, greedy=True))
        self.assertEqual([3, 4] + streamed, generation.generate(model, [[3, 4]], 10, greedy=True)[0])

    def test_attn_mask_never_fully_masked(self) -> None:
        pad_mask = torch.tensor([[0, 0, 1, 1]])
        mask = MiniGPT.build_attn_mask(pad_mask, past_len=0, T=4, device=torch.device("cpu"))
        self.assertTrue(bool(mask.any(dim=-1).all()))
        self.assertFalse(bool(mask[0, 0, 3, 0]))


if __name__ == "__main__":
    unittest.main()
