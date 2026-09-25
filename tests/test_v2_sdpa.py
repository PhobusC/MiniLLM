from __future__ import annotations

import unittest

import torch

from minillm.config import Config
from minillm.model import CausalSelfAttention, MiniGPT


class SDPATests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_sdpa_matches_manual_attention(self) -> None:
        attn = CausalSelfAttention(32, 4, 32, 0.0).eval()
        x = torch.randn(2, 16, 32)
        fast = attn(x)
        attn.use_manual = True
        slow = attn(x)
        torch.testing.assert_close(fast, slow, atol=1e-5, rtol=1e-5)

    def test_full_model_sdpa_matches_manual(self) -> None:
        cfg = Config(d_model=32, n_heads=4, n_layers=2, d_ff=64, vocab_size=100, max_seq_len=32, dropout=0.0)
        model = MiniGPT(cfg).eval()
        ids = torch.randint(0, 100, (2, 20))
        fast = model(ids)
        for block in model.blocks:
            block.attn.use_manual = True
        torch.testing.assert_close(fast, model(ids), atol=1e-5, rtol=1e-5)

    def test_return_weights_still_available(self) -> None:
        attn = CausalSelfAttention(32, 4, 32, 0.0)
        out, weights = attn(torch.randn(1, 8, 32), return_weights=True)
        self.assertEqual(weights.shape, (1, 4, 8, 8))
        self.assertEqual(out.shape, (1, 8, 32))

    def test_sdpa_dropout_only_in_training(self) -> None:
        attn = CausalSelfAttention(32, 4, 32, 0.5)
        x = torch.randn(1, 8, 32)
        attn.eval()
        torch.testing.assert_close(attn(x), attn(x))
        attn.train()
        self.assertFalse(torch.allclose(attn(x), attn(x)))


if __name__ == "__main__":
    unittest.main()
