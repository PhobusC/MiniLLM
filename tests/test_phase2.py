from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from minillm.config import Config
from minillm.model import CausalSelfAttention, FeedForward, MiniGPT, TokenPositionalEmbedding, TransformerBlock

B, T = 2, 16


def small_cfg() -> Config:
    return Config(d_model=32, n_heads=4, n_layers=2, d_ff=64, vocab_size=100, max_seq_len=32, dropout=0.0)


class Phase2Tests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.cfg = small_cfg()

    def test_embedding_shape(self) -> None:
        emb = TokenPositionalEmbedding(100, 32, 32, 0.0)
        out = emb(torch.randint(0, 100, (B, T)))
        self.assertEqual(out.shape, (B, T, 32))

    def test_embedding_rejects_long_sequence(self) -> None:
        emb = TokenPositionalEmbedding(100, 32, 8, 0.0)
        with self.assertRaises(ValueError):
            emb(torch.randint(0, 100, (B, 9)))

    def test_attention_shape_and_mask(self) -> None:
        attn = CausalSelfAttention(32, 4, 32, 0.0)
        out, weights = attn(torch.randn(B, T, 32), return_weights=True)
        self.assertEqual(out.shape, (B, T, 32))
        self.assertEqual(weights.shape, (B, 4, T, T))
        upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        self.assertTrue(torch.all(weights[..., upper] == 0))
        torch.testing.assert_close(weights.sum(-1), torch.ones(B, 4, T))

    def test_no_future_leakage(self) -> None:
        model = MiniGPT(self.cfg).eval()
        ids = torch.randint(0, 100, (1, T))
        altered = ids.clone()
        altered[0, 10:] = (altered[0, 10:] + 1) % 100
        with torch.no_grad():
            a, b = model(ids), model(altered)
        torch.testing.assert_close(a[:, :10], b[:, :10])
        self.assertFalse(torch.allclose(a[:, 10:], b[:, 10:]))

    def test_block_and_ffn_shapes(self) -> None:
        x = torch.randn(B, T, 32)
        self.assertEqual(FeedForward(32, 64, 0.0)(x).shape, x.shape)
        self.assertEqual(TransformerBlock(32, 4, 64, 32, 0.0)(x).shape, x.shape)

    def test_model_forward_backward(self) -> None:
        model = MiniGPT(self.cfg)
        ids = torch.randint(0, 100, (B, T))
        logits = model(ids)
        self.assertEqual(logits.shape, (B, T, 100))
        loss = F.cross_entropy(logits.view(-1, 100), ids.view(-1))
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_weight_tying(self) -> None:
        model = MiniGPT(self.cfg)
        self.assertIs(model.lm_head.weight, model.embed.tok.weight)

    def test_v1_param_count_in_range(self) -> None:
        n = MiniGPT(Config()).num_params()
        self.assertTrue(5_000_000 <= n <= 20_000_000, n)


if __name__ == "__main__":
    unittest.main()
