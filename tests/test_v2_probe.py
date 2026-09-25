from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

import train
from minillm.config import Config
from minillm.model import MiniGPT
from minillm.probe import ActivationProbe, attention_entropy, format_table, grad_norms

T = 12


def tiny() -> MiniGPT:
    torch.manual_seed(0)
    cfg = Config(d_model=32, n_heads=4, n_layers=2, d_ff=64, vocab_size=100, max_seq_len=32, dropout=0.0)
    return MiniGPT(cfg).eval()


class ProbeTests(unittest.TestCase):
    def test_records_every_layer(self) -> None:
        model = tiny()
        with ActivationProbe(model) as probe:
            model(torch.randint(0, 100, (2, T)))
        expected = {"embed", "ln_f"} | {f"block{i}.{p}" for i in range(2) for p in ("attn", "gelu", "ff", "resid")}
        self.assertEqual(set(probe.stats), expected)
        self.assertIn("frac_inactive", probe.stats["block0.gelu"])
        self.assertEqual(len(probe.head_entropy["block1.attn"]), 4)

    def test_output_unchanged_with_probe(self) -> None:
        model = tiny()
        ids = torch.randint(0, 100, (2, T))
        plain = model(ids)
        with ActivationProbe(model):
            probed = model(ids)
        torch.testing.assert_close(plain, probed, atol=1e-5, rtol=1e-5)

    def test_hooks_removed_on_exit(self) -> None:
        model = tiny()
        with ActivationProbe(model):
            pass
        for module in model.modules():
            self.assertEqual(len(module._forward_hooks), 0)
        self.assertTrue(all(not b.attn.use_manual and b.attn.last_weights is None for b in model.blocks))

    def test_attention_entropy_bounds(self) -> None:
        uniform = torch.full((1, 1, 4, 4), 0.25)
        self.assertAlmostEqual(attention_entropy(uniform).item(), math.log(4), places=5)
        onehot = torch.zeros(1, 1, 4, 4)
        onehot[..., 0] = 1.0
        self.assertAlmostEqual(attention_entropy(onehot).item(), 0.0, places=5)

    def test_grad_norms_groups(self) -> None:
        model = tiny().train()
        ids = torch.randint(0, 100, (2, T))
        logits = model(ids)
        F.cross_entropy(logits.view(-1, 100), ids.view(-1)).backward()
        norms = grad_norms(model)
        self.assertEqual(set(norms), {"embed", "block0", "block1", "ln_f"})
        self.assertTrue(all(v > 0 for v in norms.values()))

    def test_format_table_has_row_per_layer(self) -> None:
        model = tiny()
        with ActivationProbe(model) as probe:
            model(torch.randint(0, 100, (1, T)))
        self.assertEqual(len(format_table(probe.stats).splitlines()), 2 + len(probe.stats))

    def test_train_writes_probe_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "c.txt"
            corpus.write_text(("The cat sat on the mat.\n<|endoftext|>\n") * 40, encoding="utf-8")
            out = root / "run"
            train.main([
                "--data", str(corpus), "--out", str(out), "--tokenizer", str(root / "tok.json"),
                "--processed-dir", str(root / "proc"), "--vocab-size", "300", "--max-seq-len", "16",
                "--batch-size", "2", "--max-steps", "4", "--warmup-steps", "1", "--log-interval", "2",
                "--probe-interval", "2", "--device", "cpu",
            ])
            with (out / "probe.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({r["step"] for r in rows}, {"2", "4"})
            self.assertTrue(any(r["layer"].startswith("grad/") and float(r["grad_norm"]) > 0 for r in rows))


if __name__ == "__main__":
    unittest.main()
