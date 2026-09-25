from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from minillm.config import Config
from minillm.dataset import IGNORE_INDEX, TextDataset, make_dataloader
from minillm.tokenizer import MiniTokenizer
from minillm.utils import get_device


SAMPLE = (
    "Once upon a time, a small cat sat on a red mat. "
    "The cat liked milk, fish, and long naps in the sun. "
) * 40


class Phase01Tests(unittest.TestCase):
    def test_config_v1_sizes(self) -> None:
        cfg = Config()
        self.assertEqual(cfg.d_model, 256)
        self.assertEqual(cfg.n_heads, 8)
        self.assertEqual(cfg.n_layers, 6)
        self.assertEqual(cfg.d_ff, 1024)
        self.assertEqual(cfg.vocab_size, 8000)
        self.assertEqual(cfg.max_seq_len, 256)
        self.assertEqual(cfg.dropout, 0.1)

    def test_device_is_torch_device(self) -> None:
        device = get_device()
        self.assertIsInstance(device, torch.device)
        self.assertIn(device.type, {"cuda", "mps", "cpu"})

    def test_encode_decode_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.txt"
            corpus.write_text(SAMPLE, encoding="utf-8")
            tok = MiniTokenizer.train([corpus], vocab_size=500, min_frequency=1)
            text = "The cat sat on the mat."
            ids = tok.encode(text, add_special_tokens=True)
            self.assertEqual(ids[0], tok.bos_id)
            self.assertEqual(ids[-1], tok.eos_id)
            decoded = tok.decode(ids)
            self.assertTrue(decoded.strip())
            self.assertIn("cat", decoded.lower())

    def test_packed_batch_shapes_and_shift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.txt"
            corpus.write_text(SAMPLE, encoding="utf-8")
            tok = MiniTokenizer.train([corpus], vocab_size=500, min_frequency=1)
            max_seq_len = 32
            dataset = TextDataset(tok, [corpus], max_seq_len=max_seq_len)
            loader = make_dataloader(dataset, batch_size=2, shuffle=False)
            batch = next(iter(loader))
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            self.assertEqual(input_ids.dtype, torch.int64)
            self.assertEqual(labels.dtype, torch.int64)
            self.assertEqual(input_ids.shape[1], max_seq_len)
            self.assertEqual(labels.shape, input_ids.shape)

            window = dataset.windows[0]
            expected_x = torch.tensor(window[:-1], dtype=torch.long)
            expected_y = torch.tensor(window[1:], dtype=torch.long)
            expected_y = expected_y.masked_fill(expected_y == tok.pad_id, IGNORE_INDEX)
            torch.testing.assert_close(dataset[0]["input_ids"], expected_x)
            torch.testing.assert_close(dataset[0]["labels"], expected_y)


if __name__ == "__main__":
    unittest.main()
