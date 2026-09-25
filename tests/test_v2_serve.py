from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

import torch
from fastapi.testclient import TestClient

import serve
from minillm.config import Config
from minillm.model import MiniGPT
from minillm.tokenizer import MiniTokenizer
from minillm.utils import save_checkpoint


class ServeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        corpus = root / "c.txt"
        corpus.write_text("Once upon a time, a café owner named Zoë baked bread. " * 50, encoding="utf-8")
        tok = MiniTokenizer.train([corpus], vocab_size=300, min_frequency=1, save_path=root / "tok.json")
        cfg = Config(d_model=32, n_heads=4, n_layers=1, d_ff=64, vocab_size=tok.vocab_size, max_seq_len=64,
                     dropout=0.0, tokenizer_path=root / "tok.json")
        torch.manual_seed(0)
        save_checkpoint(root / "ckpt.pt", {"model": MiniGPT(cfg).state_dict(), "optimizer": {},
                                           "config": dataclasses.asdict(cfg), "step": 0})
        cls.client = TestClient(serve.create_app(root / "ckpt.pt", torch.device("cpu")))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_index_serves_chat_page(self) -> None:
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn('id="messages"', res.text)

    def test_generate_streams_nonempty_text(self) -> None:
        torch.manual_seed(1)  # untrained model: fix sampling so it doesn't emit <eos> first
        with self.client.stream("POST", "/generate", json={"prompt": "Once upon a time"}) as res:
            self.assertEqual(res.status_code, 200)
            chunks = list(res.iter_text())
        self.assertTrue("".join(chunks).strip())

    def test_incremental_decode_never_splits_characters(self) -> None:
        tok = MiniTokenizer.load(Path(self.tmp.name) / "tok.json")
        text = "Zoë ran to the café — naïve ☕ 🍞 done."
        ids = tok.encode(text)
        chunks = list(serve.incremental_decode(tok, ids))
        self.assertEqual("".join(chunks), tok.decode(ids))
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all("�" not in c for c in chunks))
        self.assertGreater(len(chunks), 1)

    def test_rejects_empty_prompt(self) -> None:
        self.assertEqual(self.client.post("/generate", json={"prompt": ""}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
