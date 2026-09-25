"""Local chat page for MiniGPT.

Usage:
    .venv/bin/python serve.py                         # uses checkpoints/v2, falls back to checkpoints/v1
    .venv/bin/python serve.py --ckpt checkpoints/v1
Then open http://127.0.0.1:8000

The model is a base language model: each message is a prompt, and the reply is its continuation.
"""

from __future__ import annotations

import argparse
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from generate import load_model
from minillm.generation import stream
from minillm.tokenizer import MiniTokenizer
from minillm.utils import get_device

WEB_DIR = Path(__file__).parent / "web"
TEMPERATURE = 0.8
TOP_K = 50
MAX_NEW_TOKENS = 300


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)


def incremental_decode(tok: MiniTokenizer, tokens: Iterable[int]) -> Iterator[str]:
    """Turn a token stream into text chunks whose concatenation equals decoding all tokens at once.

    Byte-level BPE can split one multi-byte character across tokens, so a chunk is held back
    while the decoded text ends in a replacement character.
    """
    ids: list[int] = []
    sent = ""
    for token in tokens:
        ids.append(token)
        text = tok.decode(ids)
        if text.endswith("�") or not text.startswith(sent):
            continue
        if len(text) > len(sent):
            yield text[len(sent):]
            sent = text
    text = tok.decode(ids)
    if text.startswith(sent) and len(text) > len(sent):
        yield text[len(sent):]


def create_app(ckpt: Path, device: torch.device) -> FastAPI:
    model, tok = load_model(ckpt, device)
    lock = threading.Lock()
    app = FastAPI(title="MiniLLM")

    def reply_chunks(prompt: str) -> Iterator[str]:
        with lock:
            tokens = stream(model, [tok.bos_id] + tok.encode(prompt), MAX_NEW_TOKENS,
                            temperature=TEMPERATURE, top_k=TOP_K, eos_id=tok.eos_id)
            yield from incremental_decode(tok, tokens)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.post("/generate")
    def generate_endpoint(req: GenerateRequest) -> StreamingResponse:
        return StreamingResponse(reply_chunks(req.prompt), media_type="text/plain; charset=utf-8")

    return app


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="Serve the MiniLLM chat page")
    p.add_argument("--ckpt", type=Path, default=None, help="default: checkpoints/v2, else checkpoints/v1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    ckpt = args.ckpt
    if ckpt is None:
        ckpt = next((c for c in (Path("checkpoints/v2"), Path("checkpoints/v1")) if (c / "ckpt.pt").is_file()), None)
        if ckpt is None:
            raise SystemExit("no checkpoint found; pass --ckpt")
    device = torch.device(args.device) if args.device else get_device()
    print(f"serving {ckpt} on http://127.0.0.1:{args.port}")
    uvicorn.run(create_app(ckpt, device), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
