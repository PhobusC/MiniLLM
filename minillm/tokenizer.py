"""Train, load, encode, and decode a byte-level BPE tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class MiniTokenizer:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer
        self.pad_id = self._require_id("<pad>")
        self.bos_id = self._require_id("<bos>")
        self.eos_id = self._require_id("<eos>")
        self.unk_id = self._require_id("<unk>")

    def _require_id(self, token: str) -> int:
        token_id = self._tok.token_to_id(token)
        if token_id is None:
            raise ValueError(f"tokenizer is missing special token {token!r}")
        return token_id

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = self._tok.encode(text).ids
        if add_special_tokens:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            special = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
            ids = [i for i in ids if i not in special]
        return self._tok.decode(ids)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: Path | str) -> MiniTokenizer:
        tokenizer = Tokenizer.from_file(str(path))
        return cls(tokenizer)

    @classmethod
    def train(
        cls,
        files: list[Path | str],
        vocab_size: int,
        save_path: Path | str | None = None,
        min_frequency: int = 2,
    ) -> MiniTokenizer:
        paths = [str(Path(f)) for f in files]
        if not paths:
            raise ValueError("need at least one text file to train a tokenizer")
        for p in paths:
            if not Path(p).is_file():
                raise FileNotFoundError(p)

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=min_frequency,
        )
        tokenizer.train(paths, trainer)
        wrapped = cls(tokenizer)
        if save_path is not None:
            wrapped.save(save_path)
        return wrapped


def train_or_load(
    files: list[Path | str],
    vocab_size: int,
    save_path: Path | str,
    min_frequency: int = 2,
) -> MiniTokenizer:
    path = Path(save_path)
    if path.is_file():
        return MiniTokenizer.load(path)
    return MiniTokenizer.train(files, vocab_size, save_path=path, min_frequency=min_frequency)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or inspect the MiniLLM BPE tokenizer")
    parser.add_argument("--files", nargs="+", type=Path, help="text files to train on")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--out", type=Path, default=Path("data/processed/tokenizer.json"))
    parser.add_argument("--roundtrip", type=str, default="", help="optional string to encode then decode")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.files:
        raise SystemExit("pass --files path/to/corpus.txt")
    tok = MiniTokenizer.train(args.files, args.vocab_size, save_path=args.out)
    print(f"saved tokenizer to {args.out} (vocab_size={tok.vocab_size})")
    sample = args.roundtrip or "Once upon a time, a small language model learned to talk."
    ids = tok.encode(sample, add_special_tokens=True)
    decoded = tok.decode(ids)
    print(f"encode ids ({len(ids)}): {ids[:32]}{'...' if len(ids) > 32 else ''}")
    print(f"decode: {decoded}")


if __name__ == "__main__":
    main()
