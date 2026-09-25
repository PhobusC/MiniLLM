"""Phase 0 smoke check: import the package, print v1 sizes, pick a device."""

from config import Config
from minillm import get_device


def main() -> None:
    cfg = Config()
    print("MiniLLM config (v1):")
    print(f"  d_model={cfg.d_model} n_heads={cfg.n_heads} n_layers={cfg.n_layers} d_ff={cfg.d_ff}")
    print(f"  vocab_size={cfg.vocab_size} max_seq_len={cfg.max_seq_len} dropout={cfg.dropout}")
    print(f"device: {get_device()}")


if __name__ == "__main__":
    main()
