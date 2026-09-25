"""Root config module. Canonical definition lives in minillm.config."""

from minillm.config import Config, default_config

__all__ = ["Config", "default_config"]


if __name__ == "__main__":
    cfg = Config()
    print(cfg)
    print(
        "v1 sizes:",
        {
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "d_ff": cfg.d_ff,
            "vocab_size": cfg.vocab_size,
            "max_seq_len": cfg.max_seq_len,
            "dropout": cfg.dropout,
        },
    )
