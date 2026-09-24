# Mini LLM Plan

Build a small **decoder-only transformer** that learns to predict the next token (word or subword) and can generate text autoregressively.

This is a from-scratch educational model: small enough to train on a laptop or single GPU, but structured like GPT-style language models.

## Goal

Given a sequence of tokens \(x_1, x_2, \ldots, x_t\), the model outputs a probability distribution over the vocabulary for \(x_{t+1}\). Training minimizes cross-entropy of that prediction. At inference, sample (or greedily pick) the next token, append it, and repeat.

## Design choices

| Choice | Decision | Why |
| --- | --- | --- |
| Architecture | Decoder-only transformer (causal self-attention) | Matches GPT; next-token prediction needs no encoder |
| Tokenization | Byte-pair encoding (BPE) with a fallback word/character path | Predicts “next word” at a useful granularity without a huge vocab |
| Framework | PyTorch | Standard for this architecture; easy to inspect tensors |
| Size (v1) | ~5–20M parameters | Trainable on a single GPU or Apple Silicon in hours, not days |
| Data (v1) | Small public text corpus (e.g. TinyStories, WikiText-2, or a local `.txt`) | Enough signal to learn grammar and local coherence |
| Context length | 128–256 tokens first; 512 later | Keeps memory and compile time small |

Default v1 hyperparameters (tunable in `config.py`):

- `d_model`: 256
- `n_heads`: 8
- `n_layers`: 6
- `d_ff`: 1024
- `vocab_size`: 8k–16k (BPE)
- `max_seq_len`: 256
- `dropout`: 0.1

## Target layout

```
MiniLLM/
  plan.md
  model.md
  requirements.txt
  config.py                 # all hyperparameters and paths
  train.py                  # training entrypoint
  generate.py               # load checkpoint and sample text
  minillm/
    __init__.py
    tokenizer.py            # train/load BPE; encode/decode
    dataset.py              # text → token ids → packed batches
    utils.py                # seed, checkpoint I/O, device
    model/
      __init__.py
      embeddings.py         # token + positional embeddings
      attention.py          # causal multi-head self-attention
      block.py              # Pre-LN transformer block (attn + FFN)
      transformer.py        # stack + LM head
  data/
    raw/                    # source text
    processed/              # tokenized shards / tokenizer files
  checkpoints/
```

## Model internals

Each layer is a standard Pre-LN decoder block:

1. **Token embedding** + **learned positional embedding** (or rotary later).
2. **Causal multi-head self-attention**: query/key/value projections, scaled dot-product, mask so position \(i\) cannot attend to \(j > i\).
3. Residual + dropout.
4. **Feed-forward**: `Linear → GELU → Linear`, residual + dropout.
5. After the last block: layer-norm, then **LM head** (`d_model → vocab_size`).
6. Loss: `CrossEntropyLoss` on shifted labels (predict token \(t+1\) from tokens \(1..t\)). Weight tying between token embedding and LM head is optional for v1.

The causal mask is what makes this decoder-only. There is no cross-attention and no encoder.

## Work in phases

### Phase 0 — Project skeleton

- Add `requirements.txt` (`torch`, `numpy`, `tokenizers` or a tiny custom BPE, `tqdm`).
- Add `config.py` as a single dataclass / namespace so train and generate share sizes.
- Create empty packages under `minillm/` matching the layout above.
- Decide device: CUDA if present, else MPS, else CPU.

**Done when:** `python -c "import minillm"` works and config prints the v1 sizes.

### Phase 1 — Tokenizer and data

- Train a BPE tokenizer on the raw corpus; persist `tokenizer.json` (or vocab + merges).
- Special tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>`.
- `encode` / `decode` round-trip on sample paragraphs.
- `TextDataset`: read text, tokenize, pack into fixed-length windows of `max_seq_len + 1` (inputs `x[:-1]`, labels `x[1:]`).
- `DataLoader` with padding only if packing is incomplete.

**Done when:** a batch is `(B, T)` int64 tensors, labels are the next-token shift, and decode(encode(text)) is readable.

### Phase 2 — Model modules

Implement bottom-up and unit-test shapes:

1. `TokenPositionalEmbedding`: `(B, T) → (B, T, C)`
2. `CausalSelfAttention`: `(B, T, C) → (B, T, C)`; verify the mask zeros future positions (attention weights above the diagonal are 0).
3. `FeedForward` + `TransformerBlock`
4. `MiniGPT` / `DecoderLM`: embedding → N blocks → norm → LM head → logits `(B, T, V)`

Keep the public surface small: `forward(input_ids) → logits`. No generation logic inside the module.

**Done when:** a random batch produces logits of shape `(B, T, vocab_size)` and a single forward+backward pass succeeds on CPU.

### Phase 3 — Training loop

`train.py`:

- Load config, tokenizer, dataset, model, AdamW.
- For each step: forward → loss → backward → clip grad → step.
- Log loss (and tokens/sec) periodically.
- Save checkpoints: `model`, `optimizer`, `config`, `step`.
- Optional: cosine LR decay with warmup; optional validation split (loss only).

Start with a tiny overfit test: 1–2 files, expect train loss to drop sharply. Then scale to the full corpus.

**Done when:** train loss trends down on the overfit set, a checkpoint reloads, and parameter count is printed.

### Phase 4 — Generation

`generate.py`:

- Load checkpoint + tokenizer.
- Prompt → token ids.
- Loop: forward last `T` tokens (or KV-cache later) → logits at last position → softmax with temperature / top-k → sample → append until `max_new_tokens` or `<eos>`.

v1 generation can re-run the full sequence each step (simpler). KV-cache is a later optimization.

**Done when:** a prompt produces coherent-ish tokens (even if not fluent at tiny scale).

### Phase 5 — Sanity and small quality pass

- Parameter count vs intended size.
- Loss curve saved (simple CSV or stdout).
- Compare greedy vs sampled output.
- Document how to train and sample in comments at the top of `train.py` / `generate.py` (no extra README unless we add one later).

## Training recipe (v1)

1. Put text in `data/raw/`.
2. `python train.py --data data/raw --out checkpoints/v1`.
3. Tokenizer trains once and is reused.
4. Run until train loss plateaus or a step budget (e.g. 5k–20k steps depending on hardware).
5. `python generate.py --ckpt checkpoints/v1 --prompt "Once upon a time"`.

## Later (not required for v1)

- Rotary embeddings (RoPE) instead of learned absolute positions.
- KV cache for faster sampling.
- Mixed precision (`torch.amp`).
- Proper eval (validation perplexity).
- Larger data and a 50M–100M “still mini” scale-up.

## Risks and how to keep it mini

- **Vocab too large** → slow softmax and sparse counts. Cap BPE around 8k–16k.
- **Context too long** → attention is \(O(T^2)\). Stay at 256 until it trains.
- **No overfit test** → hard to tell bugs from undertraining. Always overfit a tiny file first.
- **Leaky causal mask** → model cheats by seeing the future; loss looks great, generation is garbage. Test the mask explicitly.

## Implementation order (checklist)

- [ ] Skeleton + `config.py` + dependencies
- [ ] Tokenizer train/load + encode/decode
- [ ] Dataset packing and next-token labels
- [ ] Embeddings, causal attention, block, full decoder
- [ ] Train loop + checkpoints
- [ ] Generate from prompt
- [ ] Overfit test, then real data run
