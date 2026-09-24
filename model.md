# Mini LLM model

Decoder-only transformer: files, modules, and data flow for next-token prediction.

## How the pieces fit

```mermaid
flowchart TB
  subgraph entry["Entry points"]
    train["train.py"]
    generate["generate.py"]
    cfg["config.py"]
  end

  subgraph pkg["minillm package"]
    tok["tokenizer.py"]
    data["dataset.py"]
    utils["utils.py"]

    subgraph nn["minillm/model"]
      emb["embeddings.py"]
      attn["attention.py"]
      block["block.py"]
      gpt["transformer.py"]
    end
  end

  subgraph disk["On disk"]
    raw["data/raw"]
    proc["data/processed"]
    ckpt["checkpoints"]
  end

  cfg --> train
  cfg --> generate
  cfg --> gpt

  raw --> tok
  tok --> proc
  tok --> data
  proc --> data
  data --> train

  emb --> block
  attn --> block
  block --> gpt

  train --> gpt
  train --> utils
  train --> ckpt

  generate --> gpt
  generate --> tok
  generate --> utils
  ckpt --> generate
```

## Neural net: decoder stack

```mermaid
flowchart LR
  ids["input_ids (B, T)"]
  emb["Token + position embed"]
  b1["Block 1 Pre-LN"]
  b2["Block 2 ... N"]
  ln["Final LayerNorm"]
  head["LM head"]
  logits["logits (B, T, V)"]

  ids --> emb --> b1 --> b2 --> ln --> head --> logits
```

Each block:

```mermaid
flowchart TB
  x["x"]
  n1["LayerNorm"]
  a["Causal multi-head self-attention"]
  r1["x + dropout"]
  n2["LayerNorm"]
  ff["FFN: Linear → GELU → Linear"]
  r2["+ dropout"]
  y["x_out"]

  x --> n1 --> a --> r1
  x --> r1
  r1 --> n2 --> ff --> r2
  r1 --> r2 --> y
```

Causal attention (next-token / “next word” constraint):

```mermaid
flowchart LR
  x["x (B, T, C)"]
  qkv["Q, K, V projections"]
  scores["QKᵀ / √d"]
  mask["Causal mask: no attend to future"]
  sm["Softmax"]
  out["scores V → output proj"]

  x --> qkv --> scores --> mask --> sm --> out
```

## Training vs generation

```mermaid
sequenceDiagram
  participant User
  participant Train as train.py
  participant Tok as tokenizer.py
  participant DS as dataset.py
  participant M as transformer.py
  participant Disk as checkpoints

  User->>Train: start training
  Train->>Tok: train or load BPE
  Train->>DS: windows of tokens
  DS->>M: input_ids, labels shifted by 1
  M->>Train: logits
  Train->>Train: cross-entropy next-token loss
  Train->>Disk: save weights + config

  User->>Train: generate.py
  Note over Train,Disk: generate.py loads the same MiniGPT
```

```mermaid
flowchart TB
  prompt["prompt text"]
  enc["tokenizer.encode"]
  fwd["model: logits at last position"]
  sample["temperature / top-k sample"]
  dec["tokenizer.decode"]
  text["generated text"]

  prompt --> enc --> fwd --> sample
  sample -->|"append token, repeat"| fwd
  sample --> dec --> text
```

## Module responsibilities

| File | Role |
| --- | --- |
| `config.py` | `d_model`, heads, layers, vocab, seq len, paths, optimizer |
| `minillm/tokenizer.py` | BPE train/load, `encode`, `decode`, special tokens |
| `minillm/dataset.py` | Raw text → packed `(input_ids, labels)` batches |
| `minillm/model/embeddings.py` | Token + positional embeddings |
| `minillm/model/attention.py` | Causal multi-head self-attention |
| `minillm/model/block.py` | Pre-LN attention + FFN residual block |
| `minillm/model/transformer.py` | Full decoder + LM head; `forward` → logits |
| `minillm/utils.py` | Device, seed, save/load checkpoint |
| `train.py` | Loop: batch → loss → backward → checkpoint |
| `generate.py` | Autoregressive sampling from a prompt |

`transformer.py` is the only nn.Module the entrypoints construct. Attention and embeddings are implementation details of that stack.
