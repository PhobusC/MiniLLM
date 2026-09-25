"""Autoregressive sampling with KV cache and left-padded batching."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from minillm.model import MiniGPT


def sample_next(logits: torch.Tensor, temperature: float, top_k: int, greedy: bool) -> torch.Tensor:
    """(B, V) logits -> (B,) next token ids."""
    if greedy:
        return logits.argmax(dim=-1)
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1)


def _left_pad(prompts: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(prompts), width), dtype=torch.long, device=device)
    for row, p in enumerate(prompts):
        ids[row, width - len(p) :] = torch.tensor(p, device=device)
        mask[row, width - len(p) :] = 1
    return ids, mask


def _positions(mask: torch.Tensor) -> torch.Tensor:
    return (mask.cumsum(-1) - 1).clamp_min(0)


@torch.no_grad()
def generate_steps(
    model: MiniGPT,
    prompts: list[list[int]],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    greedy: bool = False,
    eos_id: int | None = None,
    pad_id: int = 0,
    use_cache: bool = True,
) -> Iterator[torch.Tensor]:
    """Yield a (B,) tensor of next tokens per step; stops early once every row has emitted eos_id."""
    if not prompts or any(len(p) == 0 for p in prompts):
        raise ValueError("every prompt needs at least one token")
    device = next(model.parameters()).device
    window = model.cfg.max_seq_len
    ids, mask = _left_pad([p[-window:] for p in prompts], pad_id, device)
    padded = bool((mask == 0).any())
    done = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    cache = None

    for _ in range(max_new_tokens):
        if not use_cache:
            ids, mask = ids[:, -window:], mask[:, -window:]
            logits = model(ids, position_ids=_positions(mask), attention_mask=mask if padded else None)
        elif cache is None or ids.size(1) > window:
            # Prefill (first step, or after cropping a full window): process the whole prompt at once.
            ids, mask = ids[:, -window:], mask[:, -window:]
            logits, cache = model(ids, position_ids=_positions(mask),
                                  attention_mask=mask if padded else None, use_cache=True)
        else:
            # Decode: feed only the newest token; earlier keys/values come from the cache.
            logits, cache = model(ids[:, -1:], position_ids=_positions(mask)[:, -1:],
                                  attention_mask=mask if padded else None, kv_cache=cache, use_cache=True)

        next_ids = sample_next(logits[:, -1], temperature, top_k, greedy)
        yield next_ids
        if eos_id is not None:
            done |= next_ids == eos_id
            if bool(done.all()):
                return
        ids = torch.cat([ids, next_ids[:, None]], dim=1)
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)


def generate(
    model: MiniGPT,
    prompts: list[list[int]],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    greedy: bool = False,
    eos_id: int | None = None,
    pad_id: int = 0,
    use_cache: bool = True,
) -> list[list[int]]:
    """Return prompt + generated ids for each prompt (eos excluded)."""
    outputs = [list(p) for p in prompts]
    finished = [False] * len(prompts)
    for next_ids in generate_steps(model, prompts, max_new_tokens, temperature, top_k, greedy,
                                   eos_id, pad_id, use_cache):
        for row, tok in enumerate(next_ids.tolist()):
            if finished[row]:
                continue
            if tok == eos_id:
                finished[row] = True
            else:
                outputs[row].append(tok)
    return outputs


def stream(
    model: MiniGPT,
    prompt: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    greedy: bool = False,
    eos_id: int | None = None,
) -> Iterator[int]:
    """Yield generated token ids for a single prompt one at a time (for streaming UIs)."""
    for next_ids in generate_steps(model, [prompt], max_new_tokens, temperature, top_k, greedy, eos_id):
        tok = int(next_ids[0])
        if tok == eos_id:
            return
        yield tok
