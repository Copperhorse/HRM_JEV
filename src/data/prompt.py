"""PrefixLM prompt rendering, tokenisation and collation for HRM-Text.

One forward pass per row: all options are enumerated as letters and the model's letter logits at
the last prompt position are read out. No truncation, ever (leaderboards forbid it).
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass

import torch

from .records import Record, render_state

MAX_OPTIONS = 16  # Open-Jev release-v2 maximum (painting-geometry); letters A-P
LETTERS = string.ascii_uppercase[:MAX_OPTIONS]
# README: `direct` condition == <|object_ref_start|>; prompt wrapped in <|im_start|> ... <|im_end|>.
CONDITION_DIRECT = "<|object_ref_start|>"
MAX_LENGTH = 4096  # HRM-Text context window

_INSTRUCTION = {
    "choice": "Choose the best option.",
    "score": "Choose the level that best fits, in order from the first level to the last.",
    "noul": "Is the answer to this question yes?",
}


def render_prompt(record: Record, order: list[int] | None = None) -> str:
    """Prompt text for `record`, with options listed in `order` (original indices; default canonical)."""
    if record.n_options > MAX_OPTIONS:
        raise ValueError(f"{record.id}: {record.n_options} options > {MAX_OPTIONS}")
    order = list(range(record.n_options)) if order is None else order
    options = "\n".join(f"{LETTERS[i]}. {record.options[j]}" for i, j in enumerate(order))
    body = (f"Context:\n{render_state(record.state)}\n\nQuestion: {record.question}\n"
            f"{_INSTRUCTION[record.kind]}\nOptions:\n{options}\nAnswer:")
    return f"<|im_start|>{CONDITION_DIRECT}{body}<|im_end|>"


def choose_order(record: Record, rng: random.Random | None) -> list[int]:
    """Canonical order, or (training only) a random permutation for `choice` rows.

    Only `choice` is shuffled: score levels are ordinal and noul is [no, yes]. Shuffling removes
    answer-position shortcuts (e.g. tile_platformer: 89% of answers at index 4).
    """
    order = list(range(record.n_options))
    if rng is not None and record.kind == "choice":
        rng.shuffle(order)
    return order


@dataclass
class Batch:
    input_ids: torch.Tensor  # [B, T] right-padded
    attention_mask: torch.Tensor  # [B, T]
    token_type_ids: torch.Tensor  # [B, T] 1 over the whole prompt (one bidirectional prefix block), 0 on pad
    n_options: torch.Tensor  # [B]
    target: torch.Tensor  # [B, MAX_OPTIONS] in the *presented* order, zero beyond n_options
    order: list[list[int]]  # presented position -> original option index

    def to(self, device) -> "Batch":
        return Batch(self.input_ids.to(device), self.attention_mask.to(device), self.token_type_ids.to(device),
                     self.n_options.to(device), self.target.to(device), self.order)


def collate(records: list[Record], tokenizer, *, rng: random.Random | None = None,
            max_length: int = MAX_LENGTH) -> Batch:
    orders = [choose_order(r, rng) for r in records]
    prompts = [render_prompt(r, o) for r, o in zip(records, orders)]
    ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    longest = max(map(len, ids))
    if longest > max_length:
        raise ValueError(f"prompt of {longest} tokens exceeds max_length={max_length}; refusing to truncate")
    pad = tokenizer.pad_token_id
    input_ids = torch.full((len(ids), longest), pad, dtype=torch.long)
    mask = torch.zeros((len(ids), longest), dtype=torch.long)
    for i, row in enumerate(ids):
        input_ids[i, : len(row)] = torch.tensor(row)
        mask[i, : len(row)] = 1
    target = torch.zeros((len(records), MAX_OPTIONS))
    for i, (r, o) in enumerate(zip(records, orders)):
        target[i, : r.n_options] = torch.tensor([r.target[j] for j in o])
    return Batch(input_ids, mask, mask.clone(), torch.tensor([r.n_options for r in records]), target, orders)


def letter_token_ids(tokenizer) -> list[int]:
    """Token ids of the bare letters A-P (the response begins right after <|im_end|>, no space)."""
    ids = [tokenizer.encode(c, add_special_tokens=False) for c in LETTERS]
    if any(len(i) != 1 for i in ids):
        raise ValueError("letters must be single tokens")
    return [i[0] for i in ids]
