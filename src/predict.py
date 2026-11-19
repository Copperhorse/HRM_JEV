"""Batched inference: rows -> per-row option logits (numpy, trimmed to each row's option count)."""
from __future__ import annotations

import contextlib

import numpy as np
import torch

from .data.prompt import collate
from .data.records import Record


def autocast_ctx(device: str, dtype: str):
    if dtype == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=torch.device(device).type,
                          dtype=torch.float16 if dtype == "fp16" else torch.bfloat16)


@torch.no_grad()
def predict(model, rows: list[Record], tokenizer, *, batch_size: int, device: str, dtype: str,
            progress: bool = False) -> list[np.ndarray]:
    """Canonical option order, no shuffling. Rows are batched by length for speed; output keeps input order."""
    was_training = model.training
    model.eval()
    order = sorted(range(len(rows)), key=lambda i: len(rows[i].question) + len(str(rows[i].state)))
    out: list[np.ndarray | None] = [None] * len(rows)
    starts = range(0, len(order), batch_size)
    if progress:
        from tqdm import tqdm

        starts = tqdm(starts, desc="predict", unit="batch")
    for s in starts:
        idx = order[s:s + batch_size]
        b = collate([rows[i] for i in idx], tokenizer).to(device)
        with autocast_ctx(device, dtype):
            logits = model(b.input_ids, b.attention_mask, b.token_type_ids, b.n_options).float().cpu().numpy()
        for j, i in enumerate(idx):
            out[i] = logits[j, :rows[i].n_options].astype(np.float64)
    model.train(was_training)
    return out  # type: ignore[return-value]
