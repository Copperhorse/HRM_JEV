"""Phase 0 item 5: real HRM-Text-1B forward pass with the PrefixLM format. Run on a GPU box (Kaggle).

    python -m src.probe_forward --device cuda:0 --dtype fp16 [--rows 32]

Checks: loads at the pinned revision; logits finite; a row's logits do not depend on batch padding; the
PrefixLM mask changes the logits vs pure causal (i.e. token_type_ids is honoured); zero-shot accuracy vs chance.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from .data.prompt import collate, letter_token_ids
from .data.records import load_split
from .models.hrm_decision import MODEL_ID, MODEL_REVISION, HrmDecisionModel, load_hrm
from .predict import autocast_ctx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--data-dir", default="data/raw")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    t = time.time()
    lm = load_hrm(args.dtype, args.device)
    print(f"loaded in {time.time() - t:.0f}s; params {sum(p.numel() for p in lm.parameters()) / 1e9:.3f}B; "
          f"L_bp_cycles {lm.model.L_bp_cycles_padded}")
    model = HrmDecisionModel(lm, letter_token_ids(tok), lora_rank=16).eval()
    rows = load_split("test", args.data_dir)[:: 10356 // args.rows][: args.rows]
    with torch.no_grad(), autocast_ctx(args.device, args.dtype):
        b = collate(rows, tok).to(args.device)
        t = time.time()
        lg = model(b.input_ids, b.attention_mask, b.token_type_ids, b.n_options).float()
        print(f"{len(rows)} rows, padded seq {b.input_ids.shape[1]}: {time.time() - t:.1f}s")
        valid = lg > -1e8
        print("finite:", bool(torch.isfinite(lg[valid]).all()), "| logit range", lg[valid].min().item(), lg[valid].max().item())
        lens = b.attention_mask.sum(-1)
        i = int(lens.argmin())  # the shortest row is guaranteed to be padded inside the batch
        s = collate([rows[i]], tok).to(args.device)
        one = model(s.input_ids, s.attention_mask, s.token_type_ids, s.n_options).float()
        n = rows[i].n_options
        print(f"padding invariance (row {i}: {int(lens[i])} tokens, batch padded to {b.input_ids.shape[1]}) max|diff|:",
              (lg[i, :n] - one[0, :n]).abs().max().item(), "(fp16 noise ~1e-2 is expected; fp32 should be ~1e-4)")
        causal = model(b.input_ids, b.attention_mask, torch.zeros_like(b.token_type_ids), b.n_options).float()
        print("max|prefix - causal| logits:", (lg - causal)[valid].abs().max().item(), "(must be > 0)")
        acc = (lg.argmax(-1) == b.target.argmax(-1)).float().mean().item()
        print(f"zero-shot argmax acc {acc:.3f} vs chance ~{np.mean([1 / r.n_options for r in rows]):.3f} "
              "(readout is initialised from lm_head letter rows; near chance is fine, this is an untrained probe)")


if __name__ == "__main__":
    main()
