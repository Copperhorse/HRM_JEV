"""LoRA + readout training of HRM-Text-1B on Open-Jev decisions.

    uv run python -m src.train --config configs/seed1.yaml [--set max_steps=50 --set run_id=probe]

Resumable: the batch order is a pure function of (seed, epoch), option shuffling is seeded per
micro-batch, and checkpoints hold adapter + readout + optimizer + scaler + step. Re-running the same
command continues from the last checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import TrainConfig
from .data.prompt import collate, letter_token_ids
from .data.records import DATASET_ID, DATASET_REVISION, Record, load_split
from .metrics import evaluate
from .models.hrm_decision import MODEL_ID, MODEL_REVISION, HrmDecisionModel, decision_loss, load_hrm
from .predict import autocast_ctx, predict
from .tracking import Tracker


def default_model_factory(cfg: TrainConfig, tokenizer) -> HrmDecisionModel:
    return HrmDecisionModel(
        load_hrm(cfg.dtype, cfg.device), letter_token_ids(tokenizer), lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, grad_checkpointing=cfg.grad_checkpointing,
        l_bp_cycles=cfg.l_bp_cycles)


def make_micro_batches(rows: list[Record], cfg: TrainConfig) -> list[list[int]]:
    """All micro-batches (lists of row indices) for every epoch; a pure function of (seed, rows, cfg).

    Per epoch: shuffle, sort within chunks of 64 batches by prompt length (less padding), then shuffle
    the batch order. A trailing partial micro-batch is dropped so every step has full-size batches.
    """
    size = [len(r.question) + len(str(r.state)) for r in rows]
    out: list[list[int]] = []
    for epoch in range(math.ceil(cfg.epochs)):
        rng = random.Random(cfg.seed * 1_000_003 + epoch)
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        chunk = cfg.batch_size * 64
        batches = []
        for s in range(0, len(idx), chunk):
            part = sorted(idx[s:s + chunk], key=size.__getitem__)
            batches += [part[i:i + cfg.batch_size] for i in range(0, len(part), cfg.batch_size)]
        batches = [b for b in batches if len(b) == cfg.batch_size]
        rng.shuffle(batches)
        if epoch + 1 > cfg.epochs:  # fractional final epoch
            batches = batches[: int(len(batches) * (cfg.epochs - epoch))]
        out += batches
    return out


def lr_factor(step: int, total: int, warmup: int, floor: float = 0.1) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def save_checkpoint(run_dir: Path, model, opt, scaler, state: dict) -> None:
    tmp, final, old = run_dir / "ckpt.tmp", run_dir / "ckpt", run_dir / "ckpt.old"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save(tmp)
    torch.save(opt.state_dict(), tmp / "optimizer.pt")
    torch.save(scaler.state_dict(), tmp / "scaler.pt")
    (tmp / "state.json").write_text(json.dumps(state))
    shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        os.replace(final, old)
    os.replace(tmp, final)
    shutil.rmtree(old, ignore_errors=True)


def train(cfg: TrainConfig, *, model_factory=default_model_factory, tokenizer=None, resume: bool = True,
          _crash_after: int | None = None) -> dict:  # _crash_after: test hook simulating a crash
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    rows = load_split("train", cfg.data_dir, english_only=cfg.english_only)
    if cfg.max_rows:
        rows = random.Random(cfg.seed).sample(rows, min(cfg.max_rows, len(rows)))
    val_all = load_split("validation", cfg.data_dir, english_only=True)
    val = random.Random(0).sample(val_all, min(cfg.eval_rows, len(val_all)))  # fixed across runs
    micro = make_micro_batches(rows, cfg)
    total_steps = len(micro) // cfg.accumulation
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)
    micro = micro[: total_steps * cfg.accumulation]

    model = model_factory(cfg, tokenizer)
    groups = model.trainable_parameters()
    opt = torch.optim.AdamW(
        [{"params": groups["lora"], "lr": cfg.lr, "base_lr": cfg.lr},
         {"params": groups["readout"], "lr": cfg.readout_lr, "base_lr": cfg.readout_lr}],
        betas=(0.9, 0.999), weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(torch.device(cfg.device).type, enabled=cfg.dtype == "fp16")
    all_params = groups["lora"] + groups["readout"]

    step, best_nll = 0, float("inf")
    ckpt = run_dir / "ckpt"
    if resume and not ckpt.exists() and (run_dir / "ckpt.old").exists():
        os.replace(run_dir / "ckpt.old", ckpt)  # crashed between the two renames in save_checkpoint
    if resume and (ckpt / "state.json").exists():
        state = json.loads((ckpt / "state.json").read_text())
        model.load_trained(ckpt)
        opt.load_state_dict(torch.load(ckpt / "optimizer.pt", map_location="cpu", weights_only=True))
        scaler.load_state_dict(torch.load(ckpt / "scaler.pt", map_location="cpu", weights_only=True))
        step, best_nll = state["step"], state["best_nll"]
        print(f"[train] resumed from step {step}", flush=True)
    elif (run_dir / "metrics.jsonl").exists():
        (run_dir / "metrics.jsonl").unlink()

    (run_dir / "run.json").write_text(json.dumps({
        "config": cfg.to_dict(), "total_steps": total_steps, "train_rows": len(rows), "val_rows": len(val),
        "dataset": [DATASET_ID, DATASET_REVISION], "model": [MODEL_ID, MODEL_REVISION], "git_commit": _git_commit(),
        "torch": torch.__version__, "python": platform.python_version(),
        "trainable_params": {k: sum(p.numel() for p in v) for k, v in groups.items()},
    }, indent=2))
    log = (run_dir / "metrics.jsonl").open("a")
    tracker = Tracker(cfg, run_dir)

    def log_row(row: dict) -> None:
        log.write(json.dumps(row) + "\n")
        log.flush()
        tracker.log(row)
        print("[train]", json.dumps(row), flush=True)

    def run_eval() -> None:
        nonlocal best_nll
        t = time.time()
        m = evaluate(predict(model, val, tokenizer, batch_size=cfg.eval_batch_size, device=cfg.device, dtype=cfg.dtype),
                     [np.asarray(r.target) for r in val])
        log_row({"step": step, "val": {k: m[k] for k in ("n", "hard_acc", "expected_acc", "nll", "brier", "ece")},
                 "eval_s": round(time.time() - t, 1)})
        if m["nll"] < best_nll:
            best_nll = m["nll"]
            model.save(run_dir / "best")

    model.train()
    t_log, tok_log, loss_log, skipped = time.time(), 0, [], 0
    while step < total_steps:
        factor = lr_factor(step, total_steps, cfg.warmup_steps)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * factor
        opt.zero_grad(set_to_none=True)
        for k in range(cfg.accumulation):
            midx = step * cfg.accumulation + k
            rng = random.Random(cfg.seed * 1_000_003 + 7919 * (midx + 1)) if cfg.shuffle_choice_options else None
            b = collate([rows[i] for i in micro[midx]], tokenizer, rng=rng).to(cfg.device)
            with autocast_ctx(cfg.device, cfg.dtype):
                logits = model(b.input_ids, b.attention_mask, b.token_type_ids, b.n_options)
            loss = decision_loss(logits, b.target, cfg.brier_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step} (micro-batch {k}); run invalid")
            scaler.scale(loss / cfg.accumulation).backward()
            loss_log.append(loss.item())
            tok_log += int(b.attention_mask.sum())
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skipped += scaler.get_scale() < scale_before  # fp16 overflow -> step skipped by the scaler
        step += 1

        if step % cfg.log_every == 0 or step == total_steps:
            dt = time.time() - t_log
            log_row({"step": step, "loss": round(float(np.mean(loss_log)), 5), "grad_norm": round(float(gnorm), 4),
                     "lr": opt.param_groups[0]["lr"], "tok_per_s": round(tok_log / dt), "skipped_steps": int(skipped),
                     "loss_scale": scaler.get_scale() if scaler.is_enabled() else None,
                     "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None})
            t_log, tok_log, loss_log = time.time(), 0, []
        if step % cfg.eval_every == 0 and step != total_steps:
            run_eval()
        if step % cfg.save_every == 0 and step != total_steps:
            save_checkpoint(run_dir, model, opt, scaler, {"step": step, "best_nll": best_nll})
        if _crash_after is not None and step == _crash_after:
            raise RuntimeError("simulated crash")

    run_eval()
    model.save(run_dir / "final")
    save_checkpoint(run_dir, model, opt, scaler, {"step": step, "best_nll": best_nll})
    log.close()
    tracker.finish({"best_val_nll": best_nll, "steps": step})
    return {"steps": step, "best_val_nll": best_nll, "run_dir": str(run_dir)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key (YAML value)")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    for kv in args.set:
        key, _, value = kv.partition("=")
        if not hasattr(cfg, key):
            raise SystemExit(f"unknown config key: {key}")
        setattr(cfg, key, yaml.safe_load(value))
    print(json.dumps(train(cfg, resume=not args.no_resume)))


if __name__ == "__main__":
    main()
