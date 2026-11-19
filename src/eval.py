"""Run a trained checkpoint over splits and save per-row logits (GPU step; analysis is src.report).

    uv run python -m src.eval --run runs/seed1 --split calibration,test,ood [--ckpt best] [--shard 0/2 --device cuda:1]

Writes runs/<id>/preds/<ckpt>/<split>[.shard-I-of-N].parquet with one row per record:
id, source, kind, n_options, is_english, hard, logits, target. Rows are evaluated in canonical option order
on the FULL split (never truncated); the English subset is a filter applied later in the report.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .config import TrainConfig
from .hub import restore_run
from .data.records import SPLITS, load_split
from .models.hrm_decision import MODEL_ID, MODEL_REVISION
from .predict import predict
from .train import default_model_factory


def preds_path(run_dir: Path, ckpt: str, split: str, shard: tuple[int, int] | None) -> Path:
    suffix = f".shard-{shard[0]}-of-{shard[1]}" if shard else ""
    return run_dir / "preds" / ckpt / f"{split}{suffix}.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best", choices=["best", "final"])
    ap.add_argument("--split", default="calibration,test,ood")
    ap.add_argument("--device")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--shard", help="I/N: evaluate rows I, I+N, ... (one shard per GPU)")
    ap.add_argument("--limit", type=int, help="first N rows of each split (dev sample)")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--hub-repo", help="<hf-user>/<repo>: fetch `--ckpt` + run.json from the Hub if runs/<id>/<ckpt> is missing")
    ap.add_argument("--restore-only", action="store_true",
                    help="only fetch from --hub-repo and exit (do this once before launching parallel --shard processes)")
    args = ap.parse_args()

    run_dir = Path(args.run)
    if args.hub_repo and not (run_dir / args.ckpt / "model.json").exists():
        if not restore_run(run_dir, args.hub_repo, items=(args.ckpt, "run.json")):
            raise SystemExit(f"no '{args.ckpt}' checkpoint for {run_dir.name} in {args.hub_repo}")
        print(f"[eval] restored {run_dir.name}/{args.ckpt} from {args.hub_repo}", flush=True)
    if args.restore_only:
        return
    cfg = TrainConfig(**json.loads((run_dir / "run.json").read_text())["config"])
    if args.device:
        cfg.device = args.device
    shard = tuple(map(int, args.shard.split("/"))) if args.shard else None
    splits = args.split.split(",")
    if bad := set(splits) - set(SPLITS):
        raise SystemExit(f"unknown splits: {bad}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = default_model_factory(cfg, tok).load_trained(run_dir / args.ckpt).eval()
    for split in splits:
        rows = load_split(split, args.data_dir)
        if shard:
            rows = rows[shard[0]::shard[1]]
        if args.limit:
            rows = rows[:args.limit]
        t = time.time()
        logits = predict(model, rows, tok, batch_size=args.batch_size or cfg.eval_batch_size, device=cfg.device,
                         dtype=cfg.dtype, progress=True)
        out = preds_path(run_dir, args.ckpt, split, shard)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "id": [r.id for r in rows], "source": [r.source for r in rows], "kind": [r.kind for r in rows],
            "n_options": [r.n_options for r in rows], "is_english": [r.is_english for r in rows],
            "hard": [r.hard for r in rows], "logits": [l.tolist() for l in logits],
            "target": [list(r.target) for r in rows],
        }).to_parquet(out)
        print(f"[eval] {split}: {len(rows)} rows in {time.time() - t:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
