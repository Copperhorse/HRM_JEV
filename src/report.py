"""CPU-only analysis of saved predictions: temperature fit + metric tables.

    uv run python -m src.report --run runs/seed1 [--ckpt best]

Temperature is fitted ONCE on the full calibration split and applied unchanged to every other split/subset.
Writes runs/<id>/preds/<ckpt>/results.json and results.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import evaluate, fit_temperature

KINDS = ("all", "choice", "noul", "score")
CAVEAT = ("Evaluated on the public `release-v2-redistributable` projection (test 10,356 / OOD 15,701 rows). "
          "Open-Jev-2B's card numbers use the original frozen mixture (10,532 / 15,920 rows, including the "
          "omitted `wikispeedia-v1` rows), so card numbers are not directly comparable. Hard accuracy excludes "
          "soft-target rows.")


def load_preds(pred_dir: Path, split: str) -> pd.DataFrame:
    files = sorted(pred_dir.glob(f"{split}.parquet")) + sorted(pred_dir.glob(f"{split}.shard-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no predictions for {split} in {pred_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if df.id.duplicated().any():
        raise ValueError(f"{split}: duplicate ids across shards")
    return df


def arrays(df: pd.DataFrame) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return [np.asarray(x) for x in df.logits], [np.asarray(x) for x in df.target]


def build_report(pred_dir: Path, splits=("test", "ood")) -> dict:
    cal = load_preds(pred_dir, "calibration")
    temperature = fit_temperature(*arrays(cal))
    rows = []
    for split in splits:
        df = load_preds(pred_dir, split)
        for subset, sub in (("english", df[df.is_english]), ("full", df)):
            for kind in KINDS:
                part = sub if kind == "all" else sub[sub.kind == kind]
                if not len(part):
                    continue
                lg, tg = arrays(part)
                raw, cal_m = evaluate(lg, tg), evaluate(lg, tg, temperature=temperature)
                rows.append({"split": split, "subset": subset, "kind": kind, "n": raw["n"], "hard_n": raw["hard_n"],
                             "hard_acc": raw["hard_acc"], "expected_acc": raw["expected_acc"],
                             "nll": cal_m["nll"], "brier": cal_m["brier"], "ece_raw": raw["ece"], "ece": cal_m["ece"]})
    return {"temperature": temperature, "calibration_rows": len(cal), "caveat": CAVEAT, "rows": rows}


def to_markdown(rep: dict) -> str:
    f = lambda x, d=4: "-" if x is None else f"{x:.{d}f}"
    lines = [f"Temperature {rep['temperature']:.4f} (fitted on {rep['calibration_rows']} calibration rows). "
             "NLL/Brier/ECE use that temperature; `ece_raw` is T=1.", "",
             "| split | subset | kind | n | hard n | hard acc | exp acc | NLL | Brier | ECE | ECE raw |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rep["rows"]:
        lines.append(f"| {r['split']} | {r['subset']} | {r['kind']} | {r['n']} | {r['hard_n']} | {f(r['hard_acc'])} | "
                     f"{f(r['expected_acc'])} | {f(r['nll'])} | {f(r['brier'])} | {f(r['ece'])} | {f(r['ece_raw'])} |")
    return "\n".join(lines) + f"\n\n{rep['caveat']}\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--splits", default="test,ood")
    args = ap.parse_args()
    pred_dir = Path(args.run) / "preds" / args.ckpt
    rep = build_report(pred_dir, tuple(args.splits.split(",")))
    (pred_dir / "results.json").write_text(json.dumps(rep, indent=2))
    md = to_markdown(rep)
    (pred_dir / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
