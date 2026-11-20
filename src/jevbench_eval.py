"""Score a trained checkpoint on the 231 public JevBench items (Benchmark Heaven / fstandhartinger/jevbench).

    python -m src.jevbench_eval --jevbench-dir /content/jevbench --run runs/seed1 --ckpt best --temperature 1.0656 \
        [--hub-repo <user>/<repo>] [--device cuda:0] [--out runs/seed1/jevbench]

Reads `datasets/public/{easy,original,hard}.jsonl` from a clone of the JevBench repo as DATA ONLY; no JevBench code is
imported or executed. Request conversion mirrors Open-Jev's `compile_request` (choice option = "key: description",
score option = level text, noul = no/yes plus a "Yes means:" line), which is how our training data was built. The scoring
re-implements JevBench's definitions as read in the repo at commit 609a664 (`jevbench/scoring.py`, `metrics.py`,
`composite_v13.py`): correct = argmax label (ties -> smallest label) equals `expected`; hard ECE = top-label ECE, 10 equal-width
bins, over the hard tier; Intelligence = tier-weighted chance-corrected accuracy with the weights renormalised over the three
public tiers. An item that does not fit the 4,096-token context is NOT truncated: it counts wrong and is left out of ECE
(JevBench's rule for invalid answers).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .data.prompt import MAX_LENGTH, MAX_OPTIONS, render_prompt
from .data.records import Record

TIERS = {"easy": "easy", "original": "standard", "hard": "hard"}  # public file -> JevBench tier name
TIER_WEIGHTS = {"easy": 0.14, "standard": 0.28, "hard": 0.30}  # judge tier (0.28) has no public items -> renormalised away
# Option-count histograms of the full frozen tiers, from jevbench/composite_v13.py: chance = mean(1 / options).
TIER_OPTION_COUNTS = {"easy": {2: 18, 4: 13, 5: 41}, "standard": {2: 32, 4: 40, 5: 12, 6: 12},
                      "hard": {2: 77, 3: 26, 4: 73, 5: 38, 6: 6}}
TIER_CHANCE = {t: sum(c / k for k, c in h.items()) / sum(h.values()) for t, h in TIER_OPTION_COUNTS.items()}

# Published numbers for context. UNVERIFIED: copied from the LEV paper (arXiv 2610.20951, Tables 8-9), measured by its authors
# with their own harness on the same 231 public items, different training data, bf16 on an H200.
# (model, accuracy %, hard-tier accuracy %, hard-tier ECE, Intelligence)
PUBLISHED = [
    ("Kev-0.8B", 63.6, 36.9, 0.172, 49.3), ("Kev-4B", 75.8, 54.1, 0.112, 67.2),
    ("LEV-1.4B (depth 4)", 68.0, 44.1, 0.231, 55.7), ("LEV-2.6B (depth 4)", 76.2, 56.8, 0.192, 67.3),
    ("Qwen3-1.7B (dense control)", 63.6, 34.2, 0.253, 50.0), ("Qwen3.5-2B (dense control)", 68.8, 42.3, None, 57.5),
    ("Qwen3.5-4B (dense control)", 73.6, 51.4, None, 63.9), ("Jev 1.13.0 (closed API)", 86.6, 73.0, 0.086, 82.2),
]


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def load_items(jevbench_dir: str | Path) -> list[dict]:
    """The 231 public items with their tier attached (file order)."""
    out = []
    for fname, tier in TIERS.items():
        for line in (Path(jevbench_dir) / "datasets" / "public" / f"{fname}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append({**json.loads(line), "tier": tier})
    return out


def item_labels(item: dict) -> list[str]:
    """Answer labels in presented option order, checked against the item's own `labels`."""
    q = item["question"]
    if q["type"] == "choice":
        labels = list(q["criteria"])
    elif q["type"] == "score":
        labels = [str(i) for i in range(len(q["criteria"]))]
    else:
        labels = ["no", "yes"]
    if sorted(labels) != sorted(str(x) for x in item["labels"]):
        raise ValueError(f"{item['id']}: labels {item['labels']} do not match the converted options {labels}")
    return labels


def to_record(item: dict) -> Record:
    q, kind = item["question"], item["question"]["type"]
    question, criteria = _text(q["instructions"]), q.get("criteria")
    if kind == "choice":
        options = [k if v is None else f"{k}: {_text(v)}" for k, v in criteria.items()]
    elif kind == "score":
        options = [_text(level) for level in criteria]
    else:
        options = ["no", "yes"]
        for key, label in (("true", "Yes"), ("false", "No")):
            if criteria and key in criteria:
                question += f"\n{label} means: {_text(criteria[key])}"
    item_labels(item)  # validates
    return Record(id=item["id"], group_id=item["id"], source=f"jevbench-{item['tier']}", kind=kind, question=question,
                  options=tuple(options), target=tuple([0.0] * len(options)), state=item["state"], language=None,
                  target_basis=None)


def argmax_label(probs: dict[str, float]) -> str:
    """JevBench's rule: strict `>` over labels in sorted order, so ties go to the lexicographically smallest label."""
    best, best_p = None, -1.0
    for k in sorted(probs):
        if probs[k] > best_p:
            best, best_p = k, probs[k]
    return best


def ece_top_label(pairs: list[tuple[float, bool]], n_bins: int = 10) -> float | None:
    if not pairs:
        return None
    n, conf_sum, correct = [0] * n_bins, [0.0] * n_bins, [0] * n_bins
    for conf, ok in pairs:
        i = min(int(min(max(conf, 0.0), 1.0) * n_bins), n_bins - 1)
        n[i] += 1
        conf_sum[i] += conf
        correct[i] += bool(ok)
    total = sum(n)
    return sum(n[i] / total * abs(correct[i] / n[i] - conf_sum[i] / n[i]) for i in range(n_bins) if n[i])


def intelligence(acc_by_tier: dict[str, float]) -> float | None:
    score = weight = 0.0
    for tier, w in TIER_WEIGHTS.items():
        if acc_by_tier.get(tier) is not None:
            score += w * min(100.0, max(0.0, 100 * (acc_by_tier[tier] - TIER_CHANCE[tier]) / (1 - TIER_CHANCE[tier])))
            weight += w
    return score / weight if weight else None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def score_rows(items: list[dict], logits: list[np.ndarray | None], temperature: float) -> list[dict]:
    """Per-item results. `logits[i] is None` = the item did not fit the context (counts wrong, no confidence)."""
    rows = []
    for item, lg in zip(items, logits):
        labels = item_labels(item)
        expected = str(item["expected"])
        base = {"id": item["id"], "tier": item["tier"], "kind": item["question"]["type"], "expected": expected}
        if lg is None:
            rows.append({**base, "valid": False, "predicted": None, "correct": False, "confidence": None, "probs": None})
            continue
        z = np.asarray(lg, dtype=np.float64) / temperature
        p = np.exp(z - z.max())
        p /= p.sum()
        probs = dict(zip(labels, p.tolist()))
        pred = argmax_label(probs)
        rows.append({**base, "valid": True, "predicted": pred, "correct": pred == expected, "confidence": max(probs.values()),
                     "probs": probs})
    return rows


def summarize(rows: list[dict]) -> dict:
    out: dict = {"n": len(rows), "invalid_over_context": sum(not r["valid"] for r in rows)}
    acc = {}
    for tier in TIER_WEIGHTS:
        rs = [r for r in rows if r["tier"] == tier]
        acc[tier] = sum(r["correct"] for r in rs) / len(rs) if rs else None
        out[f"n_{tier}"] = len(rs)
    k = sum(r["correct"] for r in rows)
    out["accuracy"] = k / len(rows)
    out["accuracy_ci95"] = wilson(k, len(rows))
    out["tier_accuracy"] = acc
    out["intelligence"] = intelligence(acc)
    out["hard_ece"] = ece_top_label([(r["confidence"], r["correct"]) for r in rows if r["tier"] == "hard" and r["confidence"] is not None])
    out["by_kind"] = {kd: sum(r["correct"] for r in rows if r["kind"] == kd) / max(1, sum(r["kind"] == kd for r in rows))
                      for kd in ("choice", "noul", "score")}
    return out


def evaluate_items(model, tokenizer, items: list[dict], *, device: str, dtype: str, long_tokens: int = 1000) -> list[np.ndarray | None]:
    """Option logits per item in presented order; None where the prompt exceeds the context window (never truncated)."""
    from .predict import predict

    recs = [to_record(i) for i in items]
    lens = [len(x) for x in tokenizer([render_prompt(r) for r in recs], add_special_tokens=False)["input_ids"]]
    out: list[np.ndarray | None] = [None] * len(recs)
    for i, r in enumerate(recs):
        if r.n_options > MAX_OPTIONS:
            raise ValueError(f"{r.id}: {r.n_options} options > {MAX_OPTIONS}")
    for sel, bs in (([i for i, n in enumerate(lens) if n <= long_tokens], 8),
                    ([i for i, n in enumerate(lens) if long_tokens < n <= MAX_LENGTH], 1)):
        if sel:
            for i, lg in zip(sel, predict(model, [recs[i] for i in sel], tokenizer, batch_size=bs, device=device, dtype=dtype,
                                          progress=True)):
                out[i] = lg
    return out


def markdown(res_raw: dict, res_t: dict, temperature: float, meta: str) -> str:
    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}"

    def row(name, r):  # a tier is None only when --limit cut it off
        t = r["tier_accuracy"]
        ece = "n/a" if r["hard_ece"] is None else f"{r['hard_ece']:.3f}"
        return (f"| {name} | {pct(r['accuracy'])} | {pct(t['easy'])} | {pct(t['standard'])} | {pct(t['hard'])} | {ece} | "
                f"{'n/a' if r['intelligence'] is None else format(r['intelligence'], '.1f')} |")
    lo, hi = res_t["accuracy_ci95"]
    lines = [f"{meta}", "",
             f"231 public JevBench items (48 easy / 72 standard / 111 hard); {res_t['invalid_over_context']} item(s) over the 4,096-token "
             f"context (counted wrong, not truncated). Accuracy 95% CI (Wilson) {100 * lo:.1f}-{100 * hi:.1f}%.", "",
             "| system | acc % | easy | standard | hard acc % | hard ECE | Intelligence |", "|---|---:|---:|---:|---:|---:|---:|",
             row("**this model, T=1**", res_raw), row(f"**this model, T={temperature:.4f}** (Open-Jev calibration)", res_t)]
    for name, a, h, e, i in PUBLISHED:
        lines.append(f"| {name} (published, UNVERIFIED) | {a:.1f} | | | {h:.1f} | {'' if e is None else f'{e:.3f}'} | {i:.1f} |")
    lines += ["", "Published rows: LEV paper (arXiv 2610.20951) Tables 8-9, their harness, other training data (Kev data / closed), not "
              "matched to ours. Intelligence uses the three public tiers only. 231 items give roughly +-6 points of uncertainty."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jevbench-dir", required=True, help="clone of github.com/fstandhartinger/jevbench (data files only are read)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best", choices=["best", "final"])
    ap.add_argument("--temperature", type=float, default=1.0, help="from our calibration split (seed1 1.0656, seed2 1.1170)")
    ap.add_argument("--device")
    ap.add_argument("--hub-repo", help="<user>/<repo>: fetch the checkpoint from the Hub if it is not on disk")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from .config import TrainConfig
    from .hub import restore_run
    from .models.hrm_decision import MODEL_ID, MODEL_REVISION
    from .train import default_model_factory

    run_dir = Path(args.run)
    if args.hub_repo and not (run_dir / args.ckpt / "model.json").exists():
        if not restore_run(run_dir, args.hub_repo, items=(args.ckpt, "run.json")):
            raise SystemExit(f"no '{args.ckpt}' checkpoint for {run_dir.name} in {args.hub_repo}")
    cfg = TrainConfig(**json.loads((run_dir / "run.json").read_text())["config"])
    if args.device:
        cfg.device = args.device
    items = load_items(args.jevbench_dir)
    if args.limit:
        items = items[:args.limit]
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = default_model_factory(cfg, tok).load_trained(run_dir / args.ckpt).eval()
    logits = evaluate_items(model, tok, items, device=cfg.device, dtype=cfg.dtype)
    rows_t = score_rows(items, logits, args.temperature)
    res_raw, res_t = summarize(score_rows(items, logits, 1.0)), summarize(rows_t)
    out = Path(args.out or run_dir / "jevbench" / args.ckpt)
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_t), encoding="utf-8")
    (out / "results.json").write_text(json.dumps({"run": run_dir.name, "ckpt": args.ckpt, "temperature": args.temperature,
                                                  "raw": res_raw, "calibrated": res_t}, indent=1), encoding="utf-8")
    md = markdown(res_raw, res_t, args.temperature, f"Run `{run_dir.name}` ({args.ckpt}), fp16 {cfg.device}")
    (out / "results.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
