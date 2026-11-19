"""Phase 0 data checks (items 1-4). Run: uv run python -m src.data.inspect [--samples 3]

Prints markdown suitable for pasting into notes/phase0.md.
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from .prompt import MAX_LENGTH, render_prompt
from .records import EXPECTED_SIZES, SPLITS, load_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--samples", type=int, default=0, help="print N full records per kind (train split)")
    ap.add_argument("--tokenizer", default="sapientinc/HRM-Text-1B")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print("## Split sizes (vs card)")
    data = {s: load_split(s, args.data_dir) for s in SPLITS}  # raises on a size mismatch
    for s in SPLITS:
        print(f"- {s}: {len(data[s]):,} (card {EXPECTED_SIZES[s]:,})")

    for s in SPLITS:
        rs = data[s]
        print(f"\n## {s}")
        print("kind:", dict(Counter(r.kind for r in rs)))
        print("source:", dict(Counter(r.source for r in rs)))
        print("language (metadata):", dict(Counter(r.language for r in rs)))
        non_en = sum(not r.is_english for r in rs)
        print(f"non-English (CJK text): {non_en} ({non_en / len(rs):.2%}); metadata en+zh: "
              f"{sum(r.language == 'en+zh' for r in rs)}")
        print("options per row:", dict(sorted(Counter(r.n_options for r in rs).items())))
        print("target_basis:", dict(Counter(r.target_basis for r in rs)))
        hard = Counter((r.kind, r.hard) for r in rs)
        print("hard (one-hot) rows by kind:", {f"{k}/{'hard' if h else 'soft'}": n for (k, h), n in sorted(hard.items())})
        lens = np.array([len(x) for x in tok([render_prompt(r) for r in rs], add_special_tokens=False)["input_ids"]])
        print(f"prompt tokens: mean {lens.mean():.0f} p50 {np.percentile(lens, 50):.0f} p99 {np.percentile(lens, 99):.0f} "
              f"max {lens.max()}; rows > {MAX_LENGTH}: {(lens > MAX_LENGTH).sum()}")
        by_src = {}
        for r, n in zip(rs, lens):
            by_src.setdefault(r.source, []).append(n)
        print("mean tokens by source:", {k: int(np.mean(v)) for k, v in by_src.items()})

    train_groups = {r.group_id for r in data["train"]}
    print("\n## Group leakage (group_id shared with train)")
    for s in SPLITS[1:]:
        print(f"- {s}: {len({r.group_id for r in data[s]} & train_groups)}")

    if args.samples:
        print("\n## Sample prompts")
        for kind in ("choice", "noul", "score"):
            for r in [r for r in data["train"] if r.kind == kind][: args.samples]:
                print(f"\n--- {r.id} | target {r.target}\n{render_prompt(r)}")


if __name__ == "__main__":
    main()
