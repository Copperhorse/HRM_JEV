"""Card-vs-ours comparison on the rows both evaluations share. Usage (repo root, CPU):
    curl -sL https://huggingface.co/ZefanCai/Open-Jev-2B/raw/main/evaluation/full-data.json -o full-data.json
    python scripts/compare_open_jev_2b_card.py full-data.json
Subtracts the card's `wikispeedia-v1` source (not in release-v2-redistributable) from its totals, then compares with runs/seed1_long/preds/best.
Card numbers are UNVERIFIED (not re-run); T = 1.0823 is ours, the card's own is 1.5188."""
import json, sys, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.report import load_preds, arrays
from src.metrics import evaluate
card = json.load(open(sys.argv[1]))["metrics"]
T = 1.0823394551476375
out = {}
for sp in ("test", "ood"):
    df = load_preds(Path("runs/seed1_long/preds/best"), sp)
    def src(s): return s
    ours = {}
    for s, g in df.groupby(df.source.str.split("/").str[0]):
        lg, tg = arrays(g)
        ours[s] = evaluate(lg, tg, temperature=T)
    cs = card[sp]["by_source"]
    # card aggregate excluding wikispeedia
    tot = card[sp]
    w = cs["wikispeedia-v1"]
    n = tot["count"] - w["count"]; hn = tot["hard_count"] - w["hard_count"]
    hok = tot["accuracy_all_rows_failures_incorrect"] * tot["hard_count"] - w["accuracy_all_rows_failures_incorrect"] * w["hard_count"]
    pm, wm = tot["probability_metrics_all_rows"], w["probability_metrics_all_rows"]
    c = dict(n=n, hard_n=hn, hard_acc=hok / hn, exp_acc=(tot["expected_accuracy_all_rows_failures_zero"] * tot["count"] - w["expected_accuracy_all_rows_failures_zero"] * w["count"]) / n,
             nll=(pm["nll"] * pm["count"] - wm["nll"] * wm["count"]) / n, brier=(pm["brier"] * pm["count"] - wm["brier"] * wm["count"]) / n)
    full = evaluate(*arrays(df), temperature=T)
    print(f"\n== {sp}: card minus wikispeedia vs ours (same {n} rows, hard_n card {hn} vs ours {full['hard_n']})")
    print(f"{'':10s} {'hard acc':>9s} {'exp acc':>8s} {'NLL':>7s} {'Brier':>7s}")
    print(f"{'Open-Jev-2B':10s} {c['hard_acc']:9.4f} {c['exp_acc']:8.4f} {c['nll']:7.4f} {c['brier']:7.4f}")
    print(f"{'ours':10s} {full['hard_acc']:9.4f} {full['expected_acc']:8.4f} {full['nll']:7.4f} {full['brier']:7.4f}")
    print(f"per source (n, hard acc card | ours; NLL card | ours):")
    # card sources may be split by workflow subdomain; aggregate by prefix
    agg = collections.defaultdict(lambda: dict(count=0, hard=0, ok=0.0, nllsum=0.0, brsum=0.0))
    for k, v in cs.items():
        p = k.split("/")[0]
        a = agg[p]; a["count"] += v["count"]; a["hard"] += v["hard_count"]; a["ok"] += v["accuracy_all_rows_failures_incorrect"] * v["hard_count"]
        pmv = v["probability_metrics_all_rows"]; a["nllsum"] += pmv["nll"] * pmv["count"]; a["brsum"] += pmv["brier"] * pmv["count"]
    for p, a in sorted(agg.items(), key=lambda x: -x[1]["count"]):
        if p == "wikispeedia-v1": continue
        o = ours.get(p)
        if o is None: print(p, "missing in ours", list(ours)); continue
        ca = a["ok"] / a["hard"]; cn = a["nllsum"] / a["count"]
        flag = "" if o["n"] == a["count"] else f" (n differs: card {a['count']} vs ours {o['n']})"
        print(f"{p:24s} n={o['n']:5d} acc {ca:.3f} | {o['hard_acc']:.3f} ({o['hard_acc']-ca:+.3f})   NLL {cn:.3f} | {o['nll']:.3f}{flag}")
