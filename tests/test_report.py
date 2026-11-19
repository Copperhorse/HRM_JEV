import numpy as np
import pandas as pd
import pytest

from src.eval import preds_path
from src.report import build_report, load_preds, to_markdown


def fake_split(prefix, n, rng, temperature=2.0):
    rows = []
    for i in range(n):
        k = int(rng.choice([2, 3, 4]))
        logits = rng.normal(0, 3, k)
        p = np.exp(logits / temperature)
        p /= p.sum()
        target = np.eye(k)[rng.choice(k, p=p)]  # labels drawn from a model that is calibrated at `temperature`
        rows.append({"id": f"{prefix}{i}", "source": "s", "kind": ["choice", "noul", "score"][i % 3], "n_options": k,
                     "is_english": i % 10 != 0, "hard": True, "logits": logits.tolist(), "target": target.tolist()})
    return pd.DataFrame(rows)


def test_report_end_to_end_with_shards(tmp_path):
    rng = np.random.default_rng(0)
    d = tmp_path / "preds" / "best"
    d.mkdir(parents=True)
    fake_split("c", 3000, rng).to_parquet(preds_path(tmp_path, "best", "calibration", None))
    for split in ("test", "ood"):
        df = fake_split(split, 200, rng)
        df.iloc[::2].to_parquet(preds_path(tmp_path, "best", split, (0, 2)))
        df.iloc[1::2].to_parquet(preds_path(tmp_path, "best", split, (1, 2)))
    rep = build_report(d)
    assert rep["temperature"] == pytest.approx(2.0, rel=0.15)  # recovered from calibration data
    by = {(r["split"], r["subset"], r["kind"]): r for r in rep["rows"]}
    assert by["test", "full", "all"]["n"] == 200 and by["test", "english", "all"]["n"] == 180
    assert by["ood", "full", "choice"]["n"] + by["ood", "full", "noul"]["n"] + by["ood", "full", "score"]["n"] == 200
    assert all(r["ece"] < r["ece_raw"] + 0.05 for r in rep["rows"] if r["kind"] == "all")
    assert "| test | english | all |" in to_markdown(rep)


def test_duplicate_ids_across_shards_rejected(tmp_path):
    df = fake_split("x", 10, np.random.default_rng(1))
    d = tmp_path / "preds" / "best"
    d.mkdir(parents=True)
    df.to_parquet(d / "test.shard-0-of-2.parquet")
    df.to_parquet(d / "test.shard-1-of-2.parquet")
    with pytest.raises(ValueError, match="duplicate"):
        load_preds(d, "test")
