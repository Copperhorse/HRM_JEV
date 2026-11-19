"""W&B is optional: no-op by default; when enabled it mirrors metrics through a (faked) wandb module."""
import sys
import types

from src.config import TrainConfig
from src.tracking import Tracker, _flatten
from tests.test_train import cfg_for, tiny_factory, tok  # noqa: F401  (fixture reuse)
from src.train import train


def test_flatten_keeps_numbers_only():
    assert _flatten({"step": 3, "loss": 0.5, "val": {"nll": 1.2, "n": 4}, "mem_gb": None, "flag": True, "s": "x"}) == {
        "step": 3, "loss": 0.5, "val/nll": 1.2, "val/n": 4}


def test_disabled_by_default_and_does_not_import_wandb(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)  # import would raise
    t = Tracker(TrainConfig(run_id="x"), tmp_path)
    t.log({"step": 1, "loss": 1.0})
    t.finish()
    assert t.run is None and not (tmp_path / "wandb_id.txt").exists()


def test_enabled_mirrors_metrics_and_reuses_run_id(tok, tmp_path, monkeypatch):  # noqa: F811
    calls = {"init": [], "log": [], "finished": 0}

    class Run:
        summary = {}

        def log(self, data, step):
            calls["log"].append((step, data))

        def finish(self):
            calls["finished"] += 1

    fake = types.SimpleNamespace(
        init=lambda **kw: calls["init"].append(kw) or Run(),
        Settings=lambda **kw: kw)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    cfg = cfg_for(tmp_path, "w", max_steps=2, wandb_project="proj")
    train(cfg, model_factory=tiny_factory, tokenizer=tok)
    init = calls["init"][0]
    assert init["project"] == "proj" and init["name"] == "w" and init["config"]["seed"] == 1
    assert init["settings"] == {"save_code": False}
    assert [s for s, _ in calls["log"]] == sorted(s for s, _ in calls["log"])  # non-decreasing steps
    assert any("loss" in d for _, d in calls["log"]) and any("val/nll" in d for _, d in calls["log"])
    assert calls["finished"] == 1
    first_id = init["id"]
    train(cfg, model_factory=tiny_factory, tokenizer=tok)  # same run dir: resumed session keeps the W&B run id
    assert calls["init"][1]["id"] == first_id
