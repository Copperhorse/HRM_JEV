"""A wiped local run dir must not cost GPU time: ckpt goes to the (fake) Hub and training resumes from it, bit for bit."""
import fnmatch
import shutil
from pathlib import Path

import pytest
import torch

from src.train import train
from tests.test_train import cfg_for, tiny_factory, tok, weights  # noqa: F401


class FakeHub:
    """In-memory stand-in for the Hub: upload_folder / snapshot_download honouring allow_patterns."""
    root = None

    class Api:
        def __init__(self, token=None): ...
        def create_repo(self, repo, **kw): ...

        def upload_folder(self, folder_path, repo_id, path_in_repo, allow_patterns, **kw):
            base = Path(folder_path)
            for f in base.rglob("*"):
                rel = f.relative_to(base).as_posix()
                if f.is_file() and any(fnmatch.fnmatch(rel, p) for p in allow_patterns):
                    dst = FakeHub.root / path_in_repo / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst)


def fake_snapshot_download(repo, repo_type, token, local_dir, allow_patterns):
    for f in FakeHub.root.rglob("*"):
        rel = f.relative_to(FakeHub.root).as_posix()
        if f.is_file() and any(fnmatch.fnmatch(rel, p) for p in allow_patterns):
            dst = Path(local_dir) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
    # like the real snapshot_download: no matching files is not an error, just an empty folder


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    import huggingface_hub
    FakeHub.root = tmp_path / "hub"
    FakeHub.root.mkdir()
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHub.Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)


def test_resume_from_hub_after_local_wipe_matches_uninterrupted(tok, tmp_path, fake_hub):  # noqa: F811
    full = cfg_for(tmp_path / "a", "r", max_steps=4)
    train(full, model_factory=tiny_factory, tokenizer=tok)
    cfg = cfg_for(tmp_path / "b", "r", max_steps=4, hub_repo="u/adapters")
    with pytest.raises(RuntimeError, match="simulated crash"):
        train(cfg, model_factory=tiny_factory, tokenizer=tok, _crash_after=3)  # hub has the step-2 checkpoint
    shutil.rmtree(cfg.run_dir)  # the Kaggle session is wiped
    assert not cfg.run_dir.exists()
    train(cfg, model_factory=tiny_factory, tokenizer=tok)  # restores ckpt from the hub, resumes at step 2
    a, b = weights(full.run_dir), weights(cfg.run_dir)
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6), k


def test_no_hub_checkpoint_starts_fresh(tok, tmp_path, fake_hub, capsys):  # noqa: F811
    out = train(cfg_for(tmp_path, "fresh", max_steps=2, hub_repo="u/adapters"), model_factory=tiny_factory, tokenizer=tok)
    assert out["steps"] == 2


def test_resume_refuses_changed_training_settings_but_allows_eval_changes(tok, tmp_path):  # noqa: F811
    cfg = cfg_for(tmp_path, "g", max_steps=4)
    with pytest.raises(RuntimeError, match="simulated crash"):
        train(cfg, model_factory=tiny_factory, tokenizer=tok, _crash_after=3)
    with pytest.raises(ValueError, match="lr"):
        train(cfg_for(tmp_path, "g", max_steps=4, lr=1e-3), model_factory=tiny_factory, tokenizer=tok)
    with pytest.raises(ValueError, match="max_steps"):
        train(cfg_for(tmp_path, "g", max_steps=6), model_factory=tiny_factory, tokenizer=tok)
    out = train(cfg_for(tmp_path, "g", max_steps=4, eval_every=1, log_every=2, dtype="fp32"), model_factory=tiny_factory,
                tokenizer=tok)  # eval/logging settings may change
    assert out["steps"] == 4


def test_restore_run_items_only_fetches_what_eval_needs(tok, tmp_path, fake_hub):  # noqa: F811
    from src.hub import restore_run
    cfg = cfg_for(tmp_path, "e", max_steps=2, hub_repo="u/adapters")
    train(cfg, model_factory=tiny_factory, tokenizer=tok)
    shutil.rmtree(cfg.run_dir)
    assert restore_run(cfg.run_dir, "u/adapters", items=("final", "run.json"))
    assert (cfg.run_dir / "final" / "model.json").exists() and (cfg.run_dir / "run.json").exists()
    assert not (cfg.run_dir / "ckpt").exists() and not (cfg.run_dir / "best").exists()
    assert not restore_run(tmp_path / "other", "u/adapters", items=("best", "run.json"))  # nothing under that run id


@pytest.mark.parametrize("which", ["best", "final"])
def test_eval_cli_restores_adapter_from_hub_in_a_fresh_session(tok, tmp_path, fake_hub, monkeypatch, which):  # noqa: F811
    import sys
    import pandas as pd
    import src.eval as ev
    cfg = cfg_for(tmp_path, "ev", max_steps=2, hub_repo="u/adapters")
    train(cfg, model_factory=tiny_factory, tokenizer=tok)
    shutil.rmtree(cfg.run_dir)  # fresh Kaggle session: nothing local
    monkeypatch.setattr(ev, "default_model_factory", tiny_factory)
    base = ["eval", "--run", str(cfg.run_dir), "--ckpt", which, "--hub-repo", "u/adapters", "--split", "calibration",
            "--limit", "4"]
    monkeypatch.setattr(sys, "argv", base + ["--restore-only"])
    ev.main()  # restore once ...
    assert (cfg.run_dir / which / "model.json").exists() and not list(cfg.run_dir.glob("preds/**/*.parquet"))
    monkeypatch.setattr(sys, "argv", ["eval", "--run", str(cfg.run_dir), "--ckpt", which, "--split", "calibration",
                                      "--limit", "4", "--shard", "0/2"])
    ev.main()  # ... then shards run without the hub
    out = pd.read_parquet(ev.preds_path(cfg.run_dir, which, "calibration", (0, 2)))
    assert len(out) == 4 and out.logits.map(len).eq(out.n_options).all()
