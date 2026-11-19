"""Hub upload is opt-in, private by default, never uploads optimizer state, and cannot kill training."""
import sys
import types

import pytest

from src.hub import PATTERNS, sync_run
from tests.test_train import cfg_for, tiny_factory, tok  # noqa: F401
from src.train import train


class FakeApi:
    calls = []
    fail = False

    def __init__(self, token=None):
        self.token = token

    def create_repo(self, repo, **kw):
        if FakeApi.fail:
            raise RuntimeError("401 Unauthorized")
        FakeApi.calls.append(("create_repo", repo, kw))

    def upload_folder(self, **kw):
        FakeApi.calls.append(("upload_folder", kw))


@pytest.fixture(autouse=True)
def fake_hub(monkeypatch):
    FakeApi.calls, FakeApi.fail = [], False
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)


def test_disabled_by_default_uploads_nothing(tok, tmp_path):  # noqa: F811
    train(cfg_for(tmp_path, "h0", max_steps=2), model_factory=tiny_factory, tokenizer=tok)
    assert FakeApi.calls == []


def test_enabled_uploads_adapter_privately_without_optimizer_state(tok, tmp_path):  # noqa: F811
    cfg = cfg_for(tmp_path, "h1", max_steps=4, hub_repo="user/adapters", save_every=2)
    train(cfg, model_factory=tiny_factory, tokenizer=tok)
    creates = [c for c in FakeApi.calls if c[0] == "create_repo"]
    uploads = [c[1] for c in FakeApi.calls if c[0] == "upload_folder"]
    assert len(uploads) >= 2  # one at the step-2 checkpoint, one at the end
    assert all(c[2]["private"] is True and c[1] == "user/adapters" for c in creates)
    u = uploads[-1]
    assert u["path_in_repo"] == "h1" and u["allow_patterns"] == PATTERNS
    assert not any("ckpt" in p or "parquet" in p or "optimizer" in p for p in PATTERNS)
    notes = (cfg.run_dir / "MODEL_NOTES.md").read_text()
    assert "sapientinc/HRM-Text-1B" in notes and "no performance claim" in notes


def test_upload_failure_does_not_kill_training(tok, tmp_path, capsys):  # noqa: F811
    FakeApi.fail = True
    out = train(cfg_for(tmp_path, "h2", max_steps=2, hub_repo="user/adapters"), model_factory=tiny_factory, tokenizer=tok)
    assert out["steps"] == 2
    assert "hub upload failed" in capsys.readouterr().out
