"""End-to-end smoke test of the training loop on a tiny random HRM (CPU, real tokenizer + real data rows)."""
import json

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, HrmTextConfig, HrmTextForCausalLM

from src.config import TrainConfig
from src.data.prompt import letter_token_ids
from src.models.hrm_decision import HrmDecisionModel
from src.train import lr_factor, make_micro_batches, train
from src.data.records import load_split


def tiny_factory(cfg, tok):
    torch.manual_seed(0)
    lm = HrmTextForCausalLM(HrmTextConfig(
        vocab_size=65536, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, head_dim=16, H_cycles=2, L_cycles=3, L_bp_cycles=[0, 3],
        max_position_embeddings=1024, pad_token_id=5, bos_token_id=6, eos_token_id=11, tie_word_embeddings=False))
    return HrmDecisionModel(lm, letter_token_ids(tok), lora_rank=4, l_bp_cycles=cfg.l_bp_cycles)


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("sapientinc/HRM-Text-1B")


def cfg_for(tmp_path, run_id, **kw):
    base = dict(run_id=run_id, out_root=str(tmp_path), dtype="fp32", device="cpu", max_rows=96, batch_size=4,
                accumulation=2, warmup_steps=1, log_every=1, eval_every=2, eval_rows=8, eval_batch_size=4,
                save_every=2, lr=5e-3, readout_lr=5e-3, grad_checkpointing=False)
    base.update(kw)
    return TrainConfig(**base)


def weights(run_dir, which="final"):
    return {**load_file(run_dir / which / "adapter" / "adapter_model.safetensors"),
            "readout": torch.load(run_dir / which / "readout.pt", weights_only=True)["weight"]}


def test_trains_logs_and_writes_artifacts(tok, tmp_path):
    cfg = cfg_for(tmp_path, "a", max_steps=4)
    out = train(cfg, model_factory=tiny_factory, tokenizer=tok)
    assert out["steps"] == 4
    rows = [json.loads(l) for l in (cfg.run_dir / "metrics.jsonl").read_text().splitlines()]
    losses = [r["loss"] for r in rows if "loss" in r]
    assert len(losses) == 4 and all(l == l and l < 1e3 for l in losses)
    assert any("val" in r for r in rows)
    for f in ("run.json", "final/readout.pt", "final/model.json", "best/readout.pt", "ckpt/optimizer.pt"):
        assert (cfg.run_dir / f).exists(), f
    run = json.loads((cfg.run_dir / "run.json").read_text())
    assert run["config"]["seed"] == 1 and run["dataset"][1] and run["model"][1]


def test_resume_reproduces_uninterrupted_run(tok, tmp_path):
    full = cfg_for(tmp_path, "full", max_steps=4)
    train(full, model_factory=tiny_factory, tokenizer=tok)
    crashed = cfg_for(tmp_path, "crashed", max_steps=4)
    with pytest.raises(RuntimeError, match="simulated crash"):
        train(crashed, model_factory=tiny_factory, tokenizer=tok, _crash_after=3)  # last checkpoint: step 2
    train(crashed, model_factory=tiny_factory, tokenizer=tok)  # resumes at step 2
    a, b = weights(full.run_dir), weights(crashed.run_dir)
    assert a.keys() == b.keys()
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6), k


def test_micro_batches_deterministic_full_size_and_no_leak(tok, tmp_path):
    rows = load_split("train", "data/raw")[:500]
    cfg = cfg_for(tmp_path, "m", batch_size=8, epochs=2)
    a, b = make_micro_batches(rows, cfg), make_micro_batches(rows, cfg)
    assert a == b and all(len(x) == 8 for x in a)
    half = len(a) // 2
    assert len({i for x in a[:half] for i in x}) == len([i for x in a[:half] for i in x])  # no repeats within an epoch
    assert make_micro_batches(rows, cfg_for(tmp_path, "m", batch_size=8, epochs=2, seed=2)) != a


def test_lr_schedule():
    assert lr_factor(0, 100, 10) == pytest.approx(0.1)
    assert lr_factor(9, 100, 10) == pytest.approx(1.0)
    assert lr_factor(100, 100, 10) == pytest.approx(0.1)
    assert all(lr_factor(s, 100, 10) <= 1.0 for s in range(100))
