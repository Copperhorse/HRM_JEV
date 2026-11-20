"""Two-process (gloo, CPU) training must equal single-process training with the same recipe, and stay resumable."""
import os
import socket

import pytest
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from src.train import assign_micro_batches, train
from tests.test_train import cfg_for, tiny_factory, weights


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, port, out_root, run_id, kw):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world),
                      LOCAL_RANK=str(rank))
    from pathlib import Path

    tok = AutoTokenizer.from_pretrained("sapientinc/HRM-Text-1B")
    crash = kw.pop("_crash_after", None)
    train(cfg_for(Path(out_root), run_id, **kw), model_factory=tiny_factory, tokenizer=tok, _crash_after=crash)


def run_two(tmp_path, run_id, **kw):
    mp.spawn(_worker, args=(2, _free_port(), str(tmp_path), run_id, kw), nprocs=2, join=True)


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("sapientinc/HRM-Text-1B")


def close(a, b, atol=1e-5):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.allclose(a[k], b[k], atol=atol), k


def test_two_ranks_match_single_process(tok, tmp_path):
    single = cfg_for(tmp_path, "single", max_steps=4)
    train(single, model_factory=tiny_factory, tokenizer=tok)
    run_two(tmp_path, "dual", max_steps=4)
    close(weights(single.run_dir), weights(tmp_path / "dual"))
    assert (tmp_path / "dual" / "metrics.jsonl").read_text().count('"loss"') == 4  # rank 0 alone logs


def test_two_rank_resume_reproduces_uninterrupted_run(tok, tmp_path):
    run_two(tmp_path, "full", max_steps=4)
    with pytest.raises(Exception, match="simulated crash"):
        run_two(tmp_path, "crashed", max_steps=4, _crash_after=3)  # last checkpoint: step 2
    run_two(tmp_path, "crashed", max_steps=4)  # resumes at step 2
    close(weights(tmp_path / "full"), weights(tmp_path / "crashed"))


def test_accumulation_must_divide_by_world(tmp_path):
    with pytest.raises(Exception, match="divisible"):
        run_two(tmp_path, "bad", accumulation=3, max_steps=1)


def test_assignment_is_a_balanced_partition():
    cost = [5, 90, 40, 70, 1, 3, 200, 8]
    for world, idxs in ((2, [0, 1, 2, 3]), (2, [4, 5, 6, 7]), (2, list(range(8))), (4, list(range(8)))):
        parts = assign_micro_batches(idxs, cost, world)
        assert sorted(i for p in parts for i in p) == idxs and {len(p) for p in parts} == {len(idxs) // world}
        assert parts == assign_micro_batches(list(reversed(idxs)), cost, world)  # deterministic
    a, b = assign_micro_batches([0, 1, 2, 3], cost, 2)
    assert abs(sum(cost[i] for i in a) - sum(cost[i] for i in b)) < abs(cost[0] + cost[2] - cost[1] - cost[3])  # beats the positional split (ranks take 0,2 / 1,3)
    assert assign_micro_batches([3, 4, 5, 6], cost, 1) == [[3, 4, 5, 6]]
