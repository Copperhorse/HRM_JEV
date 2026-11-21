import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from src.config import TrainConfig
from src.data import sharc
from src.data.prompt import render_prompt


def _item(i: int, answer: str, *, scenario="I am 20.", history=None, tree=None, snippet=None) -> dict:
    return {"utterance_id": f"u{i}", "tree_id": tree or f"t{i}", "source_url": "https://www.gov.uk/x",
            "snippet": snippet or f"Rule {i}: you must be over 18 and a resident.", "question": f"Can I apply {i}?",
            "scenario": scenario, "answer": answer, "evidence": [],
            "history": history if history is not None else []}


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    answers = ["Yes", "No", "Irrelevant", "Are you a resident?"] * 5  # 20 utterances
    train = [_item(i, a) for i, a in enumerate(answers)]
    train.append(dict(train[0]))  # exact duplicate
    dev = [_item(100 + i, a) for i, a in enumerate(answers[:4])]
    test = [_item(200, "Yes")]
    z = tmp_path / "data" / "sharc" / "sharc1-official.zip"
    z.parent.mkdir(parents=True)
    with zipfile.ZipFile(z, "w") as f:
        for name, rows in (("train", train), ("dev", dev), ("test", test)):
            f.writestr(f"sharc1-official/json/sharc_{name}.json", json.dumps(rows))
    monkeypatch.setattr(sharc, "SHARC_SHA256", hashlib.sha256(z.read_bytes()).hexdigest())
    monkeypatch.setattr(sharc, "EXPECTED_SIZES", {"train": len(train), "dev": len(dev), "test": len(test)})
    return tmp_path / "data"


def test_labels_and_views(fake):
    recs = sharc.load_sharc("dev", fake, view="choice")
    assert [r.target.index(1.0) for r in recs] == [0, 1, 2, 3]  # yes, no, irrelevant, inquire
    assert all(r.kind == "choice" and r.n_options == 4 and r.source == "sharc" and r.is_english for r in recs)
    noul = sharc.load_sharc("dev", fake, view="noul")
    assert [(r.options, r.target) for r in noul] == [(("no", "yes"), (0.0, 1.0)), (("no", "yes"), (1.0, 0.0))]
    assert all(r.hard for r in recs + noul)


def test_state_renders_sections_only_when_present(fake):
    plain = sharc.to_record(_item(1, "Yes", scenario="", history=[]), "choice").state
    assert plain.startswith("Rule text:") and "User scenario" not in plain and "Follow-up" not in plain
    full = sharc.to_record(_item(1, "Yes", history=[{"follow_up_question": "Are you a resident?",
                                                      "follow_up_answer": "Yes"}]), "choice").state
    assert "User scenario: I am 20." in full and "Q: Are you a resident?\nA: Yes" in full


def test_dedup_and_prompt_fits(fake):
    items = sharc.load_items("train", fake)
    assert len(items) == 20  # the exact duplicate is gone
    p = render_prompt(sharc.to_record(items[0], "choice"))
    assert p.startswith("<|im_start|><|object_ref_start|>Context:\nRule text:") and p.count("\nD. ask a follow-up") == 1


def test_sample_is_seeded_and_disjoint(fake):
    a = sharc.sample_sharc(8, 4, seed=1, data_dir=fake)
    assert a == sharc.sample_sharc(8, 4, seed=1, data_dir=fake)
    assert a != sharc.sample_sharc(8, 4, seed=2, data_dir=fake)
    choice, noul = [r for r in a if r.kind == "choice"], [r for r in a if r.kind == "noul"]
    assert len(choice) == 8 and len(noul) == 4
    uid = lambda r: r.id.split(":")[1]
    assert not {uid(r) for r in choice} & {uid(r) for r in noul}
    assert len({r.id for r in a}) == 12


def test_checksum_mismatch_is_refused(fake, monkeypatch):
    monkeypatch.setattr(sharc, "SHARC_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        sharc.load_items("dev", fake)


def test_size_check_and_unknown_split(fake, monkeypatch):
    monkeypatch.setattr(sharc, "EXPECTED_SIZES", {"train": 1, "dev": 1, "test": 1})
    with pytest.raises(ValueError, match="expected 1"):
        sharc.load_items("dev", fake)
    with pytest.raises(ValueError, match="unknown ShARC split"):
        sharc.load_items("val", fake)


def test_shipped_config_loads():
    cfg = TrainConfig.from_yaml(Path(__file__).parents[1] / "configs" / "seed1_sharc.yaml")
    assert (cfg.max_rows, cfg.sharc_choice_rows, cfg.sharc_noul_rows) == (16000, 6000, 2000)
    assert TrainConfig.from_yaml(Path(__file__).parents[1] / "configs" / "seed1.yaml").sharc_choice_rows == 0


def test_long_sharc_config_matches_seed1_long_plus_sharc():
    root = Path(__file__).parents[1] / "configs"
    a, b = TrainConfig.from_yaml(root / "seed1_long.yaml"), TrainConfig.from_yaml(root / "seed1_long_sharc.yaml")
    assert (b.max_rows, b.sharc_choice_rows, b.sharc_noul_rows) == (60000, 6000, 2000)
    same = ("seed", "dtype", "batch_size", "accumulation", "lr", "readout_lr", "warmup_steps", "eval_every", "eval_rows")
    assert all(getattr(a, k) == getattr(b, k) for k in same)
