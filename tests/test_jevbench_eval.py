"""JevBench public-item scoring: conversion, JevBench's metric definitions (hand-computed), over-context handling."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from transformers import AutoTokenizer

from src.jevbench_eval import (TIER_CHANCE, argmax_label, ece_top_label, evaluate_items, intelligence, item_labels, load_items,
                               score_rows, summarize, to_record, wilson)
from tests.test_train import cfg_for, tiny_factory

CHOICE = {"id": "c1", "tier": "easy", "expected": "b", "labels": ["a", "b"], "state": "hello",
          "question": {"type": "choice", "instructions": "Pick.", "criteria": {"a": "first one", "b": None}}}
SCORE = {"id": "s1", "tier": "standard", "expected": 1, "labels": ["0", "1", "2"], "state": {"k": [1, 2]},
         "question": {"type": "score", "instructions": "Rate.", "criteria": ["low", "mid", "high"]}}
NOUL = {"id": "n1", "tier": "hard", "expected": "yes", "labels": ["no", "yes"], "state": "text",
        "question": {"type": "noul", "instructions": "Is it?", "criteria": {"true": "it is", "false": "it is not"}}}


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("sapientinc/HRM-Text-1B")


def test_conversion_matches_open_jev_compile_request():
    c, s, n = to_record(CHOICE), to_record(SCORE), to_record(NOUL)
    assert c.options == ("a: first one", "b") and c.kind == "choice"          # "name: description", bare name if none
    assert s.options == ("low", "mid", "high") and item_labels(SCORE) == ["0", "1", "2"]
    assert n.options == ("no", "yes") and n.question == "Is it?\nYes means: it is\nNo means: it is not"
    with pytest.raises(ValueError, match="do not match"):
        item_labels({**CHOICE, "labels": ["a", "zzz"]})


def test_metrics_by_hand():
    assert argmax_label({"b": 0.5, "a": 0.5}) == "a"                           # tie -> smallest label
    # bins: conf 0.95 (right), 0.95 (wrong) -> |0.5-0.95|; conf 0.55 (right) -> |1-0.55|; ECE = (2*0.45 + 0.45)/3
    assert ece_top_label([(0.95, True), (0.95, False), (0.55, True)]) == pytest.approx((2 * 0.45 + 0.45) / 3)
    assert ece_top_label([]) is None
    assert intelligence({t: TIER_CHANCE[t] for t in TIER_CHANCE}) == pytest.approx(0.0)
    assert intelligence({t: 1.0 for t in TIER_CHANCE}) == pytest.approx(100.0)
    assert intelligence({"easy": 1.0}) == pytest.approx(100.0)                 # weights renormalised over present tiers
    lo, hi = wilson(7, 10)
    assert lo < 0.7 < hi


def test_score_rows_and_invalid_item():
    rows = score_rows([CHOICE, SCORE, NOUL], [np.array([0.0, 2.0]), np.array([0.0, 3.0, 0.0]), None], temperature=1.0)
    assert rows[0]["predicted"] == "b" and rows[0]["correct"]
    assert rows[1]["predicted"] == "1" and rows[1]["correct"]                   # expected is the int 1 -> "1"
    assert rows[2]["valid"] is False and rows[2]["correct"] is False and rows[2]["confidence"] is None
    s = summarize(rows)
    assert s["accuracy"] == pytest.approx(2 / 3) and s["invalid_over_context"] == 1 and s["hard_ece"] is None
    # temperature changes confidence, never the argmax
    hot = score_rows([CHOICE], [np.array([0.0, 2.0])], temperature=4.0)[0]
    assert hot["predicted"] == "b" and hot["confidence"] < rows[0]["confidence"]


def test_over_context_item_is_not_truncated_and_model_runs_on_the_rest(tok, tmp_path):
    long_item = {**CHOICE, "id": "long", "state": "word " * 5000}
    model = tiny_factory(cfg_for(tmp_path, "x"), tok).eval()
    out = evaluate_items(model, tok, [CHOICE, SCORE, NOUL, long_item], device="cpu", dtype="fp32")
    assert out[3] is None and [len(o) for o in out[:3]] == [2, 3, 2] and all(np.isfinite(o).all() for o in out[:3])


@pytest.mark.skipif(not os.environ.get("JEVBENCH_DIR"), reason="set JEVBENCH_DIR to a clone of fstandhartinger/jevbench")
def test_real_public_items_all_convert():
    items = load_items(os.environ["JEVBENCH_DIR"])
    assert len(items) == 231 and {i["tier"] for i in items} == {"easy", "standard", "hard"}
    for it in items:
        assert 2 <= len(to_record(it).options) <= 16
