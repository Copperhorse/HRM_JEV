import math

import numpy as np
import pytest

from src.metrics import evaluate, fit_temperature, softmax


def test_hand_computed_values():
    logits = [np.array([0.0, 0.0]), np.array([math.log(3), 0.0])]
    targets = [np.array([1.0, 0.0]), np.array([0.5, 0.5])]  # one hard, one soft
    m = evaluate(logits, targets)
    assert m["n"] == 2 and m["hard_n"] == 1
    assert m["hard_acc"] == 1.0  # soft row excluded; tie argmax -> first option, which is correct
    assert m["expected_acc"] == pytest.approx((1.0 + 0.5) / 2)
    nll0 = math.log(2)
    nll1 = -(0.5 * math.log(0.75) + 0.5 * math.log(0.25))
    assert m["nll"] == pytest.approx((nll0 + nll1) / 2)
    brier0 = 0.25 + 0.25
    brier1 = 0.25**2 + 0.25**2
    assert m["brier"] == pytest.approx((brier0 + brier1) / 2)
    # ECE, 15 bins: row0 conf .5 (bin 7), correct 1.0 ; row1 conf .75 (bin 11), expected correct .5
    assert m["ece"] == pytest.approx((abs(0.5 - 1.0) + abs(0.75 - 0.5)) / 2)


def test_hard_accuracy_excludes_soft_rows_and_handles_none():
    wrong_soft = evaluate([np.array([5.0, 0.0])], [np.array([0.4, 0.6])])
    assert wrong_soft["hard_acc"] is None and wrong_soft["hard_n"] == 0
    assert wrong_soft["expected_acc"] == pytest.approx(0.4)


def test_variable_option_counts_and_temperature_scaling():
    logits = [np.array([2.0, 0.0, -1.0]), np.array([0.0, 3.0])]
    targets = [np.array([1.0, 0, 0]), np.array([0, 1.0])]
    cold, hot = evaluate(logits, targets, temperature=0.1), evaluate(logits, targets, temperature=10.0)
    assert cold["nll"] < hot["nll"] and cold["hard_acc"] == hot["hard_acc"] == 1.0
    assert softmax(np.array([1.0, 2.0, 3.0])).sum() == pytest.approx(1.0)


def test_fit_temperature_recovers_known_value():
    rng = np.random.default_rng(0)
    logits = [rng.normal(0, 3, int(rng.choice([2, 3, 5]))) for _ in range(400)]
    # Targets are exactly softmax(l / 2): cross-entropy is minimised at T = 2.
    targets = [softmax(l, 2.0) for l in logits]
    assert fit_temperature(logits, targets) == pytest.approx(2.0, rel=1e-3)


def test_evaluate_rejects_bad_input():
    with pytest.raises(ValueError):
        evaluate([], [])
    with pytest.raises(ValueError):
        evaluate([np.zeros(2)], [np.ones(2), np.ones(2)])
