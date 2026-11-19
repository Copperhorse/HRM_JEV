"""Decision metrics and temperature calibration (numpy; no torch dependency).

Definitions follow the Open-Jev-2B card / reference `jev.metrics` so numbers are comparable:
- hard accuracy: only rows with an exactly one-hot target (soft-target rows excluded);
- expected accuracy: mean target mass at the predicted option, all rows;
- NLL / Brier use all rows (Brier = sum-of-squares distance to the target distribution);
- ECE: top-label, 15 equal-width bins, correctness = target mass at the predicted option.
Rows have different option counts, so logits are kept as a list of 1-D arrays.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def evaluate(logits: Sequence[np.ndarray], targets: Sequence[np.ndarray], *, temperature: float = 1.0,
             n_bins: int = 15) -> dict:
    if len(logits) != len(targets) or not len(logits):
        raise ValueError("logits and targets must have equal nonzero length")
    nll = brier = exp_acc = 0.0
    hard_n = hard_ok = 0
    bins = np.zeros((n_bins, 3))  # count, sum confidence, sum expected correctness
    for lg, t in zip(logits, targets):
        p = softmax(lg, temperature)
        t = np.asarray(t, dtype=np.float64)
        pred = int(np.argmax(p))
        nll -= float((t * np.log(np.maximum(p, 1e-15))).sum())
        brier += float(((p - t) ** 2).sum())
        correct = float(t[pred])
        exp_acc += correct
        if t.max() == 1.0:
            hard_n += 1
            hard_ok += correct == 1.0
        b = min(int(p[pred] * n_bins), n_bins - 1)
        bins[b] += (1, p[pred], correct)
    n = len(logits)
    ece = float(np.abs(bins[:, 1] - bins[:, 2]).sum() / n)
    return {"n": n, "hard_n": hard_n, "hard_acc": hard_ok / hard_n if hard_n else None, "expected_acc": exp_acc / n,
            "nll": nll / n, "brier": brier / n, "ece": ece}


def fit_temperature(logits: Sequence[np.ndarray], targets: Sequence[np.ndarray], *, lo: float = 0.05,
                    hi: float = 20.0) -> float:
    """Single positive temperature minimising calibration-set cross-entropy (grid + golden section on log T)."""
    shifted = [np.asarray(l, np.float64) - np.max(l) for l in logits]
    ts = [np.asarray(t, np.float64) for t in targets]

    def loss(logt: float) -> float:
        inv = math.exp(-logt)
        return sum(np.log(np.exp(s * inv).sum()) - (t * s).sum() * inv for s, t in zip(shifted, ts)) / len(ts)

    grid = np.linspace(math.log(lo), math.log(hi), 25)
    vals = [loss(g) for g in grid]
    i = int(np.argmin(vals))
    a, b = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
    r = (math.sqrt(5) - 1) / 2
    c, d = b - r * (b - a), a + r * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(30):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - r * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + r * (b - a)
            fd = loss(d)
    return math.exp((a + b) / 2)
