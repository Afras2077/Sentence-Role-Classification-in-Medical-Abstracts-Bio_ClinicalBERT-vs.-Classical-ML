"""
Unit tests for evaluation metrics and statistical tests.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.evaluate import (
    compute_metrics,
    per_class_metrics,
    fast_macro_f1,
    mcnemar_exact,
    paired_bootstrap_diff,
)


def test_compute_metrics():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["acc"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["kappa"] == 1.0


def test_fast_macro_f1():
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4, 1, 1, 2, 3, 4])
    mf1_fast = fast_macro_f1(y_true, y_pred, k=5)
    metrics = compute_metrics(y_true, y_pred)
    assert np.isclose(mf1_fast, metrics["macro_f1"], atol=1e-4)


def test_mcnemar_exact():
    y_true = np.array([0, 1, 2, 3, 4])
    # Model A gets index 0 right, index 1 wrong
    p_a = np.array([0, 0, 2, 3, 4])
    # Model B gets index 0 wrong, index 1 right
    p_b = np.array([1, 1, 2, 3, 4])

    n_a, n_b, p_val = mcnemar_exact(y_true, p_a, p_b)
    assert n_a == 1
    assert n_b == 1
    assert p_val == 1.0


def test_paired_bootstrap_diff():
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 5, size=n)
    p_a = rng.integers(0, 5, size=n)
    p_b = y_true.copy()  # perfect model B
    groups = np.repeat(np.arange(20), 10)  # 20 abstracts with 10 sentences each

    mean_diff, ci_low, ci_high, diffs = paired_bootstrap_diff(
        y_true, p_a, p_b, groups, n_boot=100, seed=42
    )
    assert mean_diff > 0.0
    assert ci_low > 0.0
    assert ci_high > ci_low
