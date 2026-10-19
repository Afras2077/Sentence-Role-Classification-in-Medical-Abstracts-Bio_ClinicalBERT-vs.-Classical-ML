from __future__ import annotations
"""
Evaluation and statistical significance module for PubMed RCT sentence classification.
Provides metrics calculation, McNemar exact test, paired cluster-bootstrap,
and visualization utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import binomtest
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """
    Computes key classification metrics: Accuracy, Weighted-F1, Macro-F1, and Cohen's Kappa.
    """
    pfx = f"{prefix}_" if prefix else ""
    return {
        f"{pfx}acc": float(accuracy_score(y_true, y_pred)),
        f"{pfx}weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        f"{pfx}macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{pfx}kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
) -> pd.DataFrame:
    """
    Computes Precision, Recall, F1, and Support for each rhetorical role class.
    """
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(labels)), zero_division=0
    )
    df = pd.DataFrame({
        "Class": labels,
        "Precision": np.round(p, 4),
        "Recall": np.round(r, 4),
        "F1": np.round(f, 4),
        "Support": s,
    })
    return df.set_index("Class")


def fast_macro_f1(y: np.ndarray, p: np.ndarray, k: int = 5) -> float:
    """
    Fast confusion-matrix based macro-F1 calculation for bootstrap loops.
    """
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    return float(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean())


def mcnemar_exact(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
) -> Tuple[int, int, float]:
    """
    Computes McNemar's exact test using the binomial distribution.
    Compares model A and model B predictions against ground truth.
    Returns:
        n_a_only: Count where only model A is correct
        n_b_only: Count where only model B is correct
        p_value: Two-sided binomial test p-value
    """
    a_ok = (p_a == y_true)
    b_ok = (p_b == y_true)
    n_a_only = int(np.sum(a_ok & ~b_ok))
    n_b_only = int(np.sum(~a_ok & b_ok))
    n_total = n_a_only + n_b_only
    p_val = float(binomtest(n_a_only, n_total, 0.5).pvalue) if n_total > 0 else 1.0
    return n_a_only, n_b_only, p_val


def paired_bootstrap_diff(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Paired bootstrap test resampling whole abstracts (clusters) with replacement.
    Accounts for intra-abstract sentence dependency.
    Calculates the 95% confidence interval for macro-F1(B) - macro-F1(A).
    Returns:
        mean_diff, ci_low, ci_high, diffs_array
    """
    rng = np.random.default_rng(seed)
    codes, _ = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(codes))]
    idx_by_group = [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    g_count = len(idx_by_group)

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        sampled_g = rng.integers(0, g_count, size=g_count)
        boot_idx = np.concatenate([idx_by_group[g] for g in sampled_g])
        y_b = y_true[boot_idx]
        diffs[b] = fast_macro_f1(y_b, p_b[boot_idx]) - fast_macro_f1(y_b, p_a[boot_idx])

    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    mean_diff = float(diffs.mean())
    return mean_diff, ci_low, ci_high, diffs


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    model_name: str,
    out_path: Optional[str] = None,
    normalize: str = "true",
) -> plt.Figure:
    """
    Plots a row-normalised confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        cbar=True,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix (Row-Normalised) — {model_name}", fontsize=12, pad=12)
    ax.set_ylabel("True Role", fontsize=11)
    ax.set_xlabel("Predicted Role", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved confusion matrix to {out_path}")
    return fig


def plot_accuracy_by_position(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    preds_dict: Dict[str, np.ndarray],
    out_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots prediction accuracy as a function of sentence position decile in the abstract.
    Matches Figure 1 (right) of Phase 3 report.
    """
    pos_decile = np.minimum((test_df["relative_position"].values * 10).astype(int), 9)
    fig, ax = plt.subplots(figsize=(8, 5))

    styles = {
        "Bio_ClinicalBERT [text]": {"color": "#1f77b4", "marker": "s", "linestyle": "-"},
        "LogReg (SGD) [text]": {"color": "#ff7f0e", "marker": "o", "linestyle": "--"},
        "XGBoost [text]": {"color": "#2ca02c", "marker": "^", "linestyle": ":"},
        "Random Forest [text]": {"color": "#d62728", "marker": "x", "linestyle": "-."},
        "LogReg + position (extension)": {"color": "#9467bd", "marker": "d", "linestyle": "-"},
    }

    for name, preds in preds_dict.items():
        accs = []
        for d in range(10):
            mask = pos_decile == d
            accs.append(accuracy_score(y_true[mask], preds[mask]))
        style = styles.get(name, {"marker": "o", "linestyle": "-"})
        ax.plot(range(10), accs, label=name, **style)

    ax.set_title("Test Accuracy by Sentence Position Decile", fontsize=12, pad=12)
    ax.set_xlabel("Position Decile in Abstract (0 = start, 9 = end)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xticks(range(10))
    ax.set_ylim(0.55, 1.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved position accuracy plot to {out_path}")
    return fig


def plot_model_comparison_bar(
    summary_df: pd.DataFrame,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generates a comparative bar plot of all models across primary evaluation metrics.
    """
    metrics = ["test_macro_f1", "test_acc", "test_weighted_f1", "test_kappa"]
    labels = ["Macro F1", "Accuracy", "Weighted F1", "Cohen's Kappa"]
    available_metrics = [m for m in metrics if m in summary_df.columns]

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data = summary_df[available_metrics].copy()
    plot_data.columns = labels[:len(available_metrics)]
    plot_data.plot(kind="bar", ax=ax, colormap="tab10", width=0.8)

    ax.set_title("Model Performance Comparison on PubMed 20k RCT Test Set", fontsize=13, pad=14)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0.65, 0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.legend(loc="lower right")
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved model comparison bar plot to {out_path}")
    return fig
