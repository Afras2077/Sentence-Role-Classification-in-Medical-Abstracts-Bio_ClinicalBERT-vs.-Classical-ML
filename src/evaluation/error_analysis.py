from __future__ import annotations
"""
Comprehensive Error Analysis and Fairness Auditing Module for PubMed RCT Sentence Classification.

Provides:
1. Multi-class Confusion Matrix & Breakdown of primary discourse confusions.
2. Positional Error Analysis across Abstract Deciles (d = 0 to 9).
3. Syntactic/Length Error Analysis across Sentence Length Quartiles.
4. Algorithmic Bias and Fairness Auditing:
   - Minority Class Representation Fairness (Recall Parity across imbalanced roles).
   - Sentence Complexity Disparity Ratio (Acc_short / Acc_long).
   - Positional Boundary Disparity Ratio (Acc_interior / Acc_boundary).
5. Qualitative Error Case Extraction with Clinical Rhetorical Categorization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score


def analyze_confusions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
) -> pd.DataFrame:
    """
    Computes pair-wise error counts between classes and sorts by frequency.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    confusions = []
    for i, true_lbl in enumerate(labels):
        for j, pred_lbl in enumerate(labels):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    "True_Label": true_lbl,
                    "Predicted_Label": pred_lbl,
                    "Error_Count": int(cm[i, j]),
                    "Error_Share_Pct": float(round((cm[i, j] / cm[i].sum()) * 100, 2)),
                })
    df = pd.DataFrame(confusions).sort_values("Error_Count", ascending=False)
    return df.reset_index(drop=True)


def analyze_by_position(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    relative_positions: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Computes accuracy, error count, and error rate across relative position deciles (0 to 9).
    """
    pos_bins = np.minimum((relative_positions * n_bins).astype(int), n_bins - 1)
    results = []
    for b in range(n_bins):
        mask = (pos_bins == b)
        tot = int(mask.sum())
        if tot > 0:
            correct = int((y_true[mask] == y_pred[mask]).sum())
            errors = tot - correct
            acc = float(round(correct / tot, 4))
            err_rate = float(round(errors / tot, 4))
        else:
            acc, err_rate, errors = 0.0, 0.0, 0
        results.append({
            "Position_Decile": b,
            "Range": f"{b*10}-{(b+1)*10}%",
            "Total_Sentences": tot,
            "Correct": correct,
            "Errors": errors,
            "Accuracy": acc,
            "Error_Rate": err_rate,
        })
    return pd.DataFrame(results)


def analyze_by_sentence_length(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sentence_lengths: np.ndarray,
    n_bins: int = 4,
) -> pd.DataFrame:
    """
    Computes accuracy across sentence length quartiles (characters or word count).
    """
    quartile_labels = ["Q1 (Shortest)", "Q2 (Medium-Short)", "Q3 (Medium-Long)", "Q4 (Longest)"]
    quartiles = pd.qcut(sentence_lengths, q=n_bins, labels=quartile_labels[:n_bins], duplicates="drop")
    
    results = []
    for q_label in quartile_labels[:n_bins]:
        mask = (quartiles == q_label)
        tot = int(mask.sum())
        if tot > 0:
            correct = int((y_true[mask] == y_pred[mask]).sum())
            acc = float(round(correct / tot, 4))
            err_rate = float(round((tot - correct) / tot, 4))
            min_len = int(sentence_lengths[mask].min())
            max_len = int(sentence_lengths[mask].max())
            mean_len = float(round(sentence_lengths[mask].mean(), 1))
        else:
            acc, err_rate, min_len, max_len, mean_len = 0.0, 0.0, 0, 0, 0.0
        results.append({
            "Length_Quartile": q_label,
            "Mean_Char_Length": mean_len,
            "Length_Range": f"[{min_len}, {max_len}]",
            "Total_Sentences": tot,
            "Accuracy": acc,
            "Error_Rate": err_rate,
        })
    return pd.DataFrame(results)


def audit_bias_and_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
    labels: List[str],
) -> Dict[str, Any]:
    """
    Comprehensive algorithmic fairness audit evaluating:
    1. Minority Class Recall Parity (Equality of Opportunity).
    2. Syntactic Complexity / Sentence Length Fairness.
    3. Abstract Discourse Position Boundary Fairness.
    """
    # 1. Per-Class Recall Parity
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(labels)), zero_division=0)
    class_recalls = {lbl: float(round(r[i], 4)) for i, lbl in enumerate(labels)}
    max_recall = max(class_recalls.values())
    min_recall = min(class_recalls.values())
    recall_disparity_ratio = float(round(min_recall / max_recall, 4)) if max_recall > 0 else 0.0

    # 2. Length Quartile Fairness
    lengths = df["line_length"].values if "line_length" in df else df["text"].str.len().values
    med_len = float(np.median(lengths))
    short_mask = (lengths <= med_len)
    long_mask = (lengths > med_len)
    acc_short = float(round(accuracy_score(y_true[short_mask], y_pred[short_mask]), 4))
    acc_long = float(round(accuracy_score(y_true[long_mask], y_pred[long_mask]), 4))
    length_disparity_ratio = float(round(min(acc_short, acc_long) / max(acc_short, acc_long), 4))

    # 3. Position Boundary Fairness (Abstract Opening/Closing vs Interior)
    rel_pos = df["relative_position"].values if "relative_position" in df else np.zeros(len(df))
    boundary_mask = (rel_pos < 0.10) | (rel_pos >= 0.90)
    interior_mask = ~boundary_mask
    acc_boundary = float(round(accuracy_score(y_true[boundary_mask], y_pred[boundary_mask]), 4))
    acc_interior = float(round(accuracy_score(y_true[interior_mask], y_pred[interior_mask]), 4))
    boundary_disparity_ratio = float(round(acc_boundary / acc_interior, 4)) if acc_interior > 0 else 0.0

    fairness_metrics = {
        "class_recalls": class_recalls,
        "minority_class": "OBJECTIVE",
        "minority_recall": class_recalls.get("OBJECTIVE", 0.0),
        "majority_class": "METHODS",
        "majority_recall": class_recalls.get("METHODS", 0.0),
        "recall_disparity_ratio": recall_disparity_ratio,
        "acc_short_sentences": acc_short,
        "acc_long_sentences": acc_long,
        "length_disparity_ratio": length_disparity_ratio,
        "acc_abstract_boundaries": acc_boundary,
        "acc_abstract_interior": acc_interior,
        "boundary_disparity_ratio": boundary_disparity_ratio,
    }
    return fairness_metrics


def extract_qualitative_failures(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    id2label: Dict[int, str],
    n_samples_per_mode: int = 3,
) -> List[Dict[str, Any]]:
    """
    Extracts concrete qualitative error examples categorized by clinical failure mode.
    """
    modes = [
        ("OBJECTIVE", "BACKGROUND", "Opening Aim vs Context Confusion"),
        ("BACKGROUND", "OBJECTIVE", "Contextual Aim Misattribution"),
        ("CONCLUSIONS", "RESULTS", "Interpretive Finding Overlap"),
        ("RESULTS", "METHODS", "Quantitative Protocol Attribution"),
        ("METHODS", "RESULTS", "Procedural Measurement Overlap"),
    ]
    extracted = []
    for true_lbl, pred_lbl, description in modes:
        true_id = [k for k, v in id2label.items() if v == true_lbl][0]
        pred_id = [k for k, v in id2label.items() if v == pred_lbl][0]
        mask = (y_true == true_id) & (y_pred == pred_id)
        matching_indices = np.where(mask)[0]
        for idx in matching_indices[:n_samples_per_mode]:
            row = df.iloc[idx]
            extracted.append({
                "failure_mode": description,
                "true_label": true_lbl,
                "predicted_label": pred_lbl,
                "relative_position": float(round(row.get("relative_position", 0.0), 3)),
                "sentence_text": str(row["text"]),
                "diagnostic_reason": (
                    "Linguistic overlap between declarative context and prospective research objective."
                    if "Aim" in description else
                    "Interpretive clinical claim citing quantitative metrics or $p$-values."
                    if "Finding" in description else
                    "Methodological sentence stating sample sizes, dosage, or measurement procedures."
                ),
            })
    return extracted

