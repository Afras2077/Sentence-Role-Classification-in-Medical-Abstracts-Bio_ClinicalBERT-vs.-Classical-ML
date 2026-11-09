from __future__ import annotations
"""
Unit tests for the Improved Transformer components and Error Analysis / Fairness module.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.transformers.improved_transformer import (
    format_position_prompt,
    build_position_conditioned_texts,
)
from src.evaluation.error_analysis import (
    analyze_confusions,
    analyze_by_position,
    analyze_by_sentence_length,
    audit_bias_and_fairness,
    extract_qualitative_failures,
)


def test_format_position_prompt():
    prompt = format_position_prompt("Patients received oral metformin.", 2, 10, 0.20, prompt_style="structured")
    assert "[LINE 2 OF 10]" in prompt
    assert "[PCT 20%]" in prompt
    assert "Patients received oral metformin." in prompt


def test_build_position_conditioned_texts():
    df = pd.DataFrame({
        "text": ["First sentence.", "Middle sentence.", "Final sentence."],
        "line_number": [0, 1, 2],
        "total_lines": [3, 3, 3],
        "relative_position": [0.0, 0.333, 0.667],
    })
    prompts = build_position_conditioned_texts(df)
    assert len(prompts) == 3
    assert "[LINE 0 OF 3]" in prompts[0]
    assert "[LINE 2 OF 3]" in prompts[2]


def test_analyze_confusions():
    labels = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
    # 0 is BACKGROUND, 1 is OBJECTIVE
    y_true = np.array([1, 1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1, 0])
    df_conf = analyze_confusions(y_true, y_pred, labels)
    assert not df_conf.empty
    top_err = df_conf.iloc[0]
    assert top_err["True_Label"] == "OBJECTIVE"
    assert top_err["Predicted_Label"] == "BACKGROUND"
    assert top_err["Error_Count"] == 2


def test_analyze_by_position():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 0, 4])  # index 3 is error
    rel_pos = np.array([0.05, 0.15, 0.25, 0.35, 0.95])
    df_pos = analyze_by_position(y_true, y_pred, rel_pos, n_bins=10)
    assert len(df_pos) == 10
    # Decile 3 should have 1 sentence and 1 error
    row_3 = df_pos[df_pos["Position_Decile"] == 3].iloc[0]
    assert row_3["Total_Sentences"] == 1
    assert row_3["Errors"] == 1
    assert row_3["Accuracy"] == 0.0


def test_audit_bias_and_fairness():
    labels = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
    y_true = np.array([0, 1, 2, 3, 4, 1, 2, 3])
    y_pred = np.array([0, 0, 2, 3, 4, 1, 2, 3])  # one error on OBJECTIVE
    df = pd.DataFrame({
        "line_length": [50, 60, 120, 150, 200, 70, 130, 160],
        "relative_position": [0.0, 0.05, 0.3, 0.5, 0.95, 0.08, 0.4, 0.6],
    })
    fairness = audit_bias_and_fairness(y_true, y_pred, df, labels)
    assert "class_recalls" in fairness
    assert "recall_disparity_ratio" in fairness
    assert "length_disparity_ratio" in fairness
    assert "boundary_disparity_ratio" in fairness
    assert 0.0 <= fairness["recall_disparity_ratio"] <= 1.0


def test_extract_qualitative_failures():
    id2label = {0: "BACKGROUND", 1: "OBJECTIVE", 2: "METHODS", 3: "RESULTS", 4: "CONCLUSIONS"}
    df = pd.DataFrame({
        "text": [
            "This study aimed to investigate drug efficacy.",
            "Normal saline was administered intravenously.",
        ],
        "relative_position": [0.05, 0.45],
    })
    y_true = np.array([1, 2])
    y_pred = np.array([0, 3])  # 1->0 (OBJ->BG), 2->3 (METH->RES)
    failures = extract_qualitative_failures(df, y_true, y_pred, id2label, n_samples_per_mode=1)
    assert len(failures) == 2
    assert failures[0]["true_label"] == "OBJECTIVE"
    assert failures[0]["predicted_label"] == "BACKGROUND"

