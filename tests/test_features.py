"""
Unit tests for feature engineering module.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from src.features.feature_engineering import (
    build_tfidf,
    extract_position_features,
    build_structural_matrix,
    combine_tfidf_and_structural,
)


def test_build_tfidf():
    vec = build_tfidf(max_features=50, ngram_range=(1, 2), min_df=1)
    corpus = ["The randomized clinical trial showed efficacy.", "Secondary outcomes were evaluated."]
    X = vec.fit_transform(corpus)
    assert X.shape[0] == 2
    assert X.shape[1] > 0


def test_extract_position_features():
    df = pd.DataFrame({
        "text": ["First sentence with 95% CI.", "Second sentence without numbers."],
        "line_number": [0, 1],
        "total_lines": [2, 2],
        "line_length": [27, 33],
        "relative_position": [0.0, 0.5],
    })
    pos_onehot, flags, std_log_len, mu, sd = extract_position_features(df)

    assert pos_onehot.shape == (2, 10)
    assert flags.shape == (2, 3)
    # Check flags for first line: is_first=1, is_last=0, has_num=1
    assert list(flags[0]) == [1, 0, 1]
    # Check flags for second line: is_first=0, is_last=1, has_num=0
    assert list(flags[1]) == [0, 1, 0]


def test_build_structural_matrix():
    df = pd.DataFrame({
        "text": ["Sentence one.", "Sentence two.", "Sentence three."],
        "line_number": [0, 1, 2],
        "total_lines": [3, 3, 3],
        "line_length": [13, 13, 15],
        "relative_position": [0.0, 0.333, 0.666],
    })
    mat, mu, sd = build_structural_matrix(df)
    # 10 position bins + 3 flags + 1 log_len = 14 features
    assert mat.shape == (3, 14)
