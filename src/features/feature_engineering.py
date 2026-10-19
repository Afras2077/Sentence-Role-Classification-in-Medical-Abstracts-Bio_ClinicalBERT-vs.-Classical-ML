from __future__ import annotations
"""
Feature engineering module for PubMed RCT sentence classification.
Provides TF-IDF vectorizers (200k for linear, 20k for tree models) and
structural sentence-level features (position deciles, boundary flags, numeric indicators).
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional, Tuple


def build_tfidf(
    max_features: int = 200000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 3,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    Constructs a TF-IDF vectoriser tuned for medical abstract text.
    Uses sublinear term frequency scaling and word 1-2 grams.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )


def extract_position_features(
    df: pd.DataFrame,
    mu: Optional[float] = None,
    sd: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Extracts positional and structural attributes for each sentence:
    - 10-bin one-hot encoding of relative position in the abstract.
    - Binary boundary indicators: is_first_sentence, is_last_sentence.
    - Numeric content indicator: presence of digits or percentage signs.
    - Standardised log character length.
    """
    log_lengths = np.log1p(df["line_length"].values)
    if mu is None:
        mu = float(log_lengths.mean())
    if sd is None or sd == 0:
        sd = float(log_lengths.std()) if float(log_lengths.std()) > 0 else 1.0

    std_log_len = ((log_lengths - mu) / sd)[:, None]

    # Position bins: 0 to 9
    rel_pos = df["relative_position"].values
    pos_bin = np.minimum((rel_pos * 10).astype(int), 9)
    pos_onehot = np.eye(10)[pos_bin]

    # Binary flags: first line, last line, contains digit or '%'
    is_first = (df["line_number"].values == 0).astype(int)
    is_last = (df["line_number"].values == (df["total_lines"].values - 1)).astype(int)
    has_num = df["text"].str.contains(r"\d|%", regex=True).fillna(False).astype(int).values

    flags = np.column_stack([is_first, is_last, has_num])
    return pos_onehot, flags, std_log_len, mu, sd


def build_structural_matrix(
    df: pd.DataFrame,
    mu: Optional[float] = None,
    sd: Optional[float] = None,
) -> Tuple[csr_matrix, float, float]:
    """
    Combines one-hot position deciles, boundary flags, and normalised log length
    into a scipy csr_matrix.
    """
    pos_onehot, flags, std_log_len, mean_val, std_val = extract_position_features(df, mu=mu, sd=sd)
    combined = np.hstack([pos_onehot, flags, std_log_len])
    return csr_matrix(combined), mean_val, std_val


def combine_tfidf_and_structural(X_tfidf: csr_matrix, X_struct: csr_matrix) -> csr_matrix:
    """Combines TF-IDF feature matrix with structural feature matrix."""
    return hstack([X_tfidf, X_struct], format="csr")


def save_vectorizer(vectorizer: TfidfVectorizer, out_path: str = "models/classical/tfidf.joblib"):
    """Serialises a fitted TF-IDF vectoriser to disk."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)
    print(f"[OK] TF-IDF vectoriser saved to {path}")


def load_vectorizer(in_path: str = "models/classical/tfidf.joblib") -> TfidfVectorizer:
    """Loads a serialized TF-IDF vectorizer."""
    return joblib.load(in_path)
