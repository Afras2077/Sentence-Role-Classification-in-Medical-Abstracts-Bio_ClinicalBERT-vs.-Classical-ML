from __future__ import annotations
"""
Classical Machine Learning model training pipeline for PubMed 20k RCT sentence classification.
Implements SGD-trained Logistic Regression, Random Forest, XGBoost, and position-augmented Logistic Regression.
"""

import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.features.feature_engineering import (
    build_tfidf,
    build_structural_matrix,
    combine_tfidf_and_structural,
    save_vectorizer,
)


def get_classical_models(seed: int = 42) -> Dict[str, Any]:
    """
    Returns initialised instances of classical ML models adhering to
    the parameters specified in the proposal and Phase 3 report.
    """
    return {
        "logreg": SGDClassifier(
            loss="log_loss",
            alpha=2e-6,
            max_iter=30,
            tol=1e-4,
            class_weight="balanced",
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.2,
            subsample=0.8,
            colsample_bytree=0.5,
            eval_metric="mlogloss",
            random_state=seed,
        ),
    }


def train_model(
    model: Any,
    X_train: Any,
    y_train: np.ndarray,
    X_dev: Optional[Any] = None,
    X_test: Optional[Any] = None,
) -> Tuple[Any, float, Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    Fits a classical classifier, measuring wall-clock training time
    and test inference latency.
    """
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    preds_dev = model.predict(X_dev) if X_dev is not None else None

    infer_time = 0.0
    preds_test = None
    if X_test is not None:
        t_infer = time.time()
        preds_test = model.predict(X_test)
        infer_time = time.time() - t_infer

    return model, train_time, preds_dev, preds_test, infer_time


def train_classical_pipeline(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label2id: Dict[str, int],
    seed: int = 42,
    run_trees: bool = True,
    run_extension: bool = True,
    output_dir: str = "models/classical/",
) -> Dict[str, Any]:
    """
    End-to-end training pipeline for all classical ML models:
    1. Fits 200k-feature TF-IDF for linear models.
    2. Fits 20k-feature TF-IDF for tree models.
    3. Fits SGD Logistic Regression, Random Forest, and XGBoost.
    4. Fits position-augmented Logistic Regression (extension).
    5. Saves models and vectorisers to output_dir.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    y_tr = train_df["label"].map(label2id).values
    y_dv = dev_df["label"].map(label2id).values
    y_te = test_df["label"].map(label2id).values

    results = {}

    # 1. 200k TF-IDF for Logistic Regression
    print("[*] Fitting 200k-feature TF-IDF vectoriser...")
    t0_vec = time.time()
    vec_200k = build_tfidf(max_features=200000)
    Xtr_200k = vec_200k.fit_transform(train_df["text"])
    Xdv_200k = vec_200k.transform(dev_df["text"])
    Xte_200k = vec_200k.transform(test_df["text"])
    vec_time_te = time.time() - t0_vec
    save_vectorizer(vec_200k, f"{output_dir}/tfidf_200k.joblib")

    # 2. Train Logistic Regression (text-only)
    print("[*] Training LogReg (SGD) on 200k features...")
    models = get_classical_models(seed=seed)
    lr, tr_time, p_dv, p_te, inf_time = train_model(
        models["logreg"], Xtr_200k, y_tr, Xdv_200k, Xte_200k
    )
    joblib.dump(lr, f"{output_dir}/logreg.joblib")
    results["LogReg (SGD) [text]"] = {
        "model": lr,
        "preds_dev": p_dv,
        "preds_test": p_te,
        "train_time_s": tr_time,
        "infer_time_test_s": vec_time_te + inf_time,
    }

    # 3. 20k TF-IDF for Trees
    if run_trees:
        print("[*] Fitting 20k-feature TF-IDF vectoriser for tree models...")
        vec_20k = build_tfidf(max_features=20000)
        Xtr_20k = vec_20k.fit_transform(train_df["text"])
        Xdv_20k = vec_20k.transform(dev_df["text"])
        Xte_20k = vec_20k.transform(test_df["text"])
        save_vectorizer(vec_20k, f"{output_dir}/tfidf_20k.joblib")

        print("[*] Training Random Forest (100 trees)...")
        rf, tr_time_rf, p_dv_rf, p_te_rf, inf_rf = train_model(
            models["random_forest"], Xtr_20k, y_tr, Xdv_20k, Xte_20k
        )
        joblib.dump(rf, f"{output_dir}/random_forest.joblib")
        results["Random Forest [text]"] = {
            "model": rf,
            "preds_dev": p_dv_rf,
            "preds_test": p_te_rf,
            "train_time_s": tr_time_rf,
            "infer_time_test_s": inf_rf,
        }

        print("[*] Training XGBoost (300 trees)...")
        xgb, tr_time_xgb, p_dv_xgb, p_te_xgb, inf_xgb = train_model(
            models["xgboost"], Xtr_20k, y_tr, Xdv_20k, Xte_20k
        )
        joblib.dump(xgb, f"{output_dir}/xgboost.joblib")
        results["XGBoost [text]"] = {
            "model": xgb,
            "preds_dev": p_dv_xgb,
            "preds_test": p_te_xgb,
            "train_time_s": tr_time_xgb,
            "infer_time_test_s": inf_xgb,
        }

    # 4. Position Extension (LogReg + structural features)
    if run_extension:
        print("[*] Building structural position features for extension...")
        S_tr, mu, sd = build_structural_matrix(train_df)
        S_dv, _, _ = build_structural_matrix(dev_df, mu=mu, sd=sd)
        S_te, _, _ = build_structural_matrix(test_df, mu=mu, sd=sd)

        Xtr_ext = combine_tfidf_and_structural(Xtr_200k, S_tr)
        Xdv_ext = combine_tfidf_and_structural(Xdv_200k, S_dv)
        Xte_ext = combine_tfidf_and_structural(Xte_200k, S_te)

        print("[*] Training LogReg + position extension...")
        lr_ext = SGDClassifier(
            loss="log_loss",
            alpha=2e-6,
            max_iter=30,
            tol=1e-4,
            class_weight="balanced",
            random_state=seed,
        )
        lr_ext, tr_time_ext, p_dv_ext, p_te_ext, inf_ext = train_model(
            lr_ext, Xtr_ext, y_tr, Xdv_ext, Xte_ext
        )
        joblib.dump(lr_ext, f"{output_dir}/logreg_position_extension.joblib")
        results["LogReg + position (extension)"] = {
            "model": lr_ext,
            "preds_dev": p_dv_ext,
            "preds_test": p_te_ext,
            "train_time_s": tr_time_ext,
            "infer_time_test_s": inf_ext,
        }

    print(f"[OK] Classical ML training complete. Artifacts saved in {output_dir}")
    return results
