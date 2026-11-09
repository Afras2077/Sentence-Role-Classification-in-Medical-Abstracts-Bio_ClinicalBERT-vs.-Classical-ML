from __future__ import annotations
"""
End-to-End Live Training, Inference, and Complete Evaluation Script for Phase 4.

Executes live:
1. Raw PubMed RCT Data Ingestion (Train, Dev, Test splits)
2. Positional Prompt Conditioning ([LINE i OF n] [PCT p%] <text>)
3. Inverse Class Weight Calculation
4. Transformer Fine-Tuning via HuggingFace Trainer
5. Inference on 30,135 Test Sentences (Predictions & Probabilities)
6. Live Metric Computation (Accuracy, Weighted F1, Macro F1, Cohen's Kappa)
7. Per-Class Precision, Recall, F1, Support Table
8. Positional Error Analysis across 10 Abstract Deciles
9. Syntactic Error Analysis across 4 Sentence Length Quartiles
10. Live Algorithmic Fairness Audit (Recall parity, Boundary parity, Length parity)
11. Statistical Significance vs Baseline (McNemar & Paired Cluster Bootstrap)
12. Direct CSV Export of all computed results
"""

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from src.data.preprocess import parse_pubmed_rct, LABELS, LABEL2ID, ID2LABEL
from src.evaluation.error_analysis import (
    analyze_confusions,
    analyze_by_position,
    analyze_by_sentence_length,
    audit_bias_and_fairness,
)


class PositionDataset(torch.utils.data.Dataset):
    """Tokenized Dataset with structural prompt conditioning."""

    def __init__(self, texts: List[str], labels: Optional[List[int]], tokenizer, max_length: int = 128):
        self.labels = labels
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length, padding=False)

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class WeightedTrainer(Trainer):
    """HuggingFace Trainer with class-weighted cross-entropy loss."""

    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def build_prompts(df: pd.DataFrame) -> List[str]:
    """Prefixes each sentence with structural position tokens."""
    prompts = []
    for _, r in df.iterrows():
        line_num = int(r.get("line_number", 0))
        tot_lines = int(r.get("total_lines", 1))
        pct = int(round(float(r.get("relative_position", 0.0)) * 100))
        txt = str(r["text"])
        prompts.append(f"[LINE {line_num} OF {tot_lines}] [PCT {pct}%] {txt}")
    return prompts


def fast_macro_f1(y: np.ndarray, p: np.ndarray, k: int = 5) -> float:
    """Fast macro-F1 calculation for paired bootstrap iterations."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    return float(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean())


def run_paired_bootstrap(
    y_true: np.ndarray,
    p_base: np.ndarray,
    p_imp: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Paired abstract-level cluster bootstrap (1,000 resamples)."""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    group_map = {g: np.where(groups == g)[0] for g in unique_groups}

    diffs = np.zeros(n_boot)
    for b in range(n_boot):
        sample_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        idx_list = [group_map[g] for g in sample_groups]
        idx = np.concatenate(idx_list) if idx_list else np.array([], dtype=int)
        if len(idx) == 0:
            continue
        y_b = y_true[idx]
        f1_imp = fast_macro_f1(y_b, p_imp[idx])
        f1_base = fast_macro_f1(y_b, p_base[idx])
        diffs[b] = f1_imp - f1_base

    mean_diff = float(np.mean(diffs))
    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    return mean_diff, ci_low, ci_high


def run_pipeline(
    train_path: str = "data/raw/train.txt",
    dev_path: str = "data/raw/dev.txt",
    test_path: str = "data/raw/test.txt",
    model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    output_dir: str = "results/live_run",
    epochs: int = 5,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    train_subsample: Optional[int] = None,
    use_position_prompts: bool = True,
    use_class_weights: bool = True,
    baseline_preds_path: Optional[str] = None,
    seed: int = 42,
):
    # 1. Environment & Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    has_gpu = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_gpu else "CPU"
    print(f"[*] Starting Pipeline on: {device_name}")
    print(f"[*] Model: {model_name}")

    # 2. Ingestion & Parsing
    print("[*] Parsing raw RCT splits...")
    train_df = parse_pubmed_rct(train_path)
    dev_df = parse_pubmed_rct(dev_path)
    test_df = parse_pubmed_rct(test_path)

    if train_subsample is not None and train_subsample < len(train_df):
        print(f"[*] Subsampling training set to {train_subsample} sentences for quick execution...")
        train_df = train_df.sample(n=train_subsample, random_state=seed).reset_index(drop=True)

    print(f"    Train: {len(train_df):,} sents | Dev: {len(dev_df):,} sents | Test: {len(test_df):,} sents")

    # 3. Position Conditioning
    if use_position_prompts:
        print("[*] Building position-conditioned prompt sequences...")
        train_texts = build_prompts(train_df)
        dev_texts = build_prompts(dev_df)
        test_texts = build_prompts(test_df)
    else:
        train_texts = train_df["text"].tolist()
        dev_texts = dev_df["text"].tolist()
        test_texts = test_df["text"].tolist()

    y_train = [LABEL2ID[l] for l in train_df["label"]]
    y_dev = [LABEL2ID[l] for l in dev_df["label"]]
    y_test = np.array([LABEL2ID[l] for l in test_df["label"]])

    # 4. Class Weights
    class_weights_t = None
    if use_class_weights:
        weights = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=np.array(y_train))
        class_weights_t = torch.tensor(weights, dtype=torch.float)
        print(f"[*] Computed inverse class weights: {np.round(weights, 4).tolist()}")

    # 5. Tokenization & Datasets
    print(f"[*] Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = PositionDataset(train_texts, y_train, tokenizer, max_length=max_length)
    dev_ds = PositionDataset(dev_texts, y_dev, tokenizer, max_length=max_length)
    test_ds = PositionDataset(test_texts, None, tokenizer, max_length=max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 6. Model Setup
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    steps_per_epoch = math.ceil(len(train_ds) / batch_size)
    warmup_steps = int(0.06 * steps_per_epoch * epochs)

    args = TrainingArguments(
        output_dir=str(out_p / "ckpt"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=has_gpu,
        seed=seed,
        logging_steps=100,
        report_to="none",
    )

    def compute_eval_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
            "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "cohen_kappa": float(cohen_kappa_score(labels, preds)),
        }

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=compute_eval_metrics,
        class_weights=class_weights_t,
    )

    # 7. Training
    print("[*] Commencing model training...")
    t_start = time.time()
    train_result = trainer.train()
    train_duration = time.time() - t_start
    print(f"[OK] Training completed in {train_duration:.1f} seconds ({train_duration/60:.1f} mins)")

    # 8. Test Inference
    print("[*] Running inference on held-out test set (30,135 sentences)...")
    t_infer_start = time.time()
    preds_output = trainer.predict(test_ds)
    infer_duration = time.time() - t_infer_start
    infer_per_1k = (infer_duration / len(test_ds)) * 1000.0

    logits = preds_output.predictions
    y_pred = np.argmax(logits, axis=-1)
    np.save(str(out_p / "live_test_predictions.npy"), y_pred)
    np.save(str(out_p / "live_test_logits.npy"), logits)

    # 9. Compute Overall Evaluation Metrics
    test_acc = float(accuracy_score(y_test, y_pred))
    test_w_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    test_m_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    test_kappa = float(cohen_kappa_score(y_test, y_pred))

    metrics_df = pd.DataFrame([{
        "Model": model_name,
        "Position_Conditioning": use_position_prompts,
        "Class_Weighted": use_class_weights,
        "Test_Accuracy": round(test_acc, 4),
        "Weighted_F1": round(test_w_f1, 4),
        "Macro_F1": round(test_m_f1, 4),
        "Cohen_Kappa": round(test_kappa, 4),
        "Train_Time_s": round(train_duration, 1),
        "Inference_s_per_1k": round(infer_per_1k, 3),
        "Hardware": device_name,
    }])
    metrics_df.to_csv(out_p / "live_overall_metrics.csv", index=False)
    print("\n=== LIVE TEST METRICS ===")
    print(metrics_df.to_string(index=False))

    # 10. Compute Per-Class Metrics
    p, r, f, s = precision_recall_fscore_support(y_test, y_pred, labels=range(len(LABELS)), zero_division=0)
    per_class_df = pd.DataFrame({
        "Rhetorical_Role": LABELS,
        "Precision": np.round(p, 4),
        "Recall": np.round(r, 4),
        "F1": np.round(f, 4),
        "Support": s,
    })
    per_class_df.to_csv(out_p / "live_per_class_metrics.csv", index=False)
    print("\n=== LIVE PER-CLASS METRICS ===")
    print(per_class_df.to_string(index=False))

    # 11. Error Analysis: Position Deciles
    pos_df = analyze_by_position(y_test, y_pred, test_df["relative_position"].values, n_bins=10)
    pos_df.to_csv(out_p / "live_error_analysis_position.csv", index=False)

    # 12. Error Analysis: Length Quartiles
    len_df = analyze_by_sentence_length(y_test, y_pred, test_df["line_length"].values, n_bins=4)
    len_df.to_csv(out_p / "live_error_analysis_length.csv", index=False)

    # 13. Algorithmic Bias & Fairness Audit
    fairness = audit_bias_and_fairness(y_test, y_pred, test_df, LABELS)
    fairness_df = pd.DataFrame([fairness])
    fairness_df.to_csv(out_p / "live_fairness_audit.csv", index=False)
    print("\n=== LIVE FAIRNESS AUDIT ===")
    print(f"Minority Recall ({fairness['minority_class']}): {fairness['minority_recall']}")
    print(f"Majority Recall ({fairness['majority_class']}): {fairness['majority_recall']}")
    print(f"Recall Disparity Ratio: {fairness['recall_disparity_ratio']}")
    print(f"Boundary Disparity Ratio: {fairness['boundary_disparity_ratio']}")
    print(f"Length Disparity Ratio: {fairness['length_disparity_ratio']}")

    # 14. Statistical Significance (if baseline predictions provided)
    if baseline_preds_path is not None and os.path.exists(baseline_preds_path):
        p_base = np.load(baseline_preds_path)
        a_ok = (p_base == y_test)
        b_ok = (y_pred == y_test)
        n_base_only = int(np.sum(a_ok & ~b_ok))
        n_imp_only = int(np.sum(~a_ok & b_ok))
        p_val = float(binomtest(n_base_only, n_base_only + n_imp_only, 0.5).pvalue)

        mean_diff, ci_l, ci_h = run_paired_bootstrap(
            y_test, p_base, y_pred, test_df["abstract_id"].values, n_boot=1000, seed=seed
        )
        sig_df = pd.DataFrame([{
            "Comparison": f"Improved vs {Path(baseline_preds_path).stem}",
            "Macro_F1_Diff": round(mean_diff, 4),
            "Bootstrap_95_CI": f"[{ci_l:.4f}, {ci_h:.4f}]",
            "Baseline_Only_Correct": n_base_only,
            "Improved_Only_Correct": n_imp_only,
            "McNemar_p_value": p_val,
        }])
        sig_df.to_csv(out_p / "live_significance.csv", index=False)
        print("\n=== LIVE STATISTICAL SIGNIFICANCE ===")
        print(sig_df.to_string(index=False))

    print(f"\n[OK] All live numbers and tables successfully saved to: {out_p.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Phase 4 Training & Evaluation")
    parser.add_argument("--train_path", type=str, default="data/raw/train.txt")
    parser.add_argument("--dev_path", type=str, default="data/raw/dev.txt")
    parser.add_argument("--test_path", type=str, default="data/raw/test.txt")
    parser.add_argument("--model_name", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    parser.add_argument("--output_dir", type=str, default="results/live_run")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--train_subsample", type=int, default=None)
    parser.add_argument("--baseline_preds", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_pipeline(
        train_path=args.train_path,
        dev_path=args.dev_path,
        test_path=args.test_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        train_subsample=args.train_subsample,
        baseline_preds_path=args.baseline_preds,
        seed=args.seed,
    )

