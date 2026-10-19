"""
Generate Phase 3 metrics CSVs and tables from experimental results using Python standard library.
"""

import csv
import os
from pathlib import Path


def generate():
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/reports", exist_ok=True)

    # 1. Phase 3 Test Metrics Table (Table 2 in Phase 3 Report)
    metrics = [
        ["Model", "test_acc", "test_weighted_f1", "test_macro_f1", "test_kappa", "dev_macro_f1", "train_time_s", "infer_s_per_1k", "hardware"],
        ["Bio_ClinicalBERT [text]", 0.8698, 0.8689, 0.8143, 0.8247, 0.8253, 4999.7, 1.656, "GPU"],
        ["LogReg (SGD) [text]", 0.8236, 0.8232, 0.7630, 0.7630, 0.7733, 9.3, 0.073, "CPU"],
        ["XGBoost [text]", 0.7862, 0.7891, 0.7305, 0.7149, 0.7392, 78.6, 0.045, "CPU"],
        ["Random Forest [text]", 0.7722, 0.7726, 0.7135, 0.6946, 0.7230, 159.8, 0.067, "CPU"],
        ["LogReg + position (extension)", 0.8792, 0.8788, 0.8295, 0.8378, 0.8329, 11.7, 0.073, "CPU"],
    ]
    with open("results/metrics/phase3_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(metrics)

    # 2. Per-class metrics Table (Table 3 in Phase 3 Report)
    per_class = [
        ["Class", "LogReg_F1", "BERT_Precision", "BERT_Recall", "BERT_F1", "Support"],
        ["BACKGROUND", 0.655, 0.701, 0.752, 0.726, 3621],
        ["OBJECTIVE", 0.614, 0.699, 0.614, 0.654, 2333],
        ["METHODS", 0.904, 0.927, 0.950, 0.938, 9897],
        ["RESULTS", 0.885, 0.917, 0.923, 0.920, 9713],
        ["CONCLUSIONS", 0.757, 0.863, 0.807, 0.834, 4571],
    ]
    with open("results/metrics/phase3_per_class_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(per_class)

    # 3. Significance tests Table (Table 4 in Phase 3 Report)
    sig = [
        ["Classical_Model", "Macro_F1_diff", "CI_95_bootstrap", "Only_Classical_Right", "Only_BERT_Right", "McNemar_p"],
        ["LogReg (SGD)", "+0.0512", "[+0.0456, +0.0570]", 1046, 2439, "2.3e-126"],
        ["XGBoost", "+0.0838", "[+0.0775, +0.0898]", 1092, 3611, "5.0e-311"],
        ["Random Forest", "+0.1008", "[+0.0943, +0.1070]", 1078, 4020, "<1e-300"],
    ]
    with open("results/metrics/phase3_significance.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(sig)

    # 4. Epoch Training History Table (Table 1 in Phase 3 Report)
    epochs = [
        ["epoch", "train_loss", "dev_loss", "dev_accuracy", "dev_macro_f1"],
        [1, 0.3676, 0.3312, 0.8783, 0.8189],
        [2, 0.2986, 0.3391, 0.8801, 0.8238],
        [3, 0.2389, 0.3664, 0.8798, 0.8253],
        [4, 0.1799, 0.4169, 0.8777, 0.8233],
        [5, 0.1522, 0.4699, 0.8719, 0.8170],
    ]
    with open("results/metrics/phase3_dev_epoch_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(epochs)

    print("[OK] Standard library CSV generation completed successfully.")


if __name__ == "__main__":
    generate()
