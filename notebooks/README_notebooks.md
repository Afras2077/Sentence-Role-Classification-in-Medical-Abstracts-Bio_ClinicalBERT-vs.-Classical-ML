# Notebook Directory

This directory contains the interactive Jupyter notebooks for Project **F26-07**:

### 1. `01_data_exploration.ipynb`
- **Phase 2 Exploratory Data Analysis (EDA)**.
- Parses raw PubMed 20k RCT abstracts.
- Extracts sentence position metadata (`line_number`, `total_lines`, `relative_position`, `line_length`).
- Visualises class distribution, abstract length distribution, label density by relative position, and sentence lengths.

### 2. `F26-07.ipynb`
- **Phase 3 Complete End-to-End Pipeline & Benchmarking Notebook**.
- Implemented and evaluated on Google Colab (Tesla T4 GPU).
- Contains:
  1. **Data Loading & Parsing**: `train.txt`, `dev.txt`, `test.txt` into structured DataFrames.
  2. **Classical ML Family**:
     - 200k-feature TF-IDF + SGD Logistic Regression (`alpha=2e-6`, `class_weight='balanced'`).
     - 20k-feature TF-IDF + Random Forest (100 trees) and XGBoost (300 trees, depth 6, GPU histogram).
  3. **Position Extension**:
     - Logistic Regression augmented with 10-bin position deciles, boundary flags, numeric indicators, and log sentence length.
  4. **Transformer Family**:
     - Fine-tuning `emilyalsentzer/Bio_ClinicalBERT` using HuggingFace Trainer (max length 128, batch size 32, 5 epochs, lr 2e-5, weight decay 0.01, fp16).
  5. **Evaluation & Statistical Significance**:
     - Test metrics (Accuracy, Weighted F1, Macro F1, Cohen's Kappa, inference latencies).
     - Per-class metrics for all 5 roles.
     - McNemar's exact test and cluster-bootstrap (1,000 resamples of abstracts) with 95% confidence intervals.
  6. **Error Analysis**:
     - Confusion matrices and error distribution across sentence position deciles.
