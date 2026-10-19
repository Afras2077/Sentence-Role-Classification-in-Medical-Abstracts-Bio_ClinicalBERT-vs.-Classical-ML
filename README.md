# Sentence-Role Classification in Medical Abstracts

## Fine-Tuned Biomedical Transformers vs. Classical Machine Learning on PubMed RCT

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Course:** AI2002 - Artificial Intelligence  
> **Project Code:** F26-07 | Track A (Research & Development)  
> **Instructor:** Hajra Waheed  
> **Department:** Department of Data Science, National University of Computer and Emerging Sciences  
> **Team Members:**  
> - **Azan Wasty** (24L-2557)  
> - **Nauman Iqbal** (24L-2565)  
> - **Afras Shahnawaz** (24L-2547)  

---

## 📖 Table of Contents

- [Overview & Problem Statement](#-overview--problem-statement)
- [Research Questions](#-research-questions)
- [Dataset (PubMed 20k RCT)](#-dataset-pubmed-20k-rct)
- [Methodology & Model Pipeline](#-methodology--model-pipeline)
- [Key Experimental Results (Phase 3)](#-key-experimental-results-phase-3)
  - [Overall Test Set Performance](#overall-test-set-performance)
  - [Per-Class Breakdown](#per-class-breakdown)
  - [Statistical Significance Testing](#statistical-significance-testing)
  - [Accuracy vs. Efficiency Trade-Off](#accuracy-vs-efficiency-trade-off)
  - [Structural Position Extension](#structural-position-extension)
- [Error Analysis](#-error-analysis)
- [Repository Structure](#-repository-structure)
- [Quickstart & Reproduction Guide](#-quickstart--reproduction-guide)
- [References & Citation](#-references--citation)

---

## 🔬 Overview & Problem Statement

Randomized controlled trial (RCT) abstracts are the cornerstone of evidence-based clinical medicine. However, a significant portion of published abstracts are **unstructured**, lacking distinct section headers. Automatically classifying each sentence into its rhetorical role enables clinicians and automated evidence-synthesis tools to extract key findings rapidly without reading entire publications.

In this project, each sentence from an RCT abstract is classified into one of five functional roles:
1. `BACKGROUND`: Context, epidemiological framing, and study rationale.
2. `OBJECTIVE`: Hypotheses, primary endpoints, and targeted research questions.
3. `METHODS`: Trial design, intervention protocols, sample sizes, and blinding.
4. `RESULTS`: Quantitative measurements, effect sizes, statistical significance ($p$-values, CIs).
5. `CONCLUSIONS`: Clinical implications, interpretations, and future directions.

---

## ❓ Research Questions

- **RQ1 (Accuracy):** How accurately can classical machine learning models (TF-IDF + linear / tree-based classifiers) classify rhetorical sentence roles based solely on sentence text?
- **RQ2 (Transformer Advantage & Significance):** Does a specialized domain transformer (`emilyalsentzer/Bio_ClinicalBERT`) provide a statistically significant improvement over classical baselines, as confirmed by non-parametric hypothesis testing?
- **RQ3 (Efficiency Trade-Off):** What is the practical trade-off between transformer performance and classical ML computational efficiency in terms of training time, inference latency, and hardware constraints?

---

## 📊 Dataset (PubMed 20k RCT)

We benchmark on the official **PubMed 20k RCT** corpus ([Dernoncourt & Lee, 2017](https://arxiv.org/abs/1710.06071)):

| Property | Value |
|---|---|
| **Dataset Source** | [Franck-Dernoncourt/pubmed-rct](https://github.com/Franck-Dernoncourt/pubmed-rct) |
| **Splits Strategy** | Abstract-level disjoint splits (zero abstract leakage across splits) |
| **Training Sentences** | 180,040 sentences (~15,000 abstracts) |
| **Validation (Dev) Sentences** | 30,212 sentences (~2,500 abstracts) |
| **Test Sentences** | 30,135 sentences (~2,500 abstracts) |
| **Target Classes** | 5 (`BACKGROUND`, `OBJECTIVE`, `METHODS`, `RESULTS`, `CONCLUSIONS`) |
| **Primary Evaluation Metric** | **Macro F1** (handles severe class imbalance) |

### Class Distribution (Training Split)

```
METHODS      [33.0%]  █████████████████████████████████   59,353
RESULTS      [32.2%]  ████████████████████████████████    57,953
CONCLUSIONS  [15.1%]  ███████████████                     27,168
BACKGROUND   [12.1%]  ████████████                        21,727
OBJECTIVE    [ 7.7%]  ████████                            13,839
```

> **Preprocessing Philosophy:** We intentionally preserve numbers, percentages, punctuation, and stopwords. Numbers and symbols are distinctive indicators for `RESULTS`, while grammatical markers and tense words (e.g., *"was"*, *"were"* vs. *"will"*, *"should"*) critically separate `METHODS` from `CONCLUSIONS`.

---

## ⚙️ Methodology & Model Pipeline

### 1. Classical Machine Learning Family
- **Feature Extraction:** Sublinear TF-IDF scaling ($1 + \log(\text{tf})$), word 1–2 grams, minimum document frequency $\ge 3$.
  - **200,000 features:** Logistic Regression.
  - **20,000 features:** Random Forest and XGBoost (reduces tree splitting over dense representations).
- **Classifiers:**
  - **SGD Logistic Regression:** Log-loss, $L_2$ penalty ($\alpha = 2 \times 10^{-6}$), 30 epochs, balanced class weights.
  - **Random Forest:** 100 estimators, $\sqrt{p}$ features per split, min leaf 3, balanced subsample weights.
  - **XGBoost:** 300 estimators, depth 6, learning rate 0.2, subsample 0.8, colsample 0.5.

### 2. Biomedical Transformer Family
- **Model:** [`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) (108.3M parameters, pretrained on PubMed abstracts and MIMIC-III clinical records).
- **Fine-Tuning:** HuggingFace `Trainer`, max token length 128 (dynamic batch padding), batch size 32, 5 epochs, learning rate $2 \times 10^{-5}$ with linear decay, 6% warmup (1,688 / 28,135 steps), weight decay 0.01, FP16 mixed precision on Tesla T4 GPU.
- **Checkpoint Selection:** Best epoch selected via validation (dev) Macro F1 (Epoch 3 checkpoint: Macro F1 = 0.8253).

---

## 📈 Key Experimental Results (Phase 3)

All reported scores are evaluated on the held-out **PubMed 20k RCT test set (30,135 sentences)**.

### Overall Test Set Performance

| Model | Accuracy | Weighted F1 | Macro F1 | Cohen's Kappa | Dev Macro F1 | Training Time | Inference Latency (per 1k sents) | Hardware |
|---|---|---|---|---|---|---|---|---|
| **Bio_ClinicalBERT [text]** | **0.8698** | **0.8689** | **0.8143** | **0.8247** | 0.8253 | 4,999.7 s | 1.656 s | GPU (T4) |
| **LogReg (SGD) [text]** | 0.8236 | 0.8232 | 0.7630 | 0.7630 | 0.7733 | **9.3 s** | 0.073 s | CPU |
| **XGBoost [text]** | 0.7862 | 0.7891 | 0.7305 | 0.7149 | 0.7392 | 78.6 s | **0.045 s** | CPU |
| **Random Forest [text]** | 0.7722 | 0.7726 | 0.7135 | 0.6946 | 0.7230 | 159.8 s | 0.067 s | CPU |
| *LogReg + position (extension)* | *0.8792* | *0.8788* | *0.8295* | *0.8378* | *0.8329* | *11.7 s* | *0.073 s* | *CPU* |

### Per-Class Breakdown

| Rhetorical Role | LogReg F1 | Bio_ClinicalBERT Precision | Bio_ClinicalBERT Recall | Bio_ClinicalBERT F1 | Δ F1 (BERT − LogReg) | Test Support |
|---|---|---|---|---|---|---|
| `BACKGROUND` | 0.655 | 0.701 | 0.752 | **0.726** | **+0.071** | 3,621 |
| `OBJECTIVE` | 0.614 | 0.699 | 0.614 | **0.654** | **+0.040** | 2,333 |
| `METHODS` | 0.904 | 0.927 | 0.950 | **0.938** | **+0.034** | 9,897 |
| `RESULTS` | 0.885 | 0.917 | 0.923 | **0.920** | **+0.035** | 9,713 |
| `CONCLUSIONS` | 0.757 | 0.863 | 0.807 | **0.834** | **+0.077** | 4,571 |

### Statistical Significance Testing

We assess whether Bio_ClinicalBERT's gains are statistically significant using:
1. **McNemar's Exact Test** on discordant test predictions.
2. **Paired Abstract-Level Bootstrap** (1,000 resamples of full abstracts to account for within-abstract sentence correlation).
3. **Bonferroni-corrected threshold:** $\alpha = 0.05 / 3 = 0.0167$.

| Baseline Model | Macro F1 Diff (BERT − Baseline) | 95% Bootstrap CI | Only Baseline Correct | Only BERT Correct | McNemar $p$-value | Significant? |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **+0.0512** | **[+0.0456, +0.0570]** | 1,046 | 2,439 | $2.3 \times 10^{-126}$ | **Yes** ($p < \alpha_{\text{Bonf}}$) |
| **XGBoost** | **+0.0838** | **[+0.0775, +0.0898]** | 1,092 | 3,611 | $5.0 \times 10^{-311}$ | **Yes** ($p < \alpha_{\text{Bonf}}$) |
| **Random Forest** | **+0.1008** | **[+0.0943, +0.1070]** | 1,078 | 4,020 | $< 10^{-300}$ | **Yes** ($p < \alpha_{\text{Bonf}}$) |

### Accuracy vs. Efficiency Trade-Off
- **Training Speedup:** SGD Logistic Regression trains in **9.3 seconds** on CPU vs. **4,999.7 seconds (~83 minutes)** for Bio_ClinicalBERT on GPU (**~540× faster**).
- **Inference Latency:** Logistic Regression processes 1,000 sentences in **0.073 seconds**, compared to **1.656 seconds** for Bio_ClinicalBERT (**~23× faster**).
- **Practical Takeaway:** Logistic Regression captures **~94% of Bio_ClinicalBERT's Macro F1** while operating with zero GPU requirement, making it an exceptional candidate for edge devices or high-throughput indexing pipelines.

### Structural Position Extension
Adding 10-bin relative position deciles, boundary flags, numeric content, and log sentence length to Logistic Regression boosts Macro F1 from **0.7630 to 0.8295**, surpassing text-only Bio_ClinicalBERT (0.8143). This demonstrates that structural abstract positioning is a dominant predictive feature for rhetorical roles.

---

## 🔍 Error Analysis

- Bio_ClinicalBERT misclassified 3,923 out of 30,135 sentences (13.0% error rate).
- **Primary Confusion Modes:**
  1. `OBJECTIVE` $\rightarrow$ `BACKGROUND` (732 sentences) & `BACKGROUND` $\rightarrow$ `OBJECTIVE` (528 sentences).
  2. `CONCLUSIONS` $\rightarrow$ `RESULTS` (493 sentences).
  3. `RESULTS` $\rightarrow$ `METHODS` (442 sentences).
- **Diagnostic Finding:** Position decile accuracy demonstrates that both model families suffer most at the abstract opening ($d=0$, where Background and Objective semantic boundaries blur) and abstract closing ($d=9$, where summary statements mirror result descriptions).

---

## 📂 Repository Structure

```
esi-triage-ml-vs-transformers/
│
├── configs/
│   └── config.yaml                     # Central pipeline hyperparameters & paths
│
├── data/
│   ├── raw/                            # PubMed 20k RCT raw splits (.txt files)
│   │   ├── train.txt
│   │   ├── dev.txt
│   │   └── test.txt
│   └── processed/                      # Preprocessed CSV splits (.gitkeep)
│
├── docs/                               # Project milestone documentation & reports
│   ├── F26-07-P1.pdf                   # Phase 1: Project Proposal
│   ├── F26-07-P2.pdf                   # Phase 2: Dataset Report & EDA
│   └── F26-07-P3.pdf                   # Phase 3: Comprehensive Benchmarking Report
│
├── models/
│   ├── classical/                      # Saved .joblib model weights & vectorisers
│   └── transformers/                   # Fine-tuned Bio_ClinicalBERT checkpoints
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Phase 2 EDA & dataset analysis
│   ├── F26-07.ipynb                    # Phase 3 full Colab training & evaluation notebook
│   └── README_notebooks.md             # Guide to running notebooks
│
├── pubmed-rct/                         # Cloned benchmark repository
│   └── PubMed_20k_RCT/
│
├── results/
│   ├── figures/                        # Confusion matrices, ROC curves, bar charts
│   │   ├── confusion_matrix_bio_clinicalbert.png
│   │   ├── accuracy_by_position.png
│   │   └── model_comparison.png
│   ├── metrics/                        # CSV tables of Phase 3 test performance
│   │   ├── phase3_metrics.csv
│   │   ├── phase3_per_class_metrics.csv
│   │   ├── phase3_significance.csv
│   │   └── phase3_dev_epoch_history.csv
│   └── reports/                        # Detailed Phase 3 markdown summary report
│       └── phase3_summary_report.md
│
├── scripts/
│   └── generate_phase3_artifacts.py    # Generates results tables and figures
│
├── src/                                # Modular, reusable Python package
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocess.py               # PubMed RCT raw parser & text cleaner
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py      # TF-IDF & structural positional features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classical/
│   │   │   ├── __init__.py
│   │   │   └── train_classical.py      # SGD LogReg, Random Forest, XGBoost
│   │   └── transformers/
│   │       ├── __init__.py
│   │       └── train_transformer.py    # Bio_ClinicalBERT Trainer setup
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py                 # Metrics, McNemar, paired cluster-bootstrap
│   └── utils/
│       ├── __init__.py
│       └── helpers.py                  # Seed setting, config loading, device helpers
│
├── tests/                              # Unit test suite
│   ├── __init__.py
│   ├── test_data_parser.py             # Abstract parsing & boundary tests
│   ├── test_features.py                # Feature extraction tests
│   └── test_evaluation.py              # Statistical test & metric tests
│
├── requirements.txt                    # Project dependencies
└── README.md                           # Main repository documentation
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup

```bash
git clone https://github.com/Afras2077/esi-triage-ml-vs-transformers.git
cd esi-triage-ml-vs-transformers

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Preprocess Raw Data

Convert raw PubMed RCT `.txt` splits into structured CSVs:

```bash
python -m src.data.preprocess
```

### 3. Run Classical ML Baseline

Fit 200k/20k TF-IDF vectorizers and train classical classifiers:

```bash
python -c "
from src.utils.helpers import load_config, set_seed
from src.data.preprocess import parse_pubmed_rct
from src.models.classical.train_classical import train_classical_pipeline

cfg = load_config()
set_seed(cfg['project']['seed'])
tr = parse_pubmed_rct(cfg['data']['train_raw'])
dv = parse_pubmed_rct(cfg['data']['dev_raw'])
te = parse_pubmed_rct(cfg['data']['test_raw'])
train_classical_pipeline(tr, dv, te, cfg['project']['label2id'])
"
```

### 4. Run Unit Tests

```bash
python -m unittest tests/test_data_parser.py
```

### 5. Colab Notebook Execution (Phase 3)
Open `notebooks/F26-07.ipynb` in [Google Colab](https://colab.research.google.com/), enable a Tesla T4 GPU (`Runtime -> Change runtime type -> T4`), and run all cells to reproduce the complete Bio_ClinicalBERT fine-tuning and significance tests.

---

## 📚 References & Citation

1. **Dernoncourt, F., & Lee, J. Y.** (2017). *PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts*. IJCNLP 2017. [arXiv:1710.06071](https://arxiv.org/abs/1710.06071).
2. **Alsentzer, E., Murphy, J. R., Boag, W., Weng, W. H., Jindi, D., Naumann, T., & McDermott, M.** (2019). *Publicly Available Clinical BERT Embeddings*. NAACL Clinical NLP Workshop. [arXiv:1904.03323](https://arxiv.org/abs/1904.03323).
3. **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K.** (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805).
4. **Pedregosa, F., et al.** (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
5. **Chen, T., & Guestrin, C.** (2016). *XGBoost: A Scalable Tree Boosting System*. ACM KDD.
