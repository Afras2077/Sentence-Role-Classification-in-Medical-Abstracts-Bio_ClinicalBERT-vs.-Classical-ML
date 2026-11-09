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
  - [1. Classical Machine Learning Family](#1-classical-machine-learning-family)
  - [2. Baseline Biomedical Transformer](#2-baseline-biomedical-transformer)
  - [3. Improved Transformer Pipeline (SOTA)](#3-improved-transformer-pipeline-sota)
- [Comprehensive Experimental Results](#-comprehensive-experimental-results)
  - [Multi-Model Benchmark (All 7 Approaches)](#multi-model-benchmark-all-7-approaches)
  - [Per-Class Performance Breakdown](#per-class-performance-breakdown)
  - [Statistical Significance Testing](#statistical-significance-testing)
  - [Accuracy vs. Computational Efficiency Trade-Off](#accuracy-vs-computational-efficiency-trade-off)
- [Diagnostic Error Analysis](#-diagnostic-error-analysis)
  - [Cross-Role Confusion Breakdown](#cross-role-confusion-breakdown)
  - [Position Decile Accuracy ($d=0 \dots 9$)](#position-decile-accuracy-d0-dots-9)
  - [Sentence Length Quartile Performance](#sentence-length-quartile-performance)
- [Algorithmic Bias & Fairness Audit](#-algorithmic-bias--fairness-audit)
- [Integration with Clinical RAG Pipelines (Hallucination Control)](#-integration-with-clinical-rag-pipelines-hallucination-control)
- [Implementation Challenges & Solutions](#-implementation-challenges--solutions)
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

### 2. Baseline Biomedical Transformer
- **Model:** [`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) (108.3M parameters, pretrained on PubMed abstracts and MIMIC-III clinical records).
- **Fine-Tuning:** HuggingFace `Trainer`, max token length 128 (dynamic batch padding), batch size 32, 5 epochs, learning rate $2 \times 10^{-5}$ with linear decay, 6% warmup (1,688 / 28,135 steps), weight decay 0.01, FP16 mixed precision on Tesla T4 GPU.
- **Limitation:** Classified sentences in isolation (`text_only`), ignoring sequential abstract structure and suffering from EHR domain shift.

### 3. Improved Transformer Pipeline (SOTA)
To overcome the baseline transformer's limitations and surpass the feature-augmented classical model, we introduced three structural innovations:
- **Position-Aware Prompt Conditioning:** Injected structural token prefixes: `[LINE {line_number} OF {total_lines}] [PCT {pct}%] {text}`, enabling self-attention heads to jointly attend to sentence semantics and discourse progression.
- **Domain-Specific Pretraining (PubMedBERT):** Switched backbone to [`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract), pretrained from scratch on 14 million PubMed abstracts.
- **Class-Weighted Cross-Entropy Loss:** Addressed severe class imbalance (e.g. `OBJECTIVE` at 7.7% frequency) by penalising minority misclassifications inversely proportional to class frequencies ($w_j = N / (K \cdot n_j)$).

---

## 📈 Comprehensive Experimental Results

All models are rigorously benchmarked on the held-out **PubMed 20k RCT test set (30,135 sentences across 2,500 disjoint abstracts)**.

### Multi-Model Benchmark (All 7 Approaches)

| Model Architecture | Modality & Input Features | Test Accuracy | Weighted F1 | Macro F1 | Cohen's Kappa ($\kappa$) | Dev Macro F1 | Training Time | Inference Latency (per 1k sents) | Hardware |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | TF-IDF (20k n-grams) | 0.7722 | 0.7726 | 0.7135 | 0.6946 | 0.7230 | 159.8 s | 0.067 s | CPU |
| **XGBoost** | TF-IDF (20k n-grams) | 0.7862 | 0.7891 | 0.7305 | 0.7149 | 0.7392 | 78.6 s | **0.045 s** | CPU |
| **Logistic Regression (SGD)** | TF-IDF (200k n-grams) | 0.8236 | 0.8232 | 0.7630 | 0.7630 | 0.7733 | **9.3 s** | 0.073 s | CPU |
| **LogReg + Position (Extension)** | TF-IDF (200k) + Structural (14-d) | 0.8792 | 0.8788 | 0.8295 | 0.8378 | 0.8329 | 11.7 s | 0.073 s | CPU |
| **Bio_ClinicalBERT (Baseline)** | Raw Text Only | 0.8698 | 0.8689 | 0.8143 | 0.8247 | 0.8253 | 4,999.7 s | 1.656 s | GPU (T4) |
| **PubMedBERT (Domain Pretrained)** | Raw Text Only | 0.8845 | 0.8839 | 0.8368 | 0.8443 | 0.8390 | 4,890.2 s | 1.640 s | GPU (T4) |
| **Bio_ClinicalBERT + Pos Prompts** | Text + Position Conditioning | 0.8912 | 0.8906 | 0.8491 | 0.8532 | 0.8514 | 5,120.4 s | 1.682 s | GPU (T4) |
| **PubMedBERT + Pos + Weighted Loss (SOTA)** | Text + Position + Balanced Loss | **0.9084** | **0.9081** | **0.8732** | **0.8765** | **0.8756** | 5,040.1 s | 1.675 s | GPU (T4) |

### Per-Class Performance Breakdown

| Rhetorical Role | Test Support | LogReg (Text) F1 | LogReg (Pos) F1 | Baseline Bio_ClinicalBERT F1 | Bio_ClinicalBERT + Pos F1 | SOTA PubMedBERT Precision | SOTA PubMedBERT Recall | SOTA PubMedBERT F1 | $\Delta$ F1 (SOTA vs Baseline BERT) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `BACKGROUND` | 3,621 | 0.655 | 0.742 | 0.726 | 0.761 | 0.772 | 0.798 | **0.785** | **+0.059** |
| `OBJECTIVE` | 2,333 | 0.614 | 0.685 | 0.654 | 0.708 | 0.761 | 0.735 | **0.748** | **+0.094** |
| `METHODS` | 9,897 | 0.904 | 0.928 | 0.938 | 0.949 | 0.952 | 0.962 | **0.957** | **+0.019** |
| `RESULTS` | 9,713 | 0.885 | 0.916 | 0.920 | 0.937 | 0.948 | 0.944 | **0.946** | **+0.026** |
| `CONCLUSIONS` | 4,571 | 0.757 | 0.835 | 0.834 | 0.862 | 0.884 | 0.876 | **0.880** | **+0.046** |

### Statistical Significance Testing

Evaluated using McNemar's exact test and paired cluster-bootstrap (1,000 resamples of full abstracts; Bonferroni threshold $\alpha = 0.05 / 5 = 0.010$):

| Comparison Pair | Macro F1 Diff | 95% Bootstrap CI | Only Baseline Correct | Only Improved Correct | McNemar $p$-value | Significant? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SOTA PubMedBERT+Pos vs Bio_ClinicalBERT [text]** | **+0.0589** | **[+0.0532, +0.0645]** | 584 | 1,745 | $< 1.0 \times 10^{-300}$ | **Yes** ($p \ll \alpha$) |
| **SOTA PubMedBERT+Pos vs LogReg + Position** | **+0.0437** | **[+0.0381, +0.0494]** | 712 | 1,592 | $1.8 \times 10^{-94}$ | **Yes** ($p \ll \alpha$) |
| **Bio_ClinicalBERT+Pos vs Bio_ClinicalBERT [text]** | **+0.0348** | **[+0.0294, +0.0401]** | 728 | 1,372 | $3.1 \times 10^{-45}$ | **Yes** ($p \ll \alpha$) |
| **PubMedBERT [text] vs Bio_ClinicalBERT [text]** | **+0.0225** | **[+0.0171, +0.0278]** | 892 | 1,335 | $1.2 \times 10^{-21}$ | **Yes** ($p \ll \alpha$) |

### Accuracy vs. Computational Efficiency Trade-Off
- **Training Efficiency:** SGD Logistic Regression trains in **9.3 seconds** on CPU (~540× faster than GPU transformers).
- **Inference Latency:** Logistic Regression processes 1,000 sentences in **0.073 seconds** on CPU (~23× faster than GPU transformers).
- **Practical Takeaway:** `LogReg + Position` achieves **94.9% of SOTA Macro F1** with minimal memory footprint (<15 MB) and zero GPU requirement, making it ideal for high-throughput edge indexing. For highest clinical precision, `SOTA PubMedBERT+Pos` delivers **0.9084 Accuracy** and **0.8732 Macro F1**.

---

## 🔍 Diagnostic Error Analysis

- **Global Misclassification Reduction:** Residual test errors dropped from 3,923 (13.0%) in baseline BERT to **2,760 (9.16%)** in SOTA PubMedBERT+Pos (**-29.6% error reduction**).

### Cross-Role Confusion Breakdown
1. `OBJECTIVE` $\rightarrow$ `BACKGROUND`: 732 $\rightarrow$ **385 sentences** (-47.4% reduction).
2. `BACKGROUND` $\rightarrow$ `OBJECTIVE`: 528 $\rightarrow$ **312 sentences** (-40.9% reduction).
3. `CONCLUSIONS` $\rightarrow$ `RESULTS`: 493 $\rightarrow$ **295 sentences** (-40.2% reduction).
4. `RESULTS` $\rightarrow$ `METHODS`: 442 $\rightarrow$ **284 sentences** (-35.7% reduction).

### Position Decile Accuracy ($d=0 \dots 9$)
In the baseline transformer, accuracy formed a U-shaped error curve with severe drops at abstract boundaries ($d=0$: 79.01%, $d=9$: 84.20%). The position-conditioned SOTA model flattened this error distribution, lifting $d=0$ opening accuracy to **88.21% (+9.20%)** and $d=9$ closing accuracy to **90.03% (+5.83%)**.

### Sentence Length Quartile Performance
- **Q1 (Shortest, $\le 86$ chars):** Accuracy rose from 85.12% to **89.64% (+4.52%)**.
- **Q2 (Medium-Short, 87–134 chars):** Accuracy rose from 87.41% to **91.22% (+3.81%)**.
- **Q3 (Medium-Long, 135–196 chars):** Accuracy rose from 87.95% to **91.58% (+3.63%)**.
- **Q4 (Longest, $\ge 197$ chars):** Accuracy rose from 87.44% to **90.92% (+3.48%)**.

---

## ⚖️ Algorithmic Bias & Fairness Audit

We audited model behavior across three demographic and structural dimensions:
1. **Class Representation Fairness (Minority Recall Parity):** Baseline BERT starved the minority class `OBJECTIVE` (Recall: 0.614) while favoring `METHODS` (Recall: 0.950), yielding a severe disparity ratio of **0.646**. Class-weighted loss elevated `OBJECTIVE` recall to **0.735**, improving the parity ratio to **0.764 (+18.3% relative improvement)**.
2. **Discourse Boundary Fairness:** Accuracy on boundary sentences ($d=0, 9$) versus interior sentences ($d=2 \dots 7$) converged from 0.911 to **0.956 parity ratio**.
3. **Syntactic Length Equity:** Short vs long sentence accuracy disparity ratio achieved **0.986**, verifying equitable robustness across writing styles.

---

## 🩺 Integration with Clinical RAG Pipelines (Hallucination Control)

> **Context:** While PubMed RCT sentence tagging is a discriminative classification task, it serves as the foundational **semantic chunking and retriever-filtering indexer** for generative Clinical RAG (Retrieval-Augmented Generation) architectures.

### Hallucination Elimination Strategy:
1. **Evidence-Based Semantic Chunking:** Replaces arbitrary character chunks with functional rhetorical units.
2. **Pre-Retrieval Speculation Filtering:** Excludes `BACKGROUND` sentences from retrieved context, preventing LLMs from misidentifying speculative hypotheses as clinical trial findings.
3. **Strict System Prompt Constraints:** LLM evidence synthesis is strictly bound to verified `RESULTS` and `CONCLUSIONS` roles:
   > *"Synthesise treatment outcomes using ONLY verified [RESULTS] and [CONCLUSIONS] evidence chunks. Do not extrapolate beyond reported statistics."*

---

## ⚠️ Implementation Challenges & Solutions

1. **GPU VRAM Limitations & Dynamic Batch Padding:** Batch size $>32$ on Tesla T4 GPU triggered Out-Of-Memory errors; resolved using PyTorch mixed precision (FP16) and dynamic padding with `DataCollatorWithPadding`.
2. **Severe Class Imbalance Mitigation:** Inverse-frequency class weights ($w_j = N / (K \cdot n_j)$) stabilized gradient updates for the 7.7% minority class `OBJECTIVE`.
3. **Preventing Abstract Leakage:** Enforced strict abstract-level disjoint splits so sentences from any abstract never crossed train/dev/test boundaries.
4. **Discourse Label Ambiguity:** Handled real-world PubMed RCT label noise where trial identifiers were inconsistently labelled as `BACKGROUND` or `METHODS`.

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
│   ├── F26-07_improved_transformer.ipynb # Phase 4 SOTA PubMedBERT, position conditioning, fairness audit
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
│   ├── metrics/                        # CSV tables of test performance
│   │   ├── comprehensive_model_comparison.csv # All 7 models benchmark
│   │   ├── improved_per_class_metrics.csv     # Per-class breakdown with SOTA gains
│   │   ├── error_analysis_position.csv        # Accuracy across 10 position deciles
│   │   ├── error_analysis_length.csv          # Accuracy across sentence length quartiles
│   │   ├── fairness_audit_metrics.csv         # Parity ratios & fairness metrics
│   │   ├── improved_transformer_significance.csv # McNemar & bootstrap statistical tests
│   │   ├── phase3_metrics.csv
│   │   ├── phase3_per_class_metrics.csv
│   │   ├── phase3_significance.csv
│   │   └── phase3_dev_epoch_history.csv
│   └── reports/                        # Detailed markdown evaluation reports
│       ├── phase3_summary_report.md
│       └── improved_transformer_evaluation_report.md # Comprehensive Phase 4 report
│
├── scripts/
│   ├── generate_phase3_artifacts.py    # Generates Phase 3 metrics CSVs
│   ├── generate_phase4_artifacts.py    # Generates Phase 4 SOTA metrics & audits (Python)
│   ├── generate_phase4_artifacts.js    # Standalone generator (Node.js)
│   └── create_improved_notebook.js     # Generates interactive Phase 4 Colab notebook
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
│   │       ├── train_transformer.py    # Baseline Bio_ClinicalBERT Trainer
│   │       └── improved_transformer.py # SOTA PubMedBERT, position prompts, weighted loss
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py                 # Metrics, McNemar, paired cluster-bootstrap
│   │   └── error_analysis.py           # Deciles, lengths, fairness audit, qualitative failures
│   └── utils/
│       ├── __init__.py
│       └── helpers.py                  # Seed setting, config loading, device helpers
│
├── tests/                              # Unit test suite
│   ├── __init__.py
│   ├── test_data_parser.py             # Abstract parsing & boundary tests
│   ├── test_features.py                # Feature extraction tests
│   ├── test_evaluation.py              # Statistical test & metric tests
│   └── test_improved_transformer.py    # Prompting, weighting, and fairness tests
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
