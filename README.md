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
- [Phase 3 Experimental Results (Baseline Benchmark)](#-phase-3-experimental-results-baseline-benchmark)
- [Phase 4 Experimental Results (Live Verified SOTA)](#-phase-4-experimental-results-live-verified-sota)
- [Diagnostic Error Analysis (Live Test Predictions)](#-diagnostic-error-analysis-live-test-predictions)
- [Algorithmic Bias & Fairness Audit (Live Audited Metrics)](#-algorithmic-bias--fairness-audit-live-audited-metrics)
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

## 📈 Phase 3 Experimental Results (Baseline Benchmark)

All reported baseline scores were empirically evaluated on the held-out **PubMed 20k RCT test set (30,135 sentences across 2,500 disjoint abstracts)**:

### Overall Test Set Performance

| Model | Accuracy | Weighted F1 | Macro F1 | Cohen's Kappa | Dev Macro F1 | Training Time | Inference Latency (per 1k sents) | Hardware |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Bio_ClinicalBERT [text]** | **0.8698** | **0.8689** | **0.8143** | **0.8247** | 0.8253 | 4,999.7 s | 1.656 s | GPU (T4) |
| **LogReg (SGD) [text]** | 0.8236 | 0.8232 | 0.7630 | 0.7630 | 0.7733 | **9.3 s** | 0.073 s | CPU |
| **XGBoost [text]** | 0.7862 | 0.7891 | 0.7305 | 0.7149 | 0.7392 | 78.6 s | **0.045 s** | CPU |
| **Random Forest [text]** | 0.7722 | 0.7726 | 0.7135 | 0.6946 | 0.7230 | 159.8 s | 0.067 s | CPU |
| *LogReg + position (extension)* | *0.8792* | *0.8788* | *0.8295* | *0.8378* | *0.8329* | *11.7 s* | *0.073 s* | *CPU* |

### Per-Class Breakdown

| Rhetorical Role | LogReg F1 | Bio_ClinicalBERT Precision | Bio_ClinicalBERT Recall | Bio_ClinicalBERT F1 | Δ F1 (BERT − LogReg) | Test Support |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `BACKGROUND` | 0.655 | 0.701 | 0.752 | **0.726** | **+0.071** | 3,621 |
| `OBJECTIVE` | 0.614 | 0.699 | 0.614 | **0.654** | **+0.040** | 2,333 |
| `METHODS` | 0.904 | 0.927 | 0.950 | **0.938** | **+0.034** | 9,897 |
| `RESULTS` | 0.885 | 0.917 | 0.923 | **0.920** | **+0.035** | 9,713 |
| `CONCLUSIONS` | 0.757 | 0.863 | 0.807 | **0.834** | **+0.077** | 4,571 |

### Statistical Significance Testing

Assessed using McNemar's exact test and paired cluster-bootstrap (1,000 resamples of full abstracts; Bonferroni threshold $\alpha = 0.05 / 3 = 0.0167$):

| Baseline Model | Macro F1 Diff (BERT − Baseline) | 95% Bootstrap CI | Only Baseline Correct | Only BERT Correct | McNemar $p$-value | Significant? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **+0.0512** | **[+0.0456, +0.0570]** | 1,046 | 2,439 | $2.3 \times 10^{-126}$ | **Yes** ($p < \alpha_{\text{Bonf}}$) |
| **XGBoost** | **+0.0838** | **[+0.0775, +0.0898]** | 1,092 | 3,611 | $5.0 \times 10^{-311}$ | **Yes** ($p < \alpha_{\text{Bonf}}$) |
| **Random Forest** | **+0.1008** | **[+0.0943, +0.1070]** | 1,078 | 4,020 | $< 10^{-300}$ | **Yes** ($p < \alpha_{\text{Bonf}}$) |

---

## 🏆 Phase 4 Experimental Results (Live Verified SOTA)

All Phase 4 results were generated via live GPU fine-tuning of `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` with position conditioning and balanced class weights, evaluated on the held-out **PubMed 20k RCT test set (30,135 sentences across 2,500 disjoint abstracts)**:

### Overall Benchmark Comparison (Classical vs Phase 3 vs Phase 4 SOTA)

| Model Architecture | Input Modality & Features | Test Accuracy | Weighted F1 | Macro F1 | Cohen's Kappa ($\kappa$) | Dev Macro F1 | Train Time (s) | Inference Latency (per 1k sents) | Hardware |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | TF-IDF (20k n-grams) | 0.7722 | 0.7726 | 0.7135 | 0.6946 | 0.7230 | 159.8 s | 0.067 s | CPU |
| **XGBoost** | TF-IDF (20k n-grams) | 0.7862 | 0.7891 | 0.7305 | 0.7149 | 0.7392 | 78.6 s | **0.045 s** | CPU |
| **Logistic Regression (SGD)** | TF-IDF (200k n-grams) | 0.8236 | 0.8232 | 0.7630 | 0.7630 | 0.7733 | **9.3 s** | 0.073 s | CPU |
| **LogReg + Position (Extension)** | TF-IDF (200k) + Structural (14-d) | 0.8792 | 0.8788 | 0.8295 | 0.8378 | 0.8329 | 11.7 s | 0.073 s | CPU |
| **Bio_ClinicalBERT [text] (Phase 3)** | Raw Text Only | 0.8698 | 0.8689 | 0.8143 | 0.8247 | 0.8253 | 4,999.7 s | 1.656 s | GPU (Tesla T4) |
| **PubMedBERT + Pos + Weighted (Phase 4 Live)** | Text + Position Conditioning + Class Weights | **0.9143** | **0.9140** | **0.8705** | **0.8850** | **0.8726** | 4,998.9 s | 1.723 s | GPU (Tesla T4) |

### Live Per-Class Performance Breakdown

| Rhetorical Role | Test Support | Precision | Recall | Phase 4 F1 | Phase 3 BERT F1 | Phase 3 LogReg F1 | $\Delta$ F1 (Phase 4 vs Phase 3 BERT) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `BACKGROUND` | 3,621 | 0.7829 | 0.8238 | **0.8029** | 0.7260 | 0.6550 | **+0.0769 (+10.6%)** |
| `OBJECTIVE` | 2,333 | 0.7238 | 0.6695 | **0.6956** | 0.6540 | 0.6140 | **+0.0416 (+6.4%)** |
| `METHODS` | 9,897 | 0.9435 | 0.9700 | **0.9566** | 0.9380 | 0.9040 | **+0.0186 (+2.0%)** |
| `RESULTS` | 9,713 | 0.9681 | 0.9277 | **0.9475** | 0.9200 | 0.8850 | **+0.0275 (+3.0%)** |
| `CONCLUSIONS` | 4,571 | 0.9387 | 0.9619 | **0.9502** | 0.8340 | 0.7570 | **+0.1162 (+13.9%)** |

---

## 🔍 Diagnostic Error Analysis (Live Test Predictions)

- **Global Misclassification Reduction:** Test errors dropped from 3,923 (13.0%) in baseline BERT to **2,582 (8.57%)** in Phase 4 PubMedBERT+Pos (**-34.2% error reduction; 1,341 fewer errors**).

### Primary Confusion Modes (True -> Predicted)
1. `OBJECTIVE` $\rightarrow$ `BACKGROUND`: 709 sentences (declarative aims overlapping with clinical context).
2. `BACKGROUND` $\rightarrow$ `OBJECTIVE`: 543 sentences (background sentences stating prospective trial questions).
3. `RESULTS` $\rightarrow$ `METHODS`: 428 sentences (outcomes describing participant numbers or treatment adherence).
4. `RESULTS` $\rightarrow$ `CONCLUSIONS`: 267 sentences (results summarizing clinical efficacy implications).
5. `METHODS` $\rightarrow$ `RESULTS`: 160 sentences (methodology citing numerical dosages or sample counts).
6. `CONCLUSIONS` $\rightarrow$ `RESULTS`: 123 sentences (concluding claims citing statistical outcomes).

### Position Decile Accuracy ($d=0 \dots 9$)
Evaluated live across 10 deciles of relative abstract position:
- **Decile 0 (0–10%):** 78.50% (Abstract opening where Background and Objective semantics blend)
- **Decile 1 (10–20%):** 84.54%
- **Decile 2 (20–30%):** 90.35%
- **Decile 3 (30–40%):** 93.26%
- **Decile 4 (40–50%):** 95.83%
- **Decile 5 (50–60%):** 97.13%
- **Decile 6 (60–70%):** 96.15%
- **Decile 7 (70–80%):** 93.18%
- **Decile 8 (80–90%):** 93.93%
- **Decile 9 (90–100%):** **98.08%** (Near-perfect closing accuracy for conclusions)

### Sentence Length Quartile Performance
- **Q1 (Shortest: $\le 86$ chars):** **91.74%** accuracy
- **Q2 (Med-Short: 87–134 chars):** **91.01%** accuracy
- **Q3 (Med-Long: 135–196 chars):** **90.57%** accuracy
- **Q4 (Longest: $\ge 197$ chars):** **92.40%** accuracy

---

## ⚖️ Algorithmic Bias & Fairness Audit (Live Audited Metrics)

Audited across structural, syntactic, and class distribution dimensions:
1. **Class Representation Fairness (Minority Recall Parity):** Class weighting lifted `OBJECTIVE` recall from 0.6140 to **0.6695** (precision 0.7238), yielding a **0.6902 recall disparity ratio** against majority `METHODS` (0.9700 recall) without harming majority performance.
2. **Discourse Boundary Fairness:** Accuracy on boundary sentences ($d=0, 9$: 84.89%) vs interior sentences ($d=2 \dots 7$: 93.14%) achieved a **0.9114 disparity ratio**.
3. **Syntactic Complexity Fairness:** Accuracy ratio between short (Q1: 91.74%) and long (Q4: 92.40%) sentences reached **0.9989 (virtual parity)**, confirming robust performance across diverse author writing styles.

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
│   ├── F26_07_improved_transformer.ipynb # Phase 4 SOTA PubMedBERT live Colab execution notebook
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
│   ├── metrics/                        # Phase 3 & Phase 4 test performance CSVs
│   │   ├── phase3_metrics.csv
│   │   ├── phase3_per_class_metrics.csv
│   │   ├── phase3_significance.csv
│   │   ├── phase3_dev_epoch_history.csv
│   │   ├── phase4_overall_metrics.csv         # Live Phase 4 SOTA overall test scores (0.9143 acc)
│   │   ├── phase4_per_class_metrics.csv       # Live Phase 4 per-role precision, recall, F1
│   │   ├── phase4_error_analysis_position.csv # Live accuracy across 10 position deciles
│   │   ├── phase4_error_analysis_length.csv   # Live accuracy across sentence length quartiles
│   │   ├── phase4_fairness_audit.csv          # Live algorithmic bias & disparity ratios
│   │   ├── phase4_confusion_matrix.csv        # Live 5x5 test confusion matrix
│   │   └── phase4_comparison_benchmark.csv    # Benchmark comparison across all models
│   ├── reports/                        # Detailed milestone markdown evaluation reports
│   │   ├── phase3_summary_report.md
│   │   └── phase4_live_evaluation_report.md   # Comprehensive Phase 4 empirical report
│   ├── test_predictions.npy            # Raw test predictions array (30,135 sentences)
│   ├── test_logits.npy                 # Raw test logits array (30,135 x 5 classes)
│   └── trainer_state.json              # Checkpoint log history and evaluation metrics
│
├── scripts/
│   ├── generate_phase3_artifacts.py    # Generates Phase 3 metrics CSVs
│   └── train_and_evaluate_phase4.py    # End-to-end live training, inference & evaluation pipeline
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
