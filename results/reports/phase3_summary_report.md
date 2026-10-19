# Phase 3 Summary Report: Sentence-Role Classification in Medical Abstracts

**Project Code:** F26-07 | Track A (Research & Development) | AI2002 Artificial Intelligence  
**Instructor:** Hajra Waheed  
**Authors:** Azan Wasty (24L-2557), Nauman Iqbal (24L-2565), Afras Shahnawaz (24L-2547)  
**Institution:** Department of Data Science, National University of Computer and Emerging Sciences  

---

## 1. Executive Summary

This study benchmarks **classical machine learning pipelines** (TF-IDF + Logistic Regression / Random Forest / XGBoost) against a **fine-tuned biomedical transformer** (`emilyalsentzer/Bio_ClinicalBERT`) on the 5-class sentence-role classification task using the **PubMed 20k RCT** benchmark dataset (Dernoncourt & Lee, 2017).

Each sentence from a published randomised controlled trial abstract is classified into one of five rhetorical roles:
- `BACKGROUND`: Context and motivation
- `OBJECTIVE`: Hypothesis or specific aim
- `METHODS`: Experimental design, participants, intervention
- `RESULTS`: Findings, measurements, statistical outcomes
- `CONCLUSIONS`: Interpretations and take-home messages

---

## 2. Dataset Overview

- **Source:** PubMed 20k RCT (official splits, grouped by abstract to prevent data leakage)
- **Training Set:** 180,040 sentences (~15,000 abstracts)
- **Validation (Dev) Set:** 30,212 sentences (~2,500 abstracts)
- **Test Set:** 30,135 sentences (~2,500 abstracts)
- **Class Distribution (Training):**
  - `METHODS`: 33.0% (59,353 sentences)
  - `RESULTS`: 32.2% (57,953 sentences)
  - `CONCLUSIONS`: 15.1% (27,168 sentences)
  - `BACKGROUND`: 12.1% (21,727 sentences)
  - `OBJECTIVE`: 7.7% (13,839 sentences)
- **Primary Metric:** **Macro F1** (due to class imbalance; `OBJECTIVE` is the minority class at <8%).

---

## 3. Experimental Setup

### Classical ML Family (TF-IDF + Classifiers)
- **Feature Extraction:** Word 1–2 grams, lowercased, minimum document frequency $df \ge 3$, sublinear term frequency scaling ($1 + \log(\text{tf})$).
  - *200,000 features* for Logistic Regression.
  - *20,000 features* for tree ensembles (Random Forest and XGBoost) to maintain computational efficiency on sparse features.
- **Models:**
  - **Logistic Regression:** Trained via SGD with `log_loss`, $L_2$ regularisation parameter $\alpha = 2 \times 10^{-6}$, 30 iterations, balanced class weighting.
  - **Random Forest:** 100 trees, $\sqrt{p}$ features per split, minimum leaf size 3, balanced subsample weighting.
  - **XGBoost:** 300 trees, maximum depth 6, learning rate 0.2, subsample ratio 0.8, column sample 0.5.

### Transformer Family
- **Model:** `emilyalsentzer/Bio_ClinicalBERT` (108.3M parameters, BERT base pretrained on PubMed abstracts and MIMIC-III clinical notes).
- **Hyperparameters:** Max token length 128 (dynamic padding), per-device batch size 32, 5 training epochs, learning rate $2 \times 10^{-5}$ with linear decay, 6% warmup (1,688 / 28,135 steps), weight decay 0.01, mixed precision (FP16) on Tesla T4 GPU.
- **Model Selection:** Best checkpoint chosen based on validation (dev) Macro F1 (Epoch 3: Macro F1 = 0.8253).

---

## 4. Empirical Results (Test Set: 30,135 Sentences)

### 4.1 Overall Model Benchmark

| Model | Accuracy | Weighted F1 | Macro F1 | Cohen's Kappa | Dev Macro F1 | Train Time (s) | Inference Latency (s / 1k sent.) | Hardware |
|---|---|---|---|---|---|---|---|---|
| **Bio_ClinicalBERT [text]** | **0.8698** | **0.8689** | **0.8143** | **0.8247** | 0.8253 | 4,999.7 s | 1.656 s | GPU (T4) |
| **LogReg (SGD) [text]** | 0.8236 | 0.8232 | 0.7630 | 0.7630 | 0.7733 | **9.3 s** | 0.073 s | CPU |
| **XGBoost [text]** | 0.7862 | 0.7891 | 0.7305 | 0.7149 | 0.7392 | 78.6 s | **0.045 s** | CPU |
| **Random Forest [text]** | 0.7722 | 0.7726 | 0.7135 | 0.6946 | 0.7230 | 159.8 s | 0.067 s | CPU |
| *LogReg + position (extension)* | *0.8792* | *0.8788* | *0.8295* | *0.8378* | *0.8329* | *11.7 s* | *0.073 s* | *CPU* |

### 4.2 Per-Class Performance Breakdown

| Rhetorical Role | LogReg F1 | Bio_ClinicalBERT Precision | Bio_ClinicalBERT Recall | Bio_ClinicalBERT F1 | Δ F1 (BERT vs LogReg) | Test Support |
|---|---|---|---|---|---|---|
| `BACKGROUND` | 0.655 | 0.701 | 0.752 | **0.726** | **+0.071** | 3,621 |
| `OBJECTIVE` | 0.614 | 0.699 | 0.614 | **0.654** | **+0.040** | 2,333 |
| `METHODS` | 0.904 | 0.927 | 0.950 | **0.938** | **+0.034** | 9,897 |
| `RESULTS` | 0.885 | 0.917 | 0.923 | **0.920** | **+0.035** | 9,713 |
| `CONCLUSIONS` | 0.757 | 0.863 | 0.807 | **0.834** | **+0.077** | 4,571 |

---

## 5. Answers to Research Questions

### RQ1: Classical Model Accuracy
- The best classical baseline on text features alone is **SGD Logistic Regression** (Macro F1 = 0.7630, Accuracy = 0.8236).
- Tree models struggled on the high-dimensional sparse TF-IDF space: XGBoost achieved Macro F1 = 0.7305, and Random Forest achieved Macro F1 = 0.7135.
- Top informative logistic regression coefficients align closely with clinical discourse:
  - `RESULTS`: *"respectively"*, *"showed"*, *"significantly"*, *"p <"*, *"ci"*
  - `OBJECTIVE`: *"whether"*, *"evaluate"*, *"investigate"*, *"examine"*, *"aim"*
  - `CONCLUSIONS`: *"should"*, *"further"*, *"future"*, *"suggest"*, *"may"*

### RQ2: Statistical Significance (Bio_ClinicalBERT vs Classical Baselines)
- **Significance Testing Framework:**
  1. *McNemar's Exact Test* on discordance pairs.
  2. *Paired Abstract-Level Bootstrap* (1,000 resamples of whole abstracts to account for intra-cluster correlation).
  3. Bonferroni-corrected significance threshold $\alpha = 0.05 / 3 = 0.0167$.

| Baseline Model | Macro F1 Diff (BERT − Baseline) | 95% Bootstrap CI | Only Baseline Correct | Only BERT Correct | McNemar p-value | Statistically Significant? |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **+0.0512** | **[+0.0456, +0.0570]** | 1,046 | 2,439 | $2.3 \times 10^{-126}$ | **Yes** ($p < \alpha_{\text{Bonf}}$) |
| **XGBoost** | **+0.0838** | **[+0.0775, +0.0898]** | 1,092 | 3,611 | $5.0 \times 10^{-311}$ | **Yes** ($p < \alpha_{\text{Bonf}}$) |
| **Random Forest** | **+0.1008** | **[+0.0943, +0.1070]** | 1,078 | 4,020 | $< 10^{-300}$ | **Yes** ($p < \alpha_{\text{Bonf}}$) |

*Conclusion:* The advantage of Bio_ClinicalBERT over all text-only classical models is statistically significant, and every 95% confidence interval is strictly positive.

### RQ3: Accuracy vs. Efficiency Trade-Off
- **Training Cost:** Bio_ClinicalBERT required **4,999.7 seconds (~83 minutes)** on an NVIDIA Tesla T4 GPU, compared to **9.3 seconds** for Logistic Regression on CPU (~540× faster).
- **Inference Latency:** Bio_ClinicalBERT evaluated 1,000 sentences in **1.656 seconds** on GPU, whereas Logistic Regression processed 1,000 sentences in **0.073 seconds** on CPU (~23× faster).
- **Parameter Footprint:** Bio_ClinicalBERT has **108.3 million parameters**, while Logistic Regression requires minimal disk and RAM footprint (<15 MB).
- **Practical Takeaway:** Logistic Regression achieves **~94% of Bio_ClinicalBERT's Macro F1** at a tiny fraction of compute cost. For resource-constrained or real-time clinical workflows, SGD Logistic Regression is a strong candidate.

---

## 6. Structural Position Extension

When 10-bin relative position deciles, sentence boundary indicators, numeric flags, and log sentence lengths are concatenated with 200k TF-IDF features, Logistic Regression achieves:
- **Macro F1:** **0.8295** (+0.0665 over text-only LogReg, outperforming text-only Bio_ClinicalBERT at 0.8143)
- **Accuracy:** **0.8792**
- **Training Time:** **11.7 seconds**

*Insight:* Much of the gap between text-only classical ML and text-only transformers is a **feature gap** (lack of structural context), rather than an inherent representation ceiling. Concluding sentences often read identically to results, and position is the primary distinguishing signal.

---

## 7. Error Analysis & Limitations

- Bio_ClinicalBERT misclassified 3,923 out of 30,135 sentences (13.0%).
- Most frequent misclassifications:
  1. `OBJECTIVE` predicted as `BACKGROUND` (732 sentences)
  2. `BACKGROUND` predicted as `OBJECTIVE` (528 sentences)
  3. `CONCLUSIONS` predicted as `RESULTS` (493 sentences)
  4. `RESULTS` predicted as `METHODS` (442 sentences)
- `OBJECTIVE` is the most difficult class for both model families (31% of true `OBJECTIVE` sentences were predicted as `BACKGROUND`).
- Real-world label noise in PubMed RCT sets an empirical ceiling (e.g., trial registry identifiers like NCT00139126 occasionally labelled as `BACKGROUND`, study aims labelled as `BACKGROUND`).

---

## 8. Next Steps (Phase 4)

1. Multi-seed training to assess random initialization variance.
2. Comprehensive error and bias analysis across sentence length quartiles and clinical specialty categories.
3. Extension of Bio_ClinicalBERT to accept sequence-level positional and hierarchical context for an apples-to-apples comparison with the structural feature extension.
4. Optional comparison against general-domain BERT (`bert-base-uncased`) and `allenai/scibert_scivocab_uncased`.
