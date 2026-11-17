# Phase 4 Live Evaluation Report: Fine-Tuned Biomedical Transformers (PubMedBERT + Position + Weighted Loss)

**Project Code:** F26-07 | Track A (Research & Development) | AI2002 Artificial Intelligence  
**Course Instructor:** Hajra Waheed  
**Authors:** Azan Wasty (24L-2557), Nauman Iqbal (24L-2565), Afras Shahnawaz (24L-2547)  
**Institution:** Department of Data Science, National University of Computer and Emerging Sciences  
**Dataset:** PubMed 20k RCT (Held-Out Test Set: 30,135 sentences across 2,500 abstracts)  
**Evaluation Date:** November 2026 (Live Colab Execution on Tesla T4 GPU)  

---

## 1. Executive Summary & Verified Live Results

In Phase 4, we addressed the structural feature gap identified in Phase 3—where classical Logistic Regression augmented with position features (`LogReg + Position`, Macro F1 = 0.8295) outperformed the text-only transformer baseline (`Bio_ClinicalBERT`, Macro F1 = 0.8143). 

We implemented and executed an improved biomedical transformer pipeline combining:
1. **In-Domain Pretraining:** [`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract) pre-trained from scratch on 14 million PubMed abstracts.
2. **Structural Position Conditioning:** Prefixing sentences with abstract location tokens: `[LINE {line_number} OF {total_lines}] [PCT {pct}%] {text}`.
3. **Class-Weighted Cross-Entropy Loss:** Inverse class frequency weights ($w_j = N / (K \cdot n_j)$) addressing minority class imbalance.

### Live Verified Performance Highlights
- **Test Accuracy:** **0.9143 (91.43%)** (Correct: 27,553 / 30,135 sentences)
- **Macro F1:** **0.8705** (Dev Macro F1: **0.8726**)
- **Weighted F1:** **0.9140**
- **Cohen's Kappa ($\kappa$):** **0.8850**
- **Total Test Misclassifications:** **2,582 errors** (compared to 3,923 in baseline Bio_ClinicalBERT $\rightarrow$ **1,341 fewer errors, a 34.2% error reduction**).
- **Training Wall-Clock Time:** **4,998.9 seconds (~83.3 minutes)** on an NVIDIA Tesla T4 GPU.
- **Inference Latency:** **1.723 seconds per 1,000 sentences** on GPU.

---

## 2. Multi-Model Benchmark Comparison

All metrics evaluated on the identical held-out **PubMed 20k RCT test set (30,135 sentences)**:

| Model Architecture | Input Features & Modality | Test Accuracy | Weighted F1 | Macro F1 | Cohen's Kappa | Dev Macro F1 | Train Time (s) | Inference Latency (per 1k sents) | Hardware |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | TF-IDF (20k n-grams) | 0.7722 | 0.7726 | 0.7135 | 0.6946 | 0.7230 | 159.8 s | 0.067 s | CPU |
| **XGBoost** | TF-IDF (20k n-grams) | 0.7862 | 0.7891 | 0.7305 | 0.7149 | 0.7392 | 78.6 s | **0.045 s** | CPU |
| **Logistic Regression (SGD)** | TF-IDF (200k n-grams) | 0.8236 | 0.8232 | 0.7630 | 0.7630 | 0.7733 | **9.3 s** | 0.073 s | CPU |
| **LogReg + Position (Extension)** | TF-IDF (200k) + Structural (14-d) | 0.8792 | 0.8788 | 0.8295 | 0.8378 | 0.8329 | 11.7 s | 0.073 s | CPU |
| **Bio_ClinicalBERT [text] (Phase 3)** | Raw Text Only | 0.8698 | 0.8689 | 0.8143 | 0.8247 | 0.8253 | 4,999.7 s | 1.656 s | GPU (Tesla T4) |
| **PubMedBERT + Pos + Weighted (Phase 4 Live)** | Text + Position Conditioning + Class Weights | **0.9143** | **0.9140** | **0.8705** | **0.8850** | **0.8726** | 4,998.9 s | 1.723 s | GPU (Tesla T4) |

### Key Observations
1. **Restoring Transformer Dominance:** Phase 4 PubMedBERT decisively surpasses `LogReg + Position` (**+3.51% Accuracy**, **+0.0410 Macro F1**), proving that deep attention models combined with structural cues unlock the highest discriminative power.
2. **Substantial Gain Over Baseline Transformer:** Compared to Phase 3 text-only `Bio_ClinicalBERT`, the improved model delivers a **+4.45% increase in Accuracy** and a **+0.0562 (+6.9%) boost in Macro F1**.

---

## 3. Live Per-Class Performance Breakdown

Evaluated on the 30,135 held-out test sentences:

| Rhetorical Role | Test Support | Precision | Recall | Phase 4 F1 | Phase 3 BERT F1 | Phase 3 LogReg F1 | $\Delta$ F1 (Phase 4 vs Phase 3 BERT) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `BACKGROUND` | 3,621 | 0.7829 | 0.8238 | **0.8029** | 0.7260 | 0.6550 | **+0.0769 (+10.6%)** |
| `OBJECTIVE` | 2,333 | 0.7238 | 0.6695 | **0.6956** | 0.6540 | 0.6140 | **+0.0416 (+6.4%)** |
| `METHODS` | 9,897 | 0.9435 | 0.9700 | **0.9566** | 0.9380 | 0.9040 | **+0.0186 (+2.0%)** |
| `RESULTS` | 9,713 | 0.9681 | 0.9277 | **0.9475** | 0.9200 | 0.8850 | **+0.0275 (+3.0%)** |
| `CONCLUSIONS` | 4,571 | 0.9387 | 0.9619 | **0.9502** | 0.8340 | 0.7570 | **+0.1162 (+13.9%)** |

### Per-Class Insights
- **`CONCLUSIONS` Breakthrough (+11.6% F1):** F1 surged from 0.8340 to **0.9502**. With position conditioning, concluding sentences are recognized almost perfectly (4,397 out of 4,571 correct; 96.19% recall).
- **`BACKGROUND` Robustness (+7.7% F1):** F1 rose to **0.8029** with 82.38% recall, effectively anchoring early abstract context.
- **`OBJECTIVE` Imbalance Relief (+4.2% F1):** Class weighting lifted `OBJECTIVE` recall from 0.6140 to **0.6695** and precision to **0.7238**, preventing minority class suppression.

---

## 4. Live Confusion Matrix & Primary Error Modes

### 5x5 Confusion Matrix (Rows = Ground Truth, Columns = Predicted Role)

| True \ Pred | `BACKGROUND` | `OBJECTIVE` | `METHODS` | `RESULTS` | `CONCLUSIONS` | Total True |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`BACKGROUND`** | **2,983** | 543 | 76 | 8 | 11 | 3,621 |
| **`OBJECTIVE`** | 709 | **1,562** | 56 | 6 | 0 | 2,333 |
| **`METHODS`** | 77 | 51 | **9,600** | 160 | 9 | 9,897 |
| **`RESULTS`** | 5 | 2 | 428 | **9,011** | 267 | 9,713 |
| **`CONCLUSIONS`** | 36 | 0 | 15 | 123 | **4,397** | 4,571 |
| **Total Predicted** | 3,810 | 2,158 | 10,175 | 9,308 | 4,684 | 30,135 |

### Top Failure Modes
1. `OBJECTIVE` $\rightarrow$ `BACKGROUND` (709 sentences): Clinical aims phrased as declarative statements (e.g. *"Cardiovascular disease remains the leading cause of mortality in diabetes"*).
2. `BACKGROUND` $\rightarrow$ `OBJECTIVE` (543 sentences): Background sentences containing intention verbs (*"This trial investigated..."*).
3. `RESULTS` $\rightarrow$ `METHODS` (428 sentences): Outcomes describing patient recruitment totals or drop-out counts that mirror methodology descriptions.
4. `RESULTS` $\rightarrow$ `CONCLUSIONS` (267 sentences): Result sentences summarizing clinical takeaway values.
5. `METHODS` $\rightarrow$ `RESULTS` (160 sentences): Procedural sentences citing numerical dosages or sample sizes.

---

## 5. Live Positional & Syntactic Error Analysis

### 5.1 Position Decile Performance ($d = 0 \dots 9$)

| Position Decile | Relative Abstract Range | Total Sentences | Accuracy | Error Rate | Diagnostic Observation |
|:---:|:---:|:---:|:---:|:---:|:---|
| **0** | 0–10% | 4,205 | **0.7850** | 21.50% | Opening sentences where Background and Objective semantics overlap. |
| **1** | 10–20% | 2,827 | **0.8454** | 15.46% | Transition from rationale to study objective/design. |
| **2** | 20–30% | 3,058 | **0.9035** | 9.65% | Core trial methodology. |
| **3** | 30–40% | 2,969 | **0.9326** | 6.74% | Intervention protocols and sample sizes. |
| **4** | 40–50% | 2,639 | **0.9583** | 4.17% | Trial measurements. |
| **5** | 50–60% | 3,378 | **0.9713** | 2.87% | Primary outcome statistics and $p$-values. |
| **6** | 60–70% | 3,140 | **0.9615** | 3.85% | Secondary outcome comparisons. |
| **7** | 70–80% | 2,887 | **0.9318** | 6.82% | Adverse events and safety profiles. |
| **8** | 80–90% | 2,998 | **0.9393** | 6.07% | Synthesis of primary findings. |
| **9** | 90–100% | 2,034 | **0.9808** | **1.92%** | Concluding recommendations (near-perfect 98.1% accuracy). |

### 5.2 Sentence Length Quartile Performance

| Length Quartile | Total Sentences | Accuracy | Error Rate | Assessment |
|:---|:---:|:---:|:---:|:---|
| **Q1 (Shortest: $\le 86$ chars)** | 7,655 | **0.9174** | 8.26% | Highly robust despite sparse lexical context. |
| **Q2 (Med-Short: 87–134 chars)** | 7,427 | **0.9101** | 8.99% | Consistent across standard descriptive lengths. |
| **Q3 (Med-Long: 135–196 chars)** | 7,540 | **0.9057** | 9.43% | Handles compound medical clauses effectively. |
| **Q4 (Longest: $\ge 197$ chars)** | 7,513 | **0.9240** | 7.60% | Highest accuracy due to rich syntactic & lexical evidence. |

---

## 6. Live Algorithmic Bias & Fairness Audit

| Fairness Dimension | Subgroup A | Subgroup B | Metric Subgroup A | Metric Subgroup B | Disparity Ratio | Assessment |
|:---|:---|:---|:---:|:---:|:---:|:---|
| **Class Imbalance Parity** | Minority (`OBJECTIVE` - 7.7%) | Majority (`METHODS` - 33.0%) | Recall: 0.6695 | Recall: 0.9700 | **0.6902** | Significant improvement from baseline (0.646), mitigating minority starvation. |
| **Discourse Boundary Equity** | Boundary ($d=0, 9$) | Interior ($d=2 \dots 7$) | Accuracy: 0.8489 | Accuracy: 0.9314 | **0.9114** | Near-parity across abstract discourse locations. |
| **Sentence Length Equity** | Short Sentences (Q1) | Long Sentences (Q4) | Accuracy: 0.9174 | Accuracy: 0.9240 | **0.9989** | Virtual parity across varying syntactic sentence lengths. |

---

## 7. Clinical RAG Integration (Hallucination Control)

In clinical Evidence-Synthesis RAG systems, this classifier functions as the **upstream semantic filter**:
1. **Semantic Chunking:** Abstracts are partitioned into rhetorical functional units rather than arbitrary token lengths.
2. **Exclusion of Speculative Context:** `BACKGROUND` sentences are filtered out prior to vector indexing, preventing generative LLMs from mistaking clinical rationale or prior unverified studies for empirical trial outcomes.
3. **Evidence Grounding:** Retrieved evidence is strictly constrained to chunks tagged as `RESULTS` and `CONCLUSIONS`, guaranteeing that downstream answer synthesis is grounded solely in verified clinical findings.

