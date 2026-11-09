const fs = require('fs');

const notebook = {
  cells: [
    {
      cell_type: "markdown",
      id: "header_1",
      metadata: {},
      source: [
        "# F26-07 | Improved Biomedical Transformers for Medical Abstract Sentence-Role Classification\n",
        "## Domain Pre-Training (PubMedBERT), Structural Position Conditioning, Class-Weighted Loss, and Algorithmic Fairness on PubMed 20k RCT\n",
        "\n",
        "**Group F26-07:** Azan Wasty (24L-2557), Nauman Iqbal (24L-2565), Afras Shahnawaz (24L-2547)  \n",
        "**Instructor:** Hajra Waheed | **Department of Data Science, FAST-NUCES**  \n",
        "\n",
        "---\n",
        "\n",
        "### 💡 Key Research Innovations in this Notebook:\n",
        "1. **Bridging the Feature Gap via Position-Aware Prompting:**\n",
        "   In Phase 3, classical `LogReg + Position` (Macro F1 = 0.8295) outperformed text-only `Bio_ClinicalBERT` (0.8143) because text-only models disregard abstract discourse structure. We introduce **position-aware prompt injection** (`[LINE i OF n] [PCT p%] <text>`), enabling the transformer attention heads to jointly reason over structural position and biomedical semantics.\n",
        "2. **Domain-Specific Abstract Pre-Training (PubMedBERT):**\n",
        "   We benchmark `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`, pre-trained from scratch on PubMed abstracts, directly resolving the hospital EHR domain-shift of `Bio_ClinicalBERT`.\n",
        "3. **Class-Weighted Cross-Entropy Loss:**\n",
        "   Mitigates severe class imbalance (the minority class `OBJECTIVE` accounts for only 7.7% of the dataset) to prevent recall starvation.\n",
        "4. **Comprehensive Fairness Auditing & Error Diagnostics:**\n",
        "   Audits minority recall parity, position decile accuracy, sentence complexity bias, and clinical RAG hallucination control."
      ]
    },
    {
      cell_type: "markdown",
      id: "sec_1",
      metadata: {},
      source: ["## 1. Setup & Environment Verification"]
    },
    {
      cell_type: "code",
      execution_count: null,
      id: "code_setup",
      metadata: {},
      outputs: [],
      source: [
        "# Install required dependencies for Colab\n",
        "%pip install -q transformers datasets accelerate xgboost scikit-learn seaborn matplotlib"
      ]
    },
    {
      cell_type: "code",
      execution_count: null,
      id: "code_imports",
      metadata: {},
      outputs: [],
      source: [
        "import os, re, time, math, random\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "from pathlib import Path\n",
        "from typing import Dict, List, Optional, Tuple, Any\n",
        "from scipy.stats import binomtest\n",
        "from sklearn.metrics import (\n",
        "    accuracy_score, f1_score, cohen_kappa_score,\n",
        "    classification_report, confusion_matrix, precision_recall_fscore_support\n",
        ")\n",
        "from sklearn.utils.class_weight import compute_class_weight\n",
        "from transformers import (\n",
        "    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,\n",
        "    TrainingArguments, Trainer, DataCollatorWithPadding\n",
        ")\n",
        "\n",
        "SEED = 42\n",
        "random.seed(SEED)\n",
        "np.random.seed(SEED)\n",
        "torch.manual_seed(SEED)\n",
        "if torch.cuda.is_available():\n",
        "    torch.cuda.manual_seed_all(SEED)\n",
        "sns.set_theme(style=\"whitegrid\")\n",
        "\n",
        "DEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "print(f\"[*] Active Compute Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})\")"
      ]
    },
    {
      cell_type: "markdown",
      id: "sec_2",
      metadata: {},
      source: ["## 2. Dataset Ingestion & Positional Feature Extraction"]
    },
    {
      cell_type: "code",
      execution_count: null,
      id: "code_parse",
      metadata: {},
      outputs: [],
      source: [
        "LABELS = ['BACKGROUND', 'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS']\n",
        "LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}\n",
        "ID2LABEL = {i: lbl for i, lbl in enumerate(LABELS)}\n",
        "\n",
        "def clean_text(text: str) -> str:\n",
        "    return re.sub(r'\\s+', ' ', str(text)).strip()\n",
        "\n",
        "def parse_pubmed_rct(filepath: str) -> pd.DataFrame:\n",
        "    rows = []\n",
        "    current_abstract = []\n",
        "    abstract_id = None\n",
        "    with open(filepath, 'r', encoding='utf-8') as f:\n",
        "        for line in f:\n",
        "            line = line.strip()\n",
        "            if line.startswith('###'):\n",
        "                if current_abstract and abstract_id:\n",
        "                    total = len(current_abstract)\n",
        "                    for i, (lbl, txt) in enumerate(current_abstract):\n",
        "                        c_txt = clean_text(txt)\n",
        "                        rows.append({\n",
        "                            'abstract_id': abstract_id,\n",
        "                            'label': lbl,\n",
        "                            'text': c_txt,\n",
        "                            'line_number': i,\n",
        "                            'total_lines': total,\n",
        "                            'line_length': len(c_txt),\n",
        "                            'relative_position': i / total if total > 0 else 0.0\n",
        "                        })\n",
        "                abstract_id = line[3:].strip()\n",
        "                current_abstract = []\n",
        "            elif line == '':\n",
        "                if current_abstract and abstract_id:\n",
        "                    total = len(current_abstract)\n",
        "                    for i, (lbl, txt) in enumerate(current_abstract):\n",
        "                        c_txt = clean_text(txt)\n",
        "                        rows.append({\n",
        "                            'abstract_id': abstract_id,\n",
        "                            'label': lbl,\n",
        "                            'text': c_txt,\n",
        "                            'line_number': i,\n",
        "                            'total_lines': total,\n",
        "                            'line_length': len(c_txt),\n",
        "                            'relative_position': i / total if total > 0 else 0.0\n",
        "                        })\n",
        "                current_abstract = []\n",
        "            else:\n",
        "                parts = line.split('\\t', 1)\n",
        "                if len(parts) == 2:\n",
        "                    current_abstract.append((parts[0].strip(), parts[1].strip()))\n",
        "    return pd.DataFrame(rows)\n",
        "\n",
        "# Load splits (ensure files are present in data/raw/ or update path)\n",
        "raw_dir = Path('data/raw')\n",
        "if not (raw_dir / 'train.txt').exists():\n",
        "    raw_dir = Path('/content/data/raw')\n",
        "\n",
        "train_df = parse_pubmed_rct(raw_dir / 'train.txt')\n",
        "dev_df = parse_pubmed_rct(raw_dir / 'dev.txt')\n",
        "test_df = parse_pubmed_rct(raw_dir / 'test.txt')\n",
        "\n",
        "print(f\"train: {len(train_df):,} sentences | {train_df['abstract_id'].nunique():,} abstracts\")\n",
        "print(f\"dev  : {len(dev_df):,} sentences | {dev_df['abstract_id'].nunique():,} abstracts\")\n",
        "print(f\"test : {len(test_df):,} sentences | {test_df['abstract_id'].nunique():,} abstracts\")"
      ]
    },
    {
      cell_type: "markdown",
      id: "sec_3",
      metadata: {},
      source: [
        "## 3. Position-Aware Prompt Conditioning\n",
        "We inject structural metadata as discrete token prefixes directly into the sentence representation:\n",
        "`[LINE {line_number} OF {total_lines}] [PCT {pct}%] {text}`"
      ]
    },
    {
      cell_type: "code",
      execution_count: null,
      id: "code_prompting",
      metadata: {},
      outputs: [],
      source: [
        "def format_position_prompt(row: pd.Series) -> str:\n",
        "    line_num = int(row['line_number'])\n",
        "    total = int(row['total_lines'])\n",
        "    pct = int(round(float(row['relative_position']) * 100))\n",
        "    return f\"[LINE {line_num} OF {total}] [PCT {pct}%] {row['text']}\"\n",
        "\n",
        "train_df['prompt_text'] = train_df.apply(format_position_prompt, axis=1)\n",
        "dev_df['prompt_text'] = dev_df.apply(format_position_prompt, axis=1)\n",
        "test_df['prompt_text'] = test_df.apply(format_position_prompt, axis=1)\n",
        "\n",
        "print(\"Sample position-conditioned prompt:\")\n",
        "print(train_df['prompt_text'].iloc[0])"
      ]
    },
    {
      cell_type: "markdown",
      id: "sec_4",
      metadata: {},
      source: ["## 4. Class Imbalance Mitigation (Loss Weighting)"]
    },
    {
      cell_type: "code",
      execution_count: null,
      id: "code_weights",
      metadata: {},
      outputs: [],
      source: [
        "y_train = np.array([LABEL2ID[l] for l in train_df['label']])\n",
        "class_weights = compute_class_weight('balanced', classes=np.arange(len(LABELS)), y=y_train)\n",
        "class_weights_t = torch.tensor(class_weights, dtype=torch.float)\n",
        "\n",
        "print(\"Inverse Class Weights:\")\n",
        "for lbl, w in zip(LABELS, class_weights):\n",
        "    pct = (train_df['label'] == lbl).mean() * 100\n",
        "    print(f\"  {lbl:12s}: weight={w:.4f} (training frequency: {pct:.1f}%)\")"
      ]
    },
    {
      cell_type: "markdown",
      id: "sec_5",
      metadata: {},
      source: [
        "## 5. Improved Transformer Training Routine\n",
        "Implements domain-adapted PubMedBERT with position conditioning and weighted cross-entropy."
      ]
    },
    {
      cell_type: "code",
      execution_count: null,
      id: "code_trainer_class",
      metadata: {},
      outputs: [],
      source: [
        "class PositionDataset(torch.utils.data.Dataset):\n",
        "    def __init__(self, texts, labels, tokenizer, max_length=128):\n",
        "        self.labels = labels\n",
        "        self.encodings = tokenizer(texts, truncation=True, max_length=max_length, padding=False)\n",
        "    def __len__(self):\n",
        "        return len(self.encodings['input_ids'])\n",
        "    def __getitem__(self, idx):\n",
        "        item = {k: v[idx] for k, v in self.encodings.items()}\n",
        "        if self.labels is not None:\n",
        "            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)\n",
        "        return item\n",
        "\n",
        "class WeightedTrainer(Trainer):\n",
        "    def __init__(self, *args, class_weights=None, **kwargs):\n",
        "        super().__init__(*args, **kwargs)\n",
        "        self.class_weights = class_weights\n",
        "    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):\n",
        "        labels = inputs.get('labels')\n",
        "        outputs = model(**inputs)\n",
        "        logits = outputs.get('logits')\n",
        "        if self.class_weights is not None:\n",
        "            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))\n",
        "        else:\n",
        "            loss_fn = nn.CrossEntropyLoss()\n",
        "        loss = loss_fn(logits, labels)\n",
        "        return (loss, outputs) if return_outputs else loss\n",
        "\n",
        "def compute_metrics(eval_pred):\n",
        "    logits, labels = eval_pred\n",
        "    preds = np.argmax(logits, axis=-1)\n",
        "    return {\n",
        "        'accuracy': float(accuracy_score(labels, preds)),\n",
        "        'weighted_f1': float(f1_score(labels, preds, average='weighted', zero_division=0)),\n",
        "        'macro_f1': float(f1_score(labels, preds, average='macro', zero_division=0)),\n",
        "        'cohen_kappa': float(cohen_kappa_score(labels, preds))\n",
        "    }"
      ]
    },
    {
      cell_type: "markdown",
      id: "sec_6",
      metadata: {},
      source: [
        "## 6. Comprehensive Multi-Model Benchmarking Results\n",
        "We compare all 7 classical and transformer approaches evaluated on the 30,135 held-out test sentences."
      ]
    },
    {
      cell_type: "code",
      execution_count: null,
      id: "code_benchmark_table",
      metadata: {},
      outputs: [],
      source: [
        "comparison_df = pd.read_csv('results/metrics/comprehensive_model_comparison.csv')\n",
        "display(comparison_df.style.highlight_max(subset=['Test_Accuracy', 'Macro_F1', 'Cohen_Kappa'], color='#d4edda'))"
      ]
    },
    {
      cell_type: "markdown",
      id: "sec_7",
      metadata: {},
      source: [
        "## 7. Statistical Significance Testing\n",
        "We perform McNemar's exact test and cluster bootstrap resampling (1,000 abstract iterations) to confirm that the improved transformer's gains over both classical baselines and baseline Bio_ClinicalBERT are statistically significant."
      ]
    },
    {
      cell_type: "code",
      execution_count: null,
      id: "code_significance",
      metadata: {},
      outputs: [],
      source: [
        "sig_df = pd.read_csv('results/metrics/improved_transformer_significance.csv')\n",
        "display(sig_df)"
      ]
    },
    {
      cell_type: "markdown",
      id: "sec_8",
      metadata: {},
      source: [
        "## 8. Detailed Error Analysis\n",
        "We systematically examine the models' failure modes across abstract position deciles and sentence length quartiles."
      ]
    },
    {
      cell_type: "code",
      execution_count: null,
      id: "code_error_analysis",
      metadata: {},
      outputs: [],
      source: [
        "pos_df = pd.read_csv('results/metrics/error_analysis_position.csv')\n",
        "len_df = pd.read_csv('results/metrics/error_analysis_length.csv')\n",
        "\n",
        "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n",
        "sns.lineplot(data=pos_df, x='Position_Decile', y='Baseline_BERT_Accuracy', marker='o', label='Baseline Bio_ClinicalBERT', ax=axes[0])\n",
        "sns.lineplot(data=pos_df, x='Position_Decile', y='Improved_SOTA_Accuracy', marker='s', label='SOTA PubMedBERT + Pos', ax=axes[0])\n",
        "axes[0].set_title('Accuracy across Sentence Position Deciles (d=0 to 9)')\n",
        "axes[0].set_xlabel('Abstract Position Decile')\n",
        "axes[0].set_ylabel('Accuracy')\n",
        "axes[0].set_ylim(0.75, 0.98)\n",
        "\n",
        "sns.barplot(data=len_df, x='Quartile_Category', y='Accuracy_Gain', palette='Blues_d', ax=axes[1])\n",
        "axes[1].set_title('Accuracy Gain by Sentence Length Quartile')\n",
        "axes[1].set_ylabel('Gain over Baseline')\n",
        "axes[1].tick_params(axis='x', rotation=15)\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ]
    },
    {
      cell_type: "markdown",
      id: "sec_9",
      metadata: {},
      source: [
        "## 9. Algorithmic Bias and Fairness Audit\n",
        "Evaluation of equity across minority classes, syntactic length variations, and structural position boundaries."
      ]
    },
    {
      cell_type: "code",
      execution_count: null,
      id: "code_fairness",
      metadata: {},
      outputs: [],
      source: [
        "fairness_df = pd.read_csv('results/metrics/fairness_audit_metrics.csv')\n",
        "display(fairness_df)"
      ]
    },
    {
      cell_type: "markdown",
      id: "sec_10",
      metadata: {},
      source: [
        "## 10. Architectural Connection to Clinical RAG Pipelines\n",
        "How sentence-role classification serves as the foundational semantic chunker and retriever filter to eliminate hallucinations in evidence-synthesis LLM workflows."
      ]
    },
    {
      cell_type: "code",
      execution_count: null,
      id: "code_rag_demo",
      metadata: {},
      outputs: [],
      source: [
        "def clinical_rag_evidence_filter(abstract_predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:\n",
        "    \"\"\"\n",
        "    Filters retrieved abstract sentences to strictly ground LLM answers\n",
        "    in verified RESULTS and CONCLUSIONS roles, preventing hallucination\n",
        "    stemming from speculative BACKGROUND statements or procedural METHODS.\n",
        "    \"\"\"\n",
        "    ALLOWED_ROLES = {'RESULTS', 'CONCLUSIONS'}\n",
        "    verified_evidence = [s for s in abstract_predictions if s['predicted_role'] in ALLOWED_ROLES]\n",
        "    return verified_evidence\n",
        "\n",
        "print(\"[OK] Clinical RAG Evidence-Filtering logic loaded.\")"
      ]
    }
  ],
  metadata: {
    kernelspec: {
      display_name: "Python 3",
      language: "python",
      name: "python3"
    },
    language_info: {
      codemirror_mode: {
        name: "ipython",
        version: 3
      },
      file_extension: ".py",
      mimetype: "text/x-python",
      name: "python",
      nbformat: 4,
      nbformat_minor: 5
    }
  },
  nbformat: 4,
  nbformat_minor: 5
};

fs.writeFileSync('notebooks/F26-07_improved_transformer.ipynb', JSON.stringify(notebook, null, 2), 'utf8');
console.log('[OK] Created notebooks/F26-07_improved_transformer.ipynb');

