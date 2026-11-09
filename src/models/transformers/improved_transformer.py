from __future__ import annotations
"""
Improved Transformer Fine-Tuning Pipeline for PubMed 20k RCT Sentence Classification.

Key Innovations:
1. Positional & Structural Conditioning (Position-Aware Prompt Injection):
   Prefixes each sentence with abstract discourse position tokens:
   "[LINE {i} OF {n}] [PCT {p}%] {text}"
   This resolves the fundamental "feature gap" where text-only transformers missed
   the sequential discourse progression of medical abstracts.

2. Domain-Specific Pretrained Backbone (PubMedBERT):
   Supports microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract (pre-trained from
   scratch on 14M PubMed abstracts), directly overcoming domain-mismatch of clinical EHR models.

3. Class-Weighted Loss Optimization:
   Addresses severe class imbalance (OBJECTIVE is only 7.7% of the dataset) using
   inverse class frequencies to prevent recall collapse.

4. Hybrid Multimodal Feature-Fusion Head (Alternative Architecture):
   Fuses transformer pooled embeddings (768-d) with structural tabular features
   (position deciles, boundary flags, log length) in an end-to-end PyTorch module.
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from src.features.feature_engineering import extract_position_features


def format_position_prompt(
    text: str,
    line_number: int,
    total_lines: int,
    relative_position: float,
    prompt_style: str = "structured",
) -> str:
    """
    Injects structural position metadata into the sentence text.

    Styles:
    - 'structured': '[LINE {i} OF {n}] [PCT {pct}%] {text}'
    - 'natural': 'Sentence {i} of {n} (at {pct}% of abstract): {text}'
    - 'bracketed': '[POS={pct}%] {text}'
    """
    pct = int(round(relative_position * 100))
    if prompt_style == "structured":
        return f"[LINE {line_number} OF {total_lines}] [PCT {pct}%] {text}"
    elif prompt_style == "natural":
        return f"Sentence {line_number + 1} of {total_lines} in abstract: {text}"
    elif prompt_style == "bracketed":
        return f"[POS={pct}%] {text}"
    else:
        return text


def build_position_conditioned_texts(
    df: pd.DataFrame,
    prompt_style: str = "structured",
) -> List[str]:
    """
    Transforms DataFrame sentences into position-conditioned text sequences.
    """
    formatted = []
    for _, row in df.iterrows():
        line_num = int(row.get("line_number", 0))
        tot_lines = int(row.get("total_lines", 1))
        rel_pos = float(row.get("relative_position", 0.0))
        txt = str(row["text"])
        formatted.append(
            format_position_prompt(txt, line_num, tot_lines, rel_pos, prompt_style=prompt_style)
        )
    return formatted


class PositionConditionedDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset providing dynamically tokenized position-aware sequences.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        label2id: Optional[Dict[str, int]] = None,
        max_length: int = 128,
        prompt_style: str = "structured",
        use_position_prompts: bool = True,
    ):
        if use_position_prompts:
            self.texts = build_position_conditioned_texts(df, prompt_style=prompt_style)
        else:
            self.texts = df["text"].tolist()

        if label2id is not None and "label" in df:
            self.labels = [label2id[l] for l in df["label"]]
        else:
            self.labels = None

        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class ClassWeightedTrainer(Trainer):
    """
    Custom HuggingFace Trainer with weighted cross-entropy loss to handle
    minority class starvation (e.g. OBJECTIVE at 7.7%).
    """

    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class HybridTransformerClassifier(nn.Module):
    """
    End-to-end multimodal architecture fusing dense Transformer contextual
    embeddings (768-d) with structural discourse vectors (14-d).
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 5,
        structural_dim: int = 14,
        fusion_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        encoder_dim = self.encoder.config.hidden_size

        self.struct_proj = nn.Sequential(
            nn.Linear(structural_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        total_dim = encoder_dim + fusion_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        structural_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use mean pooling or [CLS] token
        cls_rep = outputs.last_hidden_state[:, 0, :]

        struct_rep = self.struct_proj(structural_features)
        fused = torch.cat([cls_rep, struct_rep], dim=-1)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            if class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}


def compute_inverse_class_weights(labels_series: pd.Series, label2id: Dict[str, int]) -> torch.Tensor:
    """
    Computes balanced class weights inversely proportional to class frequencies:
    w_j = N / (K * n_j)
    """
    y = np.array([label2id[l] for l in labels_series])
    classes = np.arange(len(label2id))
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float)


def compute_metrics_fn(eval_pred) -> Dict[str, float]:
    """Computes comprehensive evaluation metrics during Trainer evaluation."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = float(accuracy_score(labels, preds))
    w_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    m_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    kappa = float(cohen_kappa_score(labels, preds))
    return {
        "accuracy": acc,
        "weighted_f1": w_f1,
        "macro_f1": m_f1,
        "cohen_kappa": kappa,
    }


def build_improved_transformer_trainer(
    model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    train_df: Optional[pd.DataFrame] = None,
    dev_df: Optional[pd.DataFrame] = None,
    labels: Optional[List[str]] = None,
    label2id: Optional[Dict[str, int]] = None,
    id2label: Optional[Dict[int, str]] = None,
    output_dir: str = "models/transformers/improved_pubmedbert_ckpt",
    epochs: int = 5,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    max_length: int = 128,
    use_position_prompts: bool = True,
    use_class_weights: bool = True,
    seed: int = 42,
) -> Tuple[Trainer, AutoTokenizer, Any]:
    """
    Initialises and returns an improved HuggingFace Trainer with:
    - Domain-adapted PubMedBERT or Bio_ClinicalBERT
    - Position-aware structural prompting
    - Class-weighted CrossEntropy loss
    - Mixed precision and dynamic padding
    """
    if labels is None:
        labels = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
    if label2id is None:
        label2id = {l: i for i, l in enumerate(labels)}
    if id2label is None:
        id2label = {i: l for l, i in label2id.items()}

    has_gpu = torch.cuda.is_available()
    print(f"[*] Initialising Improved Transformer: {model_name}")
    print(f"    - GPU Available: {has_gpu}")
    print(f"    - Position Prompts: {use_position_prompts}")
    print(f"    - Class Weighting: {use_class_weights}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    train_ds = PositionConditionedDataset(
        train_df,
        tokenizer,
        label2id=label2id,
        max_length=max_length,
        use_position_prompts=use_position_prompts,
    ) if train_df is not None else None

    dev_ds = PositionConditionedDataset(
        dev_df,
        tokenizer,
        label2id=label2id,
        max_length=max_length,
        use_position_prompts=use_position_prompts,
    ) if dev_df is not None else None

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    steps_per_epoch = math.ceil(len(train_ds) / batch_size) if train_ds is not None else 1000
    warmup_steps = int(warmup_ratio * steps_per_epoch * epochs)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=has_gpu,
        seed=seed,
        logging_steps=200,
        report_to="none",
    )

    class_weights_tensor = None
    if use_class_weights and train_df is not None:
        class_weights_tensor = compute_inverse_class_weights(train_df["label"], label2id)
        print(f"    - Class weights: {class_weights_tensor.tolist()}")

    trainer = ClassWeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
        class_weights=class_weights_tensor,
    )

    return trainer, tokenizer, model

