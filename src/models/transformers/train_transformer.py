from __future__ import annotations
"""
Transformer fine-tuning pipeline for PubMed 20k RCT sentence classification.
Fine-tunes emilyalsentzer/Bio_ClinicalBERT with HuggingFace Trainer.
"""

import math
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)


class SentenceDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for sentence tokenization with on-the-fly or pre-tokenized features."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        label2id: Dict[str, int],
        max_length: int = 128,
    ):
        self.texts = df["text"].tolist()
        self.labels = [label2id[l] for l in df["label"]] if "label" in df else None
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            max_length=max_length,
            padding=False,  # dynamic padding via DataCollator
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


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute accuracy and macro-F1 for HuggingFace Trainer evaluations."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def build_transformer_trainer(
    model_name: str,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    labels: List[str],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    output_dir: str = "models/transformers/bio_clinicalbert_ckpt",
    epochs: int = 5,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    max_length: int = 128,
    seed: int = 42,
) -> Tuple[Trainer, AutoTokenizer, Any]:
    """
    Initialises tokenizer, pre-trained model, datasets, data collator,
    and HuggingFace Trainer with specified hyper-parameters.
    """
    has_gpu = torch.cuda.is_available()
    print(f"[*] Initialising {model_name} (GPU available: {has_gpu})...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    train_ds = SentenceDataset(train_df, tokenizer, label2id, max_length=max_length)
    dev_ds = SentenceDataset(dev_df, tokenizer, label2id, max_length=max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    steps_per_epoch = math.ceil(len(train_ds) / batch_size)
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

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    return trainer, tokenizer, model
