from __future__ import annotations
"""
Transformer fine-tuning pipeline for PubMed 20k RCT sentence classification.
Supports standard fine-tuning (Bio_ClinicalBERT) as well as enhanced modes:
- Position-aware structural prompt injection
- Domain-specific PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)
- Class-weighted CrossEntropy loss for minority class balance
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)


class SentenceDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for sentence tokenization with optional position-prompt conditioning."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        label2id: Dict[str, int],
        max_length: int = 128,
        use_position_prompts: bool = False,
    ):
        if use_position_prompts and "line_number" in df and "total_lines" in df:
            self.texts = [
                f"[LINE {int(r['line_number'])} OF {int(r['total_lines'])}] "
                f"[PCT {int(round(float(r.get('relative_position', 0.0)) * 100))}%] {r['text']}"
                for _, r in df.iterrows()
            ]
        else:
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


class ClassWeightedTrainer(Trainer):
    """Custom Trainer implementing class-weighted CrossEntropy loss."""

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


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute accuracy, weighted-F1, macro-F1, and kappa for HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(labels, preds)),
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
    use_position_prompts: bool = False,
    use_class_weights: bool = False,
    seed: int = 42,
) -> Tuple[Trainer, AutoTokenizer, Any]:
    """
    Initialises tokenizer, pre-trained model, datasets, data collator,
    and HuggingFace Trainer with specified hyper-parameters.
    """
    has_gpu = torch.cuda.is_available()
    print(f"[*] Initialising {model_name} (GPU available: {has_gpu})...")
    print(f"    - Position Prompts: {use_position_prompts}")
    print(f"    - Class Weights: {use_class_weights}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    train_ds = SentenceDataset(
        train_df,
        tokenizer,
        label2id,
        max_length=max_length,
        use_position_prompts=use_position_prompts,
    )
    dev_ds = SentenceDataset(
        dev_df,
        tokenizer,
        label2id,
        max_length=max_length,
        use_position_prompts=use_position_prompts,
    )
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

    class_weights_tensor = None
    if use_class_weights:
        y_train = np.array([label2id[l] for l in train_df["label"]])
        weights = compute_class_weight("balanced", classes=np.arange(len(labels)), y=y_train)
        class_weights_tensor = torch.tensor(weights, dtype=torch.float)
        print(f"    - Computed class weights: {class_weights_tensor.tolist()}")

    trainer = ClassWeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
    )

    return trainer, tokenizer, model
