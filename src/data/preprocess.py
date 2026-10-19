"""
Data loading and preprocessing pipeline for PubMed 20k RCT sentence-role classification.
Parses raw abstract files into structured datasets with position metadata.
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    import yaml
except ImportError:
    yaml = None

try:
    import pandas as pd
except ImportError:
    pd = None


LABELS = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
LABEL2ID = {lbl: idx for idx, lbl in enumerate(LABELS)}
ID2LABEL = {idx: lbl for lbl, idx in LABEL2ID.items()}


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flush_abstract(abstract_id: str, current_abstract: List[Tuple[str, str]], rows: List[dict]):
    """Helper to append sentences of an abstract with positional metadata."""
    total_lines = len(current_abstract)
    for i, (label, text) in enumerate(current_abstract):
        clean_s = clean_text(text)
        rel_pos = (i / total_lines) if total_lines > 0 else 0.0
        rows.append({
            "abstract_id": abstract_id,
            "label": label,
            "text": clean_s,
            "line_number": i,
            "total_lines": total_lines,
            "line_length": len(clean_s),
            "relative_position": rel_pos,
        })


def clean_text(text: str) -> str:
    """
    Normalise whitespace while preserving casing, numbers, punctuation,
    and stopwords as per project Phase 2 and Phase 3 methodology.
    """
    return re.sub(r"\s+", " ", str(text)).strip()


def parse_pubmed_rct(filepath: str) -> pd.DataFrame:
    """
    Parses a raw PubMed RCT formatted .txt file into a structured pandas DataFrame.
    Format:
        ###<abstract_id>
        <LABEL>\\t<sentence text>
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found at: {filepath}")

    rows: List[dict] = []
    abstract_id: Optional[str] = None
    current_abstract: List[Tuple[str, str]] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("###"):
                if current_abstract and abstract_id:
                    _flush_abstract(abstract_id, current_abstract, rows)
                abstract_id = line[3:].strip()
                current_abstract = []
            elif line == "":
                if current_abstract and abstract_id:
                    _flush_abstract(abstract_id, current_abstract, rows)
                current_abstract = []
            else:
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    current_abstract.append((parts[0].strip(), parts[1].strip()))

        # Flush last abstract if file has no trailing blank line
        if current_abstract and abstract_id:
            _flush_abstract(abstract_id, current_abstract, rows)

    if pd is not None:
        return pd.DataFrame(rows)
    return rows


def prepare_datasets(config_path: str = "configs/config.yaml") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reads raw train.txt, dev.txt, test.txt, parses them, validates integrity,
    and exports train.csv, val.csv, and test.csv into data/processed/.
    """
    config = load_config(config_path)
    data_cfg = config.get("data", {})

    train_raw = data_cfg.get("train_raw", "data/raw/train.txt")
    dev_raw = data_cfg.get("dev_raw", "data/raw/dev.txt")
    test_raw = data_cfg.get("test_raw", "data/raw/test.txt")
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed/"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Parsing train dataset: {train_raw}")
    train_df = parse_pubmed_rct(train_raw)
    print(f"[*] Parsing dev dataset: {dev_raw}")
    val_df = parse_pubmed_rct(dev_raw)
    print(f"[*] Parsing test dataset: {test_raw}")
    test_df = parse_pubmed_rct(test_raw)

    # Sanity checks
    print(f"[OK] Train set: {len(train_df):,} sentences across {train_df['abstract_id'].nunique():,} abstracts")
    print(f"[OK] Dev set:   {len(val_df):,} sentences across {val_df['abstract_id'].nunique():,} abstracts")
    print(f"[OK] Test set:  {len(test_df):,} sentences across {test_df['abstract_id'].nunique():,} abstracts")

    # Export to processed_dir
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[OK] Processed files saved to {processed_dir}")
    return train_df, val_df, test_df


if __name__ == "__main__":
    prepare_datasets()
