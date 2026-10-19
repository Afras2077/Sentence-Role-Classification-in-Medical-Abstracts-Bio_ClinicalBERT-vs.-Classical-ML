from __future__ import annotations
"""
Utility functions for PubMed RCT sentence classification project.
Handles random seeding, configuration loading, device checks, and filesystem management.
"""

import os
import random
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List


def set_seed(seed: int = 42):
    """Sets random seeds across python, numpy, and torch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Loads YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(dirs: List[str]):
    """Creates directory list if they do not exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """Returns available PyTorch device (cuda, mps, or cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
