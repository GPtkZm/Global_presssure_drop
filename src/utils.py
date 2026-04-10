"""
utils.py
--------
Shared utility functions used across the pipeline:
  - Seeding for reproducibility
  - Per-dataset normalisation statistics (mean / std)
  - Saving / loading normalisation stats to disk (JSON)
  - Metric helpers (MAE, MSE, MRE/MAPE, R², RMSE)
"""

import json
import os
import random

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Small epsilon to avoid division by zero in normalization and metrics
_EPS = 1e-8


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix random seeds for Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def compute_stats(values: np.ndarray):
    """Return (mean, std) for a 1-D or 2-D array.  std is clipped to ≥ _EPS."""
    mean = float(np.mean(values))
    std = float(np.std(values))
    std = max(std, _EPS)
    return mean, std


def normalize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Z-score normalise *values* using pre-computed mean and std."""
    return (values - mean) / std


def denormalize(values, mean: float, std: float):
    """Inverse of z-score normalisation.  Accepts tensors or ndarrays."""
    return values * std + mean


# ---------------------------------------------------------------------------
# Checkpoint / stats persistence
# ---------------------------------------------------------------------------

def save_norm_stats(stats: dict, path: str) -> None:
    """Persist normalisation statistics to a JSON file.

    Parameters
    ----------
    stats : dict
        Mapping of stat name → value (all values must be JSON-serialisable).
    path : str
        Output file path (parent directories are created if needed).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_norm_stats(path: str) -> dict:
    """Load normalisation statistics from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = _EPS) -> float:
    """Mean Absolute Percentage Error (in percent)."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + _EPS))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def mre(y_true: np.ndarray, y_pred: np.ndarray, eps: float = _EPS) -> float:
    """Mean Relative Error (same as MAPE, in percent)."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a dict with MAE, MSE, MRE, MAPE, R², and RMSE."""
    return {
        "MAE": mae(y_true, y_pred),
        "MSE": mse(y_true, y_pred),
        "MRE": mre(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
    }
