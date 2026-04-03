"""
evaluate.py
-----------
Standalone evaluation script for the trained pressure-drop GNN.

What this script does
~~~~~~~~~~~~~~~~~~~~~
1. Loads the best model checkpoint saved by train.py.
2. Runs inference on the test split.
3. De-normalises predictions and ground-truth labels.
4. Computes MAE, MAPE, R², RMSE on the original pressure-drop scale.
5. Saves a scatter plot (predicted vs true) to results/scatter.png.

Usage
~~~~~
  python -m src.evaluate
  python -m src.evaluate --split test      # default
  python -m src.evaluate --split val
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DROPOUT,
    HIDDEN_DIM,
    NUM_LAYERS,
    RESULTS_DIR,
    SEED,
)
from src.dataset import PressureDropDataset
from src.model import HeteroGNN
from src.utils import compute_all_metrics, denormalize, load_norm_stats, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained pressure-drop GNN."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(CHECKPOINT_DIR, "best_model.pt"),
        help="Path to the model checkpoint.",
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> HeteroGNN:
    """Load the HeteroGNN from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Recover architecture dimensions from the checkpoint metadata
    point_in_dim = ckpt["point_in_dim"]
    face_in_dim = ckpt["face_in_dim"]
    train_args = ckpt.get("args", {})

    model = HeteroGNN(
        point_in_dim=point_in_dim,
        face_in_dim=face_in_dim,
        hidden_dim=train_args.get("hidden_dim", HIDDEN_DIM),
        num_layers=train_args.get("num_layers", NUM_LAYERS),
        dropout=train_args.get("dropout", DROPOUT),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def run_inference(model, loader, norm_stats, device):
    """Collect de-normalised predictions and ground-truth values.

    Returns
    -------
    y_true : np.ndarray  (N,)
    y_pred : np.ndarray  (N,)
    """
    all_pred, all_true = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).squeeze(-1).cpu().numpy()
            true = batch.y.squeeze(-1).cpu().numpy()

            pred_orig = denormalize(pred, norm_stats["drop_mean"], norm_stats["drop_std"])
            true_orig = denormalize(true, norm_stats["drop_mean"], norm_stats["drop_std"])

            all_pred.append(pred_orig)
            all_true.append(true_orig)

    return np.concatenate(all_true), np.concatenate(all_pred)


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    """Generate and save a scatter plot of predicted vs true pressure drop."""
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_true, y_pred, alpha=0.6, s=20, edgecolors="none", color="steelblue")

    # Perfect prediction line
    lim_min = min(y_true.min(), y_pred.min()) * 0.95
    lim_max = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1.5, label="y = x")

    ax.set_xlabel("True Pressure Drop (Pa)", fontsize=13)
    ax.set_ylabel("Predicted Pressure Drop (Pa)", fontsize=13)
    ax.set_title("Predicted vs True Global Pressure Drop", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.grid(True, linestyle="--", alpha=0.4)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Scatter plot saved to {save_path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load normalisation stats ----------------------------------------
    stats_path = os.path.join(CHECKPOINT_DIR, "norm_stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Normalisation stats not found at {stats_path}. "
            "Please run train.py first."
        )
    norm_stats = load_norm_stats(stats_path)

    # ---- Dataset & loader ------------------------------------------------
    # Pass norm_stats so that the dataset reuses the training-set vocabulary
    # and normalisation parameters without re-scanning the files.
    dataset = PressureDropDataset(
        split=args.split,
        norm_stats=norm_stats,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(f"Evaluating on '{args.split}' split ({len(dataset)} samples) …")

    # ---- Load model ------------------------------------------------------
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found at {args.checkpoint}. "
            "Please run train.py first."
        )
    model = load_model(args.checkpoint, device)

    # ---- Inference -------------------------------------------------------
    y_true, y_pred = run_inference(model, loader, norm_stats, device)

    # ---- Metrics ---------------------------------------------------------
    metrics = compute_all_metrics(y_true, y_pred)
    print(
        f"\n{'='*60}\n"
        f"Evaluation results on '{args.split}' split\n"
        f"  MAE:  {metrics['MAE']:.2f} Pa\n"
        f"  MAPE: {metrics['MAPE']:.2f}%\n"
        f"  R²:   {metrics['R2']:.4f}\n"
        f"  RMSE: {metrics['RMSE']:.2f} Pa\n"
        f"{'='*60}"
    )

    # Save metrics to a text file
    metrics_path = os.path.join(RESULTS_DIR, f"metrics_{args.split}.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Split: {args.split}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"Metrics saved to {metrics_path}")

    # ---- Scatter plot ----------------------------------------------------
    scatter_path = os.path.join(RESULTS_DIR, "scatter.png")
    plot_scatter(y_true, y_pred, scatter_path)


if __name__ == "__main__":
    main()
