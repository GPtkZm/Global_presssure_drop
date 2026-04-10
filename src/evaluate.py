"""
evaluate.py
-----------
Standalone evaluation script for the trained pressure-drop model.

What this script does
~~~~~~~~~~~~~~~~~~~~~
1. Loads the best model checkpoint saved by train.py.
2. Runs inference on the test split.
3. De-normalises predictions and ground-truth labels.
4. Computes MAE, MSE, MRE, MAPE, R², RMSE on the original scale.
5. Saves a scatter plot (predicted vs true) to results/scatter.png.
6. Prints and saves a per-case prediction table sorted by relative error.

Usage
~~~~~
  python -m src.evaluate
  python -m src.evaluate --split test      # default
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    GLOBAL_FEATURE_COLUMNS,
    MODEL_TYPE,
    RESULTS_DIR,
    SEED,
)
from src.dataset import PressureDropDataset
from src.models import build_model
from src.utils import compute_all_metrics, denormalize, load_norm_stats, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained pressure-drop model."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
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


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load the model from a checkpoint file using the saved model_cfg."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    model_cfg = ckpt.get("model_cfg", {})
    # Fall back to checkpoint metadata for legacy checkpoints
    if "point_in_dim" not in model_cfg:
        model_cfg["point_in_dim"] = ckpt["point_in_dim"]
    if "face_in_dim" not in model_cfg:
        model_cfg["face_in_dim"] = ckpt["face_in_dim"]
    if "edge_in_dim" not in model_cfg:
        model_cfg["edge_in_dim"] = ckpt.get("edge_in_dim", 2)
    if "global_feature_dim" not in model_cfg:
        model_cfg["global_feature_dim"] = ckpt.get(
            "global_feature_dim", len(GLOBAL_FEATURE_COLUMNS)
        )
    if "model_type" not in model_cfg:
        train_args = ckpt.get("args", {})
        model_cfg["model_type"] = train_args.get("model_type", MODEL_TYPE)

    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def run_inference(model, loader, norm_stats, device):
    """Collect de-normalised predictions and ground-truth values."""
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch, batch.global_features).view(-1).cpu().numpy()
            true = batch.y.view(-1).cpu().numpy()
            pred_orig = denormalize(pred, norm_stats["drop_mean"], norm_stats["drop_std"])
            true_orig = denormalize(true, norm_stats["drop_mean"], norm_stats["drop_std"])
            all_pred.append(pred_orig)
            all_true.append(true_orig)
    return np.concatenate(all_true), np.concatenate(all_pred)


def run_inference_with_ids(model, loader, norm_stats, device, case_ids):
    """Run inference and return per-sample (id, true, pred) triples."""
    all_ids = []
    all_true = []
    all_pred = []
    idx = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch, batch.global_features).view(-1).cpu().numpy()
            true = batch.y.view(-1).cpu().numpy()
            pred_orig = denormalize(pred, norm_stats["drop_mean"], norm_stats["drop_std"])
            true_orig = denormalize(true, norm_stats["drop_mean"], norm_stats["drop_std"])
            batch_size = len(pred_orig)
            all_ids.extend(case_ids[idx: idx + batch_size])
            all_true.extend(true_orig.tolist())
            all_pred.extend(pred_orig.tolist())
            idx += batch_size
    return all_ids, all_true, all_pred


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    """Generate and save a scatter plot of predicted vs true pressure drop."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.6, s=20, edgecolors="none", color="steelblue")
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


def print_and_save_case_details(case_ids, y_true, y_pred, results_dir):
    """Print a per-case prediction table and save it as CSV."""
    abs_err = [abs(t - p) for t, p in zip(y_true, y_pred)]
    rel_err = [
        100.0 * abs(t - p) / (abs(t) + 1e-8) for t, p in zip(y_true, y_pred)
    ]

    rows = sorted(
        zip(case_ids, y_true, y_pred, abs_err, rel_err),
        key=lambda r: r[4],
        reverse=True,
    )

    header = (
        f"{'Case ID':<30} {'True (Pa)':>14} {'Pred (Pa)':>14} "
        f"{'Abs Err (Pa)':>14} {'Rel Err (%)':>12}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print("Per-case prediction accuracy (sorted by Relative Error ↓)")
    print(sep)
    print(header)
    print(sep)
    for cid, tr, pr, ae, re in rows:
        print(f"{str(cid):<30} {tr:>14.2f} {pr:>14.2f} {ae:>14.2f} {re:>12.2f}%")
    print(sep)

    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "test_case_details.csv")
    df = pd.DataFrame(
        {
            "Case_ID": [r[0] for r in rows],
            "True_Pa": [r[1] for r in rows],
            "Predicted_Pa": [r[2] for r in rows],
            "Absolute_Error_Pa": [r[3] for r in rows],
            "Relative_Error_pct": [r[4] for r in rows],
        }
    )
    df.to_csv(csv_path, index=False)
    print(f"Per-case details saved to {csv_path}")


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
    dataset = PressureDropDataset(split=args.split, norm_stats=norm_stats)
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
    model = load_model_from_checkpoint(args.checkpoint, device)

    # ---- Inference -------------------------------------------------------
    y_true, y_pred = run_inference(model, loader, norm_stats, device)

    # ---- Metrics ---------------------------------------------------------
    metrics = compute_all_metrics(y_true, y_pred)
    print(
        f"\n{'='*60}\n"
        f"Evaluation results on '{args.split}' split\n"
        f"  MAE:  {metrics['MAE']:.2f} Pa\n"
        f"  MSE:  {metrics['MSE']:.2f} Pa²\n"
        f"  MRE:  {metrics['MRE']:.2f}%\n"
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

    # ---- Per-case details (test split only) ------------------------------
    if args.split == "test":
        case_ids = dataset.labels_df["ID"].tolist()
        cids, y_true_list, y_pred_list = run_inference_with_ids(
            model, loader, norm_stats, device, case_ids
        )
        print_and_save_case_details(cids, y_true_list, y_pred_list, RESULTS_DIR)


if __name__ == "__main__":
    main()
