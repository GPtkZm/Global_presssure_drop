"""
train_fusion.py
---------------
Training script for the three-way fusion model.

Supports two separate tasks (selected via ``--task``):
  - ``drop``          : graph-level global pressure-drop prediction (Task A)
  - ``node_pressure`` : point-level pressure distribution prediction (Task B)

Usage
~~~~~
  # Task A – global pressure drop
  python -m src.train_fusion --task drop --batch_size 8 --epochs 300

  # Task B – node-level pressure distribution
  python -m src.train_fusion --task node_pressure --batch_size 8 --epochs 300

  # Custom cloud path, subsampling, and architecture
  python -m src.train_fusion --task drop \\
      --cloud_npy data/cloud/my_cloud.npy \\
      --max_points 2048 --cloud_k 16 \\
      --cloud_hidden 128 --cloud_layers 6 \\
      --lr 5e-4
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CLOUD_INPUT_DIM,
    CLOUD_K,
    CLOUD_NPY_PATH,
    DROPOUT,
    EPOCHS,
    GLOBAL_FEATURE_COLUMNS,
    GLOBAL_MLP_DIM,
    LR,
    LR_FACTOR,
    LR_MIN,
    LR_PATIENCE,
    MAX_CLOUD_POINTS,
    PATIENCE,
    RESULTS_DIR,
    SEED,
)
from src.dataset_fusion import FusionDataset
from src.models import build_fusion_model
from src.utils import compute_all_metrics, denormalize, set_seed


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train the three-way fusion model.")
    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        default="drop",
        choices=["drop", "node_pressure"],
        help="Training task: 'drop' (Task A) or 'node_pressure' (Task B)",
    )
    # Data
    parser.add_argument("--cloud_npy", type=str, default=None,
                        help="Override path to the point-cloud .npy file")
    parser.add_argument("--max_points", type=int, default=MAX_CLOUD_POINTS,
                        help="Max point-cloud points per sample (random subsample if larger)")
    # Architecture
    parser.add_argument("--cloud_hidden", type=int, default=128,
                        help="Hidden dimension of the PointCloudGNN")
    parser.add_argument("--cloud_layers", type=int, default=6,
                        help="Number of GINEConv layers in PointCloudGNN")
    parser.add_argument("--cloud_k", type=int, default=CLOUD_K,
                        help="K for KNN graph construction")
    parser.add_argument("--topo_hidden", type=int, default=64,
                        help="Hidden dimension of the CADTopoEncoder")
    parser.add_argument("--global_mlp_dim", type=int, default=GLOBAL_MLP_DIM,
                        help="Hidden/output dim for GlobalFeatureMLP")
    # Training
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(model, loader, criterion, device, norm_stats, task: str):
    """Full-pass evaluation.

    Returns (avg_loss, metrics_dict) with metrics on original scale.
    For task='node_pressure', metrics are on per-point pressure scale.
    For task='drop', metrics are on pressure-drop scale.
    """
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            if task == "drop":
                dp_pred, _, _ = model(batch, task="drop")
                y = batch.y.view(-1)
                pred_flat = dp_pred.view(-1)
                loss = criterion(pred_flat, y)
                total_loss += loss.item() * y.shape[0]

                pred_np = denormalize(
                    pred_flat.cpu().numpy(),
                    norm_stats["drop_mean"],
                    norm_stats["drop_std"],
                )
                true_np = denormalize(
                    y.cpu().numpy(),
                    norm_stats["drop_mean"],
                    norm_stats["drop_std"],
                )
                all_pred.append(pred_np)
                all_true.append(true_np)

            else:  # node_pressure
                _, p_pred, cloud_batch = model(batch, task="node_pressure")
                # cloud_y is stored as (N, 1) per batch
                cloud_n = batch.cloud_n.view(-1)
                cloud_batch_cpu = torch.arange(
                    cloud_n.shape[0], device=cloud_n.device
                ).repeat_interleave(cloud_n)
                y_node = batch.cloud_y.view(-1)  # (N,)
                pred_flat = p_pred.view(-1)
                loss = criterion(pred_flat, y_node)
                total_loss += loss.item() * y_node.shape[0]

                pred_np = denormalize(
                    pred_flat.cpu().numpy(),
                    norm_stats["cloud_pres_mean"],
                    norm_stats["cloud_pres_std"],
                )
                true_np = denormalize(
                    y_node.cpu().numpy(),
                    norm_stats["cloud_pres_mean"],
                    norm_stats["cloud_pres_std"],
                )
                all_pred.append(pred_np)
                all_true.append(true_np)

    n = sum(p.size for p in all_pred)
    avg_loss = total_loss / max(n, 1)
    y_pred_all = np.concatenate(all_pred)
    y_true_all = np.concatenate(all_true)
    metrics = compute_all_metrics(y_true_all, y_pred_all)
    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Task: {args.task}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Build datasets --------------------------------------------------
    cloud_path = args.cloud_npy or CLOUD_NPY_PATH
    print("Building FusionDataset (train) …")
    train_dataset = FusionDataset(
        split="train",
        cloud_npy_path=cloud_path,
        max_points=args.max_points,
    )
    norm_stats = train_dataset.norm_stats

    print("Building FusionDataset (test) …")
    test_dataset = FusionDataset(
        split="test",
        norm_stats=norm_stats,
        cloud_npy_path=cloud_path,
        max_points=args.max_points,
    )
    print(f"  train={len(train_dataset)}  test={len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # ---- Derive input dimensions from a sample ---------------------------
    sample = train_dataset[0]
    point_in_dim = sample["point"].x.shape[1]
    face_in_dim = sample["face"].x.shape[1]
    edge_in_dim = sample["edge"].x.shape[1]
    print(f"  point_in_dim={point_in_dim}  face_in_dim={face_in_dim}  edge_in_dim={edge_in_dim}")

    # ---- Build model -----------------------------------------------------
    model_cfg = {
        "point_in_dim": point_in_dim,
        "face_in_dim": face_in_dim,
        "edge_in_dim": edge_in_dim,
        "global_feature_dim": len(GLOBAL_FEATURE_COLUMNS),
        "cloud_in_dim": CLOUD_INPUT_DIM,
        "cloud_hidden_dim": args.cloud_hidden,
        "cloud_num_layers": args.cloud_layers,
        "cloud_k": args.cloud_k,
        "topo_hidden_dim": args.topo_hidden,
        "global_mlp_dim": args.global_mlp_dim,
        "dropout": args.dropout,
    }
    model = build_fusion_model(model_cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=LR_MIN,
    )
    criterion = nn.MSELoss()

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR, f"best_fusion_{args.task}_model.pt"
    )

    best_test_loss = float("inf")
    patience_counter = 0
    history = []

    print(f"\nStarting training ({args.epochs} epochs max, patience={args.patience}) …\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        num_train = 0

        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}",
            leave=False,
        ):
            batch = batch.to(device)
            optimizer.zero_grad()

            if args.task == "drop":
                dp_pred, _, _ = model(batch, task="drop")
                y = batch.y.view(-1)
                loss = criterion(dp_pred.view(-1), y)
                n = y.shape[0]

            else:  # node_pressure
                _, p_pred, _ = model(batch, task="node_pressure")
                y_node = batch.cloud_y.view(-1)
                loss = criterion(p_pred.view(-1), y_node)
                n = y_node.shape[0]

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_train_loss += loss.item() * n
            num_train += n

        train_loss = total_train_loss / max(num_train, 1)

        # Evaluate (denormalised metrics)
        train_loss_eval, train_metrics = evaluate(
            model, train_loader, criterion, device, norm_stats, args.task
        )
        test_loss, test_metrics = evaluate(
            model, test_loader, criterion, device, norm_stats, args.task
        )
        scheduler.step(test_loss)
        elapsed = time.time() - t0

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_metrics["MAE"],
                "train_r2": train_metrics["R2"],
                "train_rmse": train_metrics["RMSE"],
                "test_loss": test_loss,
                "test_mae": test_metrics["MAE"],
                "test_r2": test_metrics["R2"],
                "test_rmse": test_metrics["RMSE"],
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f}  "
            f"train_MAE={train_metrics['MAE']:.4f}  "
            f"train_R²={train_metrics['R2']:.4f}  "
            f"| "
            f"test_loss={test_loss:.4f}  "
            f"test_MAE={test_metrics['MAE']:.4f}  "
            f"test_R²={test_metrics['R2']:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"[{elapsed:.1f}s]"
        )

        # ---- Early stopping & checkpoint ---------------------------------
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_loss": test_loss,
                    "args": vars(args),
                    "model_cfg": model_cfg,
                    "task": args.task,
                },
                checkpoint_path,
            )
            print(f"  ✓ Best model saved  (test_loss={best_test_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"\nEarly stopping triggered after {epoch} epochs "
                    f"(no improvement for {args.patience} consecutive epochs)."
                )
                break

    # ---- Save training history -------------------------------------------
    history_path = os.path.join(
        CHECKPOINT_DIR, f"fusion_{args.task}_training_history.npz"
    )
    np.savez(
        history_path,
        **{k: [h[k] for h in history] for k in history[0]},
    )
    print(f"Training history saved to {history_path}")

    # ---- Final evaluation on test set ------------------------------------
    print("\nLoading best checkpoint for final test evaluation …")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    _, final_metrics = evaluate(
        model, test_loader, criterion, device, norm_stats, args.task
    )

    unit = "Pa" if args.task == "drop" else "(pressure unit)"
    print(
        f"\n{'='*60}\n"
        f"Final Test Results  [task={args.task}]\n"
        f"  MAE:  {final_metrics['MAE']:.4f} {unit}\n"
        f"  MSE:  {final_metrics['MSE']:.4f} {unit}²\n"
        f"  MRE:  {final_metrics['MRE']:.2f}%\n"
        f"  R²:   {final_metrics['R2']:.4f}\n"
        f"  RMSE: {final_metrics['RMSE']:.4f} {unit}\n"
        f"{'='*60}"
    )

    # Save per-case details for Task A
    if args.task == "drop":
        _save_drop_case_details(model, test_loader, norm_stats, device, test_dataset)


def _save_drop_case_details(model, test_loader, norm_stats, device, test_dataset):
    """Save per-case pressure-drop predictions to CSV."""
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            dp_pred, _, _ = model(batch, task="drop")
            pred_np = denormalize(
                dp_pred.view(-1).cpu().numpy(),
                norm_stats["drop_mean"],
                norm_stats["drop_std"],
            )
            true_np = denormalize(
                batch.y.view(-1).cpu().numpy(),
                norm_stats["drop_mean"],
                norm_stats["drop_std"],
            )
            all_pred.extend(pred_np.tolist())
            all_true.extend(true_np.tolist())

    case_ids = test_dataset.labels_df["ID"].tolist()
    abs_err = [abs(t - p) for t, p in zip(all_true, all_pred)]
    rel_err = [
        100.0 * abs(t - p) / (abs(t) + 1e-8) for t, p in zip(all_true, all_pred)
    ]

    rows = sorted(
        zip(case_ids, all_true, all_pred, abs_err, rel_err),
        key=lambda r: r[4],
        reverse=True,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "fusion_drop_test_case_details.csv")
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


if __name__ == "__main__":
    main()
