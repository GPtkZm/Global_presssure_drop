"""
train.py
--------
Training loop for the global pressure-drop prediction pipeline.

What this script does
~~~~~~~~~~~~~~~~~~~~~
1. Builds train / val / test splits from the label CSV using the 'split' column.
2. Computes normalisation statistics on the training set (or reloads them).
3. Trains the HeteroGNN with Adam + ReduceLROnPlateau scheduling.
4. Applies early stopping based on validation loss.
5. Saves the best model checkpoint and training history.

Usage
~~~~~
  python -m src.train              # use defaults from config.py
  python -m src.train --epochs 100 --batch_size 8 --lr 5e-4
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DROPOUT,
    EPOCHS,
    HIDDEN_DIM,
    LR,
    LR_FACTOR,
    LR_MIN,
    LR_PATIENCE,
    NUM_LAYERS,
    PATIENCE,
    SEED,
)
from src.dataset import PressureDropDataset
from src.model import HeteroGNN
from src.utils import compute_all_metrics, denormalize, load_norm_stats, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train the pressure-drop GNN.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def evaluate_split(model, loader, criterion, device, norm_stats):
    """Run one full pass over *loader* and return loss + metrics (de-normalised).

    Parameters
    ----------
    model : HeteroGNN
    loader : DataLoader
    criterion : loss function
    device : torch.device
    norm_stats : dict  – needed for de-normalisation

    Returns
    -------
    tuple : (avg_loss, mae, mape, r2)  all on the original pressure-drop scale
    """
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).squeeze(-1)   # (B,)
            y = batch.y.squeeze(-1)           # (B,)
            loss = criterion(pred, y)
            total_loss += loss.item() * len(y)

            # Store de-normalised values for metric computation
            pred_np = denormalize(
                pred.cpu().numpy(),
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

    n = sum(len(p) for p in all_pred)
    avg_loss = total_loss / n
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    metrics = compute_all_metrics(y_true, y_pred)
    return avg_loss, metrics["MAE"], metrics["MAPE"], metrics["R2"]


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Build datasets --------------------------------------------------
    print("Loading training dataset and computing normalisation statistics …")
    train_dataset = PressureDropDataset(split="train")
    norm_stats = train_dataset.norm_stats

    # Reuse the stats already computed from the training set
    val_dataset = PressureDropDataset(split="val", norm_stats=norm_stats)
    test_dataset = PressureDropDataset(split="test", norm_stats=norm_stats)

    print(
        f"  train={len(train_dataset)}  val={len(val_dataset)}  test={len(test_dataset)}"
    )
    print(f"  num_face_types={train_dataset.num_face_types}")

    # ---- DataLoaders -----------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ---- Derive input dimensions from first sample -----------------------
    sample = train_dataset[0]
    point_in_dim = sample["point"].x.shape[1]
    face_in_dim = sample["face"].x.shape[1]
    print(f"  point_in_dim={point_in_dim}  face_in_dim={face_in_dim}")

    # ---- Model, optimiser, scheduler, loss --------------------------------
    model = HeteroGNN(
        point_in_dim=point_in_dim,
        face_in_dim=face_in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=LR_MIN,
        verbose=True,
    )
    criterion = nn.MSELoss()

    # ---- Training loop ----------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    print("\nStarting training …\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        num_train = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch).squeeze(-1)    # (B,)
            y = batch.y.squeeze(-1)            # (B,)
            loss = criterion(pred, y)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_train_loss += loss.item() * len(y)
            num_train += len(y)

        train_loss = total_train_loss / num_train

        # Validation
        val_loss, val_mae, val_mape, val_r2 = evaluate_split(
            model, val_loader, criterion, device, norm_stats
        )
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_mape": val_mape,
                "val_r2": val_r2,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_MAE={val_mae:.2f}  "
            f"val_MAPE={val_mape:.2f}%  "
            f"val_R²={val_r2:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"[{elapsed:.1f}s]"
        )

        # Check for improvement and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                    "point_in_dim": point_in_dim,
                    "face_in_dim": face_in_dim,
                    "num_face_types": train_dataset.num_face_types,
                },
                checkpoint_path,
            )
            print(f"  ✓ Best model saved  (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"\nEarly stopping triggered after {epoch} epochs "
                    f"(no improvement for {args.patience} consecutive epochs)."
                )
                break

    # ---- Final evaluation on test set ------------------------------------
    print("\nLoading best checkpoint for final test evaluation …")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_mae, test_mape, test_r2 = evaluate_split(
        model, test_loader, criterion, device, norm_stats
    )
    print(
        f"\n{'='*60}\n"
        f"Test results\n"
        f"  Loss (normalised MSE): {test_loss:.4f}\n"
        f"  MAE:  {test_mae:.2f}\n"
        f"  MAPE: {test_mape:.2f}%\n"
        f"  R²:   {test_r2:.4f}\n"
        f"{'='*60}"
    )

    # Save training history as a NumPy archive for later analysis
    history_path = os.path.join(CHECKPOINT_DIR, "training_history.npz")
    np.savez(
        history_path,
        epoch=[h["epoch"] for h in history],
        train_loss=[h["train_loss"] for h in history],
        val_loss=[h["val_loss"] for h in history],
        val_mae=[h["val_mae"] for h in history],
        val_mape=[h["val_mape"] for h in history],
        val_r2=[h["val_r2"] for h in history],
    )
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
