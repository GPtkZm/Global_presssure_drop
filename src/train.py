"""
train.py
--------
Training loop for the global pressure-drop prediction pipeline.

What this script does
~~~~~~~~~~~~~~~~~~~~~
1. Builds train / test splits from the label CSV using the 'split' column.
2. Computes normalisation statistics on the training set (or reloads them).
3. Trains the selected model (from config MODEL_TYPE) with Adam +
   ReduceLROnPlateau scheduling.
4. Applies early stopping based on test loss.
5. Saves the best model checkpoint and training history.
6. After training, prints and saves per-case predictions for the test set.

DDP support
~~~~~~~~~~~
When ``USE_DDP=True`` in config.py, launch with:
    torchrun --nproc_per_node=<NUM_GPUS> main.py

Usage (single GPU / CPU)
~~~~~~~~~~~~~~~~~~~~~~~~
  python -m src.train              # use defaults from config.py
  python -m src.train --epochs 100 --batch_size 8 --lr 5e-4
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DROPOUT,
    EPOCHS,
    GLOBAL_FEATURE_COLUMNS,
    GLOBAL_MLP_DIM,
    HIDDEN_DIM,
    LR,
    LR_FACTOR,
    LR_MIN,
    LR_PATIENCE,
    MODEL_TYPE,
    NUM_GPUS,
    NUM_LAYERS,
    PATIENCE,
    RESULTS_DIR,
    SEED,
    USE_DDP,
)
from src.dataset import PressureDropDataset
from src.models import build_model
from src.utils import compute_all_metrics, denormalize, load_norm_stats, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train the pressure-drop model.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--model_type", type=str, default=MODEL_TYPE,
        help="Model type: 'heterognn' or 'transformer'",
    )
    return parser.parse_args()


def is_main_process():
    """Return True if this is the main (rank-0) process or single-GPU."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_raw_model(model):
    """Unwrap DDP to get the underlying module."""
    return model.module if isinstance(model, DDP) else model


def evaluate_split(model, loader, criterion, device, norm_stats):
    """Run one full pass over *loader* and return loss + all metrics (de-normalised).

    Returns
    -------
    tuple : (avg_loss, metrics_dict)  – metrics on the original scale
    """
    # Unwrap DDP for inference
    raw_model = get_raw_model(model)
    raw_model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = raw_model(batch, batch.global_features).view(-1)
            y = batch.y.view(-1)
            loss = criterion(pred, y)
            total_loss += loss.item() * y.shape[0]

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

    n = sum(p.size for p in all_pred)
    avg_loss = total_loss / n
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    metrics = compute_all_metrics(y_true, y_pred)
    return avg_loss, metrics


def evaluate_split_with_ids(model, loader, norm_stats, device, case_ids):
    """Run inference and return per-sample (id, true, pred) triples."""
    raw_model = get_raw_model(model)
    raw_model.eval()
    all_ids = []
    all_true = []
    all_pred = []
    idx = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = raw_model(batch, batch.global_features).view(-1).cpu().numpy()
            true = batch.y.view(-1).cpu().numpy()

            pred_orig = denormalize(pred, norm_stats["drop_mean"], norm_stats["drop_std"])
            true_orig = denormalize(true, norm_stats["drop_mean"], norm_stats["drop_std"])

            batch_size = len(pred_orig)
            all_ids.extend(case_ids[idx: idx + batch_size])
            all_true.extend(true_orig.tolist())
            all_pred.extend(pred_orig.tolist())
            idx += batch_size

    return all_ids, all_true, all_pred


def print_and_save_case_details(case_ids, y_true, y_pred, results_dir):
    """Print a per-case prediction table and save it as CSV.

    Sorted by Relative Error (%) descending (worst cases first).
    """
    abs_err = [abs(t - p) for t, p in zip(y_true, y_pred)]
    rel_err = [
        100.0 * abs(t - p) / (abs(t) + 1e-8) for t, p in zip(y_true, y_pred)
    ]

    rows = sorted(
        zip(case_ids, y_true, y_pred, abs_err, rel_err),
        key=lambda r: r[4],
        reverse=True,
    )

    header = f"{'Case ID':<30} {'True (Pa)':>14} {'Pred (Pa)':>14} {'Abs Err (Pa)':>14} {'Rel Err (%)':>12}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print("Per-case prediction accuracy on TEST set (sorted by Relative Error ↓)")
    print(sep)
    print(header)
    print(sep)
    for cid, tr, pr, ae, re in rows:
        print(f"{str(cid):<30} {tr:>14.2f} {pr:>14.2f} {ae:>14.2f} {re:>12.2f}%")
    print(sep)

    # Save CSV
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


def setup_ddp():
    """Initialise the default process group for DDP."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main():
    args = parse_args()

    # ---- DDP initialisation ----------------------------------------------
    use_ddp = USE_DDP and torch.cuda.is_available()
    local_rank = 0
    if use_ddp:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        print(f"Using device: {device}  |  DDP: {use_ddp}  |  Model: {args.model_type}")

    set_seed(args.seed)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Build datasets --------------------------------------------------
    if is_main_process():
        print("Loading training dataset and computing normalisation statistics …")
    train_dataset = PressureDropDataset(split="train")
    norm_stats = train_dataset.norm_stats
    test_dataset = PressureDropDataset(split="test", norm_stats=norm_stats)

    if is_main_process():
        print(f"  train={len(train_dataset)}  test={len(test_dataset)}")
        print(f"  num_face_types={train_dataset.num_face_types}")

    # ---- DataLoaders -----------------------------------------------------
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        train_shuffle = False
    else:
        train_sampler = None
        test_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=0,
    )

    # ---- Derive input dimensions -----------------------------------------
    sample = train_dataset[0]
    point_in_dim = sample["point"].x.shape[1]
    face_in_dim = sample["face"].x.shape[1]
    edge_in_dim = sample["edge"].x.shape[1]
    if is_main_process():
        print(f"  point_in_dim={point_in_dim}  edge_in_dim={edge_in_dim}  face_in_dim={face_in_dim}")

    # ---- Build model via factory ------------------------------------------
    model_cfg = {
        "model_type": args.model_type,
        "point_in_dim": point_in_dim,
        "face_in_dim": face_in_dim,
        "edge_in_dim": edge_in_dim,
        "global_feature_dim": len(GLOBAL_FEATURE_COLUMNS),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "global_mlp_dim": GLOBAL_MLP_DIM,
    }
    model = build_model(model_cfg).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    if is_main_process():
        raw_model = get_raw_model(model)
        num_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
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

    # ---- Training loop ----------------------------------------------------
    best_test_loss = float("inf")
    patience_counter = 0
    history = []
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    if is_main_process():
        print(f"\nStarting training ({args.epochs} epochs max, patience={args.patience}) …\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if use_ddp:
            train_sampler.set_epoch(epoch)

        # ---- Train -------------------------------------------------------
        model.train()
        raw_model_train = get_raw_model(model)
        total_train_loss = 0.0
        num_train = 0
        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}",
            leave=False,
            disable=not is_main_process(),
        ):
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = raw_model_train(batch, batch.global_features).view(-1)
            y = batch.y.view(-1)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_train_loss += loss.item() * y.shape[0]
            num_train += y.shape[0]

        train_loss = total_train_loss / num_train

        # Collect full train metrics (de-normalised)
        train_loss_full, train_metrics = evaluate_split(
            model, train_loader, criterion, device, norm_stats
        )
        # Test metrics
        test_loss, test_metrics = evaluate_split(
            model, test_loader, criterion, device, norm_stats
        )

        scheduler.step(test_loss)
        elapsed = time.time() - t0

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_metrics["MAE"],
                "train_mse": train_metrics["MSE"],
                "train_mre": train_metrics["MRE"],
                "train_r2": train_metrics["R2"],
                "train_rmse": train_metrics["RMSE"],
                "test_loss": test_loss,
                "test_mae": test_metrics["MAE"],
                "test_mse": test_metrics["MSE"],
                "test_mre": test_metrics["MRE"],
                "test_r2": test_metrics["R2"],
                "test_rmse": test_metrics["RMSE"],
            }
        )

        if is_main_process():
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f}  "
                f"train_MAE={train_metrics['MAE']:.2f}  "
                f"train_MSE={train_metrics['MSE']:.2f}  "
                f"train_MRE={train_metrics['MRE']:.2f}%  "
                f"train_R²={train_metrics['R2']:.4f}  "
                f"| "
                f"test_loss={test_loss:.4f}  "
                f"test_MAE={test_metrics['MAE']:.2f}  "
                f"test_MSE={test_metrics['MSE']:.2f}  "
                f"test_MRE={test_metrics['MRE']:.2f}%  "
                f"test_R²={test_metrics['R2']:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"[{elapsed:.1f}s]"
            )

        # ---- Early stopping & checkpoint (rank-0 only) -------------------
        if is_main_process():
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
                raw_model = get_raw_model(model)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "test_loss": test_loss,
                        "args": vars(args),
                        "model_cfg": model_cfg,
                        "point_in_dim": point_in_dim,
                        "edge_in_dim": edge_in_dim,
                        "face_in_dim": face_in_dim,
                        "global_feature_dim": len(GLOBAL_FEATURE_COLUMNS),
                        "num_face_types": train_dataset.num_face_types,
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

        # Synchronise early-stopping signal across DDP ranks
        if use_ddp:
            stop_tensor = torch.tensor(
                [1 if patience_counter >= args.patience else 0],
                device=device,
            )
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item() == 1:
                break

    # ---- Save training history -------------------------------------------
    if is_main_process():
        history_path = os.path.join(CHECKPOINT_DIR, "training_history.npz")
        np.savez(
            history_path,
            **{k: [h[k] for h in history] for k in history[0]},
        )
        print(f"Training history saved to {history_path}")

    # ---- Final per-case evaluation on test set ---------------------------
    if is_main_process():
        print("\nLoading best checkpoint for final test evaluation …")
        ckpt = torch.load(checkpoint_path, map_location=device)
        raw_model = get_raw_model(model)
        raw_model.load_state_dict(ckpt["model_state_dict"])

        _, final_test_metrics = evaluate_split(
            model, test_loader, criterion, device, norm_stats
        )
        print(
            f"\n{'='*60}\n"
            f"Final Test Results\n"
            f"  MAE:  {final_test_metrics['MAE']:.2f} Pa\n"
            f"  MSE:  {final_test_metrics['MSE']:.2f} Pa²\n"
            f"  MRE:  {final_test_metrics['MRE']:.2f}%\n"
            f"  MAPE: {final_test_metrics['MAPE']:.2f}%\n"
            f"  R²:   {final_test_metrics['R2']:.4f}\n"
            f"  RMSE: {final_test_metrics['RMSE']:.2f} Pa\n"
            f"{'='*60}"
        )

        # Per-case details
        test_case_ids = test_dataset.labels_df["ID"].tolist()
        case_ids, y_true, y_pred = evaluate_split_with_ids(
            model, test_loader, norm_stats, device, test_case_ids
        )
        print_and_save_case_details(case_ids, y_true, y_pred, RESULTS_DIR)

    cleanup_ddp()


if __name__ == "__main__":
    main()
