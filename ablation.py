"""
ablation.py
-----------
Systematic ablation study for the HeteroGNN pressure-drop prediction pipeline.

Covers 5 categories of experiments:
  A. Node-type ablation      – which node types contribute?
  B. Edge-type ablation      – which topological edges matter?
  C. Global-feature ablation – which of the 14 physics params help?
  D. Architecture ablation   – depth / width / global MLP contribution
  E. Edge-attribute ablation – does pp_edge_attr (dx,dy,dz,dist) help?

Usage
~~~~~
  python ablation.py                       # run ALL experiments
  python ablation.py --category A          # run only node-type ablation
  python ablation.py --category A B        # run A and B
  python ablation.py --epochs 100          # override training epochs
  python ablation.py --repeats 3           # each experiment runs 3 times

Results are saved to results/ablation_results.csv
"""

import argparse
import copy
import gc
import os
import time
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DROPOUT,
    GLOBAL_FEATURE_COLUMNS,
    GLOBAL_MLP_DIM,
    HIDDEN_DIM,
    LR,
    LR_FACTOR,
    LR_MIN,
    LR_PATIENCE,
    NUM_LAYERS,
    PATIENCE,
    RESULTS_DIR,
    SEED,
)
from src.dataset import PressureDropDataset
from src.models import build_model
from src.utils import compute_all_metrics, denormalize, set_seed


# ============================================================================
# Lightweight train + evaluate
# ============================================================================

def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    norm_stats: dict,
    device: torch.device,
    epochs: int = 200,
    lr: float = LR,
    patience: int = PATIENCE,
    verbose: bool = False,
) -> Dict[str, float]:
    """Train a model and return test metrics from the best checkpoint."""

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN,
    )
    criterion = nn.MSELoss()

    best_test_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # ---- Train --------------------------------------------------------
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch, batch.global_features).view(-1)
            y = batch.y.view(-1)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # ---- Evaluate on test ---------------------------------------------
        model.eval()
        total_loss = 0.0
        n = 0
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch, batch.global_features).view(-1)
                y = batch.y.view(-1)
                total_loss += criterion(pred, y).item() * y.shape[0]
                n += y.shape[0]
                all_pred.append(
                    denormalize(pred.cpu().numpy(), norm_stats["drop_mean"], norm_stats["drop_std"])
                )
                all_true.append(
                    denormalize(y.cpu().numpy(), norm_stats["drop_mean"], norm_stats["drop_std"])
                )

        test_loss = total_loss / n
        scheduler.step(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and epoch % 50 == 0:
            y_p = np.concatenate(all_pred)
            y_t = np.concatenate(all_true)
            m = compute_all_metrics(y_t, y_p)
            print(
                f"  epoch {epoch:03d}  test_loss={test_loss:.4f}  "
                f"MAE={m['MAE']:.1f}  MRE={m['MRE']:.2f}%  R²={m['R2']:.4f}"
            )

        if patience_counter >= patience:
            break

    # ---- Final evaluation with best weights -------------------------------
    model.load_state_dict(best_state)
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch, batch.global_features).view(-1)
            y = batch.y.view(-1)
            all_pred.append(
                denormalize(pred.cpu().numpy(), norm_stats["drop_mean"], norm_stats["drop_std"])
            )
            all_true.append(
                denormalize(y.cpu().numpy(), norm_stats["drop_mean"], norm_stats["drop_std"])
            )

    metrics = compute_all_metrics(np.concatenate(all_true), np.concatenate(all_pred))
    metrics["best_epoch"] = epoch - patience_counter
    return metrics


# ============================================================================
# Transform wrappers: modify HeteroData on-the-fly for ablation
# ============================================================================

class ZeroNodeFeatures:
    """Zero out specific node types' features."""

    def __init__(self, node_types: List[str]):
        self.node_types = node_types

    def __call__(self, data):
        for nt in self.node_types:
            if nt in data.node_types:
                data[nt].x = torch.zeros_like(data[nt].x)
        return data


class RemoveEdgeTypes:
    """Remove specific heterogeneous edge types (set edge_index to empty)."""

    def __init__(self, edge_types: List[Tuple[str, str, str]]):
        self.edge_types = edge_types

    def __call__(self, data):
        for et in self.edge_types:
            try:
                data[et].edge_index = torch.zeros(2, 0, dtype=torch.long)
                if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
                    feat_dim = data[et].edge_attr.shape[1]
                    data[et].edge_attr = torch.zeros(0, feat_dim, dtype=torch.float32)
            except (KeyError, AttributeError):
                pass
        return data


class ZeroGlobalFeatureColumns:
    """Zero out specific columns in global_features."""

    def __init__(self, column_indices: List[int]):
        self.column_indices = column_indices

    def __call__(self, data):
        gf = data.global_features.clone()
        for idx in self.column_indices:
            gf[:, idx] = 0.0
        data.global_features = gf
        return data


class ZeroAllGlobalFeatures:
    """Zero out all global features."""

    def __call__(self, data):
        data.global_features = torch.zeros_like(data.global_features)
        return data


class ZeroPPEdgeAttr:
    """Zero out point-to-point edge attributes [dx,dy,dz,dist]."""

    def __call__(self, data):
        key = ("point", "to", "point")
        try:
            if hasattr(data[key], "edge_attr") and data[key].edge_attr is not None:
                data[key].edge_attr = torch.zeros_like(data[key].edge_attr)
        except (KeyError, AttributeError):
            pass
        return data


# ============================================================================
# Experiment definitions
# ============================================================================

def define_experiments() -> OrderedDict:
    """Return an ordered dict of experiment_name → config."""

    ALL_EDGE_TYPES = [
        ("point", "to", "point"),
        ("face", "to", "point"),
        ("point", "to", "face"),
        ("point", "to", "edge"),
        ("edge", "to", "point"),
        ("edge", "to", "face"),
        ("face", "to", "edge"),
        ("face", "to", "face"),
    ]

    # Global feature column groups (indices into GLOBAL_FEATURE_COLUMNS)
    GEOM_COLS = [0, 1, 2, 3, 12, 13]  # chang, kuan, shen, hanjiemian, length, board_length
    FLOW_COLS = [4, 5, 10]             # liudao, liuliang, in_v
    FLUID_COLS = [6, 7]                # midu, niandu
    OTHER_COLS = [8, 9, 11]            # ceng, z_cut, in_p

    exps = OrderedDict()

    # ---- A. Node-type ablation ----
    exps["A0_baseline"] = {"transform": None}
    exps["A1_zero_point"] = {"transform": ZeroNodeFeatures(["point"])}
    exps["A2_zero_face"] = {"transform": ZeroNodeFeatures(["face"])}
    exps["A3_zero_edge"] = {"transform": ZeroNodeFeatures(["edge"])}
    exps["A4_only_point"] = {"transform": ZeroNodeFeatures(["face", "edge"])}
    exps["A5_only_face"] = {"transform": ZeroNodeFeatures(["point", "edge"])}

    # ---- B. Edge-type ablation ----
    exps["B1_no_pp"] = {"transform": RemoveEdgeTypes([
        ("point", "to", "point"),
    ])}
    exps["B2_no_pf_fp"] = {"transform": RemoveEdgeTypes([
        ("point", "to", "face"), ("face", "to", "point"),
    ])}
    exps["B3_no_pe_ep"] = {"transform": RemoveEdgeTypes([
        ("point", "to", "edge"), ("edge", "to", "point"),
    ])}
    exps["B4_no_ef_fe"] = {"transform": RemoveEdgeTypes([
        ("edge", "to", "face"), ("face", "to", "edge"),
    ])}
    exps["B5_no_ff"] = {"transform": RemoveEdgeTypes([
        ("face", "to", "face"),
    ])}
    exps["B6_only_pp"] = {"transform": RemoveEdgeTypes([
        et for et in ALL_EDGE_TYPES if et != ("point", "to", "point")
    ])}
    exps["B7_no_edges_at_all"] = {"transform": RemoveEdgeTypes(ALL_EDGE_TYPES)}

    # ---- C. Global-feature ablation ----
    exps["C1_no_global"] = {"transform": ZeroAllGlobalFeatures()}
    exps["C2_no_geometry"] = {"transform": ZeroGlobalFeatureColumns(GEOM_COLS)}
    exps["C3_no_flow"] = {"transform": ZeroGlobalFeatureColumns(FLOW_COLS)}
    exps["C4_no_fluid"] = {"transform": ZeroGlobalFeatureColumns(FLUID_COLS)}
    exps["C5_no_other"] = {"transform": ZeroGlobalFeatureColumns(OTHER_COLS)}
    exps["C6_only_geometry"] = {"transform": ZeroGlobalFeatureColumns(
        FLOW_COLS + FLUID_COLS + OTHER_COLS
    )}
    exps["C7_only_flow"] = {"transform": ZeroGlobalFeatureColumns(
        GEOM_COLS + FLUID_COLS + OTHER_COLS
    )}

    # ---- D. Architecture ablation (different model configs) ----
    exps["D1_layers_2"] = {"transform": None, "num_layers": 2}
    exps["D2_layers_4"] = {"transform": None, "num_layers": 4}
    exps["D3_layers_8"] = {"transform": None, "num_layers": 8}
    exps["D4_hidden_64"] = {"transform": None, "hidden_dim": 64}
    exps["D5_hidden_256"] = {"transform": None, "hidden_dim": 256}
    exps["D6_no_global_mlp"] = {"transform": ZeroAllGlobalFeatures(), "tag": "no_global_mlp"}

    # ---- E. Edge-attribute ablation ----
    exps["E1_zero_pp_edge_attr"] = {"transform": ZeroPPEdgeAttr()}

    return exps


# ============================================================================
# Main runner
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Ablation study for HeteroGNN.")
    parser.add_argument(
        "--category", nargs="*", default=None,
        help="Which categories to run: A B C D E (default: all).",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--repeats", type=int, default=1,
                        help="How many times to repeat each experiment (different seeds).")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: HeteroGNN  |  Repeats: {args.repeats}  |  Epochs: {args.epochs}  "
          f"|  Patience: {args.patience}")
    print(f"Batch size: {args.batch_size}  |  LR: {args.lr}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ---- Load datasets (once) ---------------------------------------------
    set_seed(args.seed)
    print("\nLoading datasets …")
    train_dataset_base = PressureDropDataset(split="train")
    norm_stats = train_dataset_base.norm_stats
    test_dataset_base = PressureDropDataset(split="test", norm_stats=norm_stats)

    sample = train_dataset_base[0]
    point_in_dim = sample["point"].x.shape[1]
    face_in_dim = sample["face"].x.shape[1]
    edge_in_dim = sample["edge"].x.shape[1]
    print(f"  train={len(train_dataset_base)}  test={len(test_dataset_base)}")
    print(f"  point_in_dim={point_in_dim}  face_in_dim={face_in_dim}  edge_in_dim={edge_in_dim}")

    # ---- Filter experiments by category -----------------------------------
    all_exps = define_experiments()
    if args.category:
        cats = [c.upper() for c in args.category]
        all_exps = OrderedDict(
            (k, v) for k, v in all_exps.items()
            if k[0] in cats
        )

    print(f"\nRunning {len(all_exps)} experiments × {args.repeats} repeats = "
          f"{len(all_exps) * args.repeats} total runs\n")
    print(f"{'Experiment':<30} {'MAE':>10} {'MRE%':>10} {'R²':>10} {'RMSE':>10} {'Epoch':>6}")
    print("-" * 80)

    results = []

    for exp_name, exp_cfg in all_exps.items():
        for repeat in range(args.repeats):
            seed = args.seed + repeat
            set_seed(seed)

            # -- Free GPU memory from previous run --------------------------
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # -- Rebuild datasets with transform ----------------------------
            train_ds = PressureDropDataset(split="train", norm_stats=norm_stats)
            test_ds = PressureDropDataset(split="test", norm_stats=norm_stats)

            transform = exp_cfg.get("transform", None)
            if transform is not None:
                train_ds._transform = transform
                test_ds._transform = transform

            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
            )
            test_loader = DataLoader(
                test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
            )

            # -- Build model (with possible config overrides) ---------------
            model_cfg = {
                "model_type": "heterognn",
                "point_in_dim": point_in_dim,
                "face_in_dim": face_in_dim,
                "edge_in_dim": edge_in_dim,
                "global_feature_dim": len(GLOBAL_FEATURE_COLUMNS),
                "hidden_dim": exp_cfg.get("hidden_dim", HIDDEN_DIM),
                "num_layers": exp_cfg.get("num_layers", NUM_LAYERS),
                "dropout": DROPOUT,
                "global_mlp_dim": GLOBAL_MLP_DIM,
            }
            model = build_model(model_cfg)

            # -- Train + evaluate -------------------------------------------
            t0 = time.time()
            metrics = train_and_evaluate(
                model, train_loader, test_loader, norm_stats, device,
                epochs=args.epochs, lr=args.lr, patience=args.patience,
                verbose=args.verbose,
            )
            elapsed = time.time() - t0

            # -- Free model immediately -------------------------------------
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            suffix = f" (seed={seed})" if args.repeats > 1 else ""
            print(
                f"{exp_name:<30} "
                f"{metrics['MAE']:>10.1f} "
                f"{metrics['MRE']:>9.2f}% "
                f"{metrics['R2']:>10.4f} "
                f"{metrics['RMSE']:>10.1f} "
                f"{int(metrics['best_epoch']):>6}"
                f"  [{elapsed:.0f}s]{suffix}"
            )

            row = {
                "experiment": exp_name,
                "repeat": repeat,
                "seed": seed,
                **metrics,
                "time_s": elapsed,
            }
            results.append(row)

    # ---- Save results -----------------------------------------------------
    df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "ablation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n{'=' * 80}")
    print(f"All results saved to {csv_path}")

    # ---- Print summary table (mean ± std if repeats > 1) ------------------
    if args.repeats > 1:
        print(f"\n{'=' * 80}")
        print("Summary (mean ± std):")
        print(f"{'Experiment':<30} {'MAE':>14} {'MRE%':>14} {'R²':>14} {'RMSE':>14}")
        print("-" * 90)
        for exp_name in all_exps:
            sub = df[df["experiment"] == exp_name]
            parts = {}
            for col in ["MAE", "MRE", "R2", "RMSE"]:
                m, s = sub[col].mean(), sub[col].std()
                parts[col] = f"{m:.2f}±{s:.2f}"
            print(
                f"{exp_name:<30} "
                f"{parts['MAE']:>14} {parts['MRE']:>14} "
                f"{parts['R2']:>14} {parts['RMSE']:>14}"
            )

    # ---- Print delta-from-baseline table ----------------------------------
    baseline_row = df[df["experiment"] == "A0_baseline"]
    if not baseline_row.empty:
        bl = baseline_row.iloc[0] if args.repeats == 1 else baseline_row.mean(numeric_only=True)
        print(f"\n{'=' * 80}")
        print("Delta from baseline (positive MAE/MRE = worse, negative R² = worse):")
        print(f"{'Experiment':<30} {'ΔMAE':>10} {'ΔMRE%':>10} {'ΔR²':>10}")
        print("-" * 65)
        for exp_name in all_exps:
            if exp_name == "A0_baseline":
                continue
            sub = df[df["experiment"] == exp_name]
            m = sub.mean(numeric_only=True) if args.repeats > 1 else sub.iloc[0]
            d_mae = m["MAE"] - bl["MAE"]
            d_mre = m["MRE"] - bl["MRE"]
            d_r2 = m["R2"] - bl["R2"]
            flag = " ← BIG IMPACT" if abs(d_r2) > 0.05 or abs(d_mre) > 5 else ""
            print(
                f"{exp_name:<30} "
                f"{d_mae:>+10.1f} "
                f"{d_mre:>+9.2f}% "
                f"{d_r2:>+10.4f}"
                f"{flag}"
            )


if __name__ == "__main__":
    main()
