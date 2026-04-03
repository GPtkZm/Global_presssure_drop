"""
main.py
-------
One-click entry point for the full pipeline:
  1. Check data files (npy + csv)
  2. Auto-add 'val' split if missing
  3. Train the dual-subnet HeteroGNN
  4. Evaluate on test set

Usage:
  python main.py                              # default: 300 epochs, batch_size=16
  python main.py --epochs 100 --batch_size 8  # custom
  python main.py --eval_only                  # skip training, only evaluate
"""

import argparse
import glob
import os
import sys

import pandas as pd


def check_data():
    """Check that data files are in place."""
    from src.config import DATA_DIR, LABEL_CSV

    print("=" * 60)
    print("Step 1: Checking data files")
    print("=" * 60)

    # Check CSV
    if not os.path.exists(LABEL_CSV):
        print(f"  ❌ CSV not found: {LABEL_CSV}")
        sys.exit(1)
    df = pd.read_csv(LABEL_CSV)
    print(f"  ✅ CSV loaded: {len(df)} cases")

    # Check npy directory
    if not os.path.isdir(DATA_DIR):
        print(f"  ❌ npy directory not found: {DATA_DIR}")
        sys.exit(1)
    npy_files = glob.glob(os.path.join(DATA_DIR, "*_topo.npy"))
    print(f"  ✅ npy files found: {len(npy_files)}")

    if len(npy_files) == 0:
        print(f"  ❌ No *_topo.npy files in {DATA_DIR}")
        sys.exit(1)

    # Check required CSV columns
    from src.config import GLOBAL_FEATURE_COLUMNS
    missing_cols = [c for c in GLOBAL_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"  ⚠️  Missing CSV columns for global features: {missing_cols}")
    else:
        print(f"  ✅ All 14 global feature columns present")

    required = ["ID", "split", "drop"]
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        print(f"  ❌ Missing required CSV columns: {missing_req}")
        sys.exit(1)

    return df


def ensure_val_split(df):
    """If CSV has no 'val' split, create one from 10% of train."""
    from src.config import LABEL_CSV

    print("\n" + "=" * 60)
    print("Step 2: Checking train/val/test splits")
    print("=" * 60)

    split_counts = df["split"].value_counts()
    print(f"  Current splits: {dict(split_counts)}")

    if "val" not in split_counts:
        print("  ⚠️  No 'val' split found, creating from 10% of train ...")
        train_mask = df["split"] == "train"
        val_indices = df[train_mask].sample(frac=0.1, random_state=42).index
        df.loc[val_indices, "split"] = "val"
        df.to_csv(LABEL_CSV, index=False)

        split_counts = df["split"].value_counts()
        print(f"  ✅ Updated splits: {dict(split_counts)}")
    else:
        print("  ✅ All splits present")

    if "train" not in split_counts:
        print("  ❌ No 'train' split found in CSV.")
        sys.exit(1)

    if "test" not in split_counts:
        print("  ❌ No 'test' split found in CSV.")
        sys.exit(1)

    return df


def run_train(args):
    """Run training."""
    print("\n" + "=" * 60)
    print("Step 3: Training")
    print("=" * 60)

    cmd = (
        f"{sys.executable} -m src.train"
        f" --epochs {args.epochs}"
        f" --batch_size {args.batch_size}"
        f" --lr {args.lr}"
        f" --hidden_dim {args.hidden_dim}"
        f" --num_layers {args.num_layers}"
        f" --dropout {args.dropout}"
        f" --patience {args.patience}"
        f" --seed {args.seed}"
    )
    print(f"  Running: {cmd}\n")
    ret = os.system(cmd)
    if ret != 0:
        print(f"  ❌ Training failed with exit code {ret}")
        sys.exit(1)


def run_eval(args):
    """Run evaluation."""
    print("\n" + "=" * 60)
    print("Step 4: Evaluation")
    print("=" * 60)

    cmd = (
        f"{sys.executable} -m src.evaluate"
        f" --split test"
        f" --batch_size {args.batch_size}"
    )
    print(f"  Running: {cmd}\n")
    ret = os.system(cmd)
    if ret != 0:
        print(f"  ❌ Evaluation failed with exit code {ret}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Global Pressure Drop Prediction - Full Pipeline"
    )
    # Training params
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    # Mode
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Skip training, only run evaluation on test set"
    )
    args = parser.parse_args()

    # Step 1
    df = check_data()

    # Step 2
    df = ensure_val_split(df)

    # Step 3
    if not args.eval_only:
        run_train(args)
    else:
        print("\n  ⏭️  Skipping training (--eval_only)")

    # Step 4
    run_eval(args)

    print("\n" + "=" * 60)
    print("✅ Pipeline complete!")
    print("=" * 60)
    print("  Outputs:")
    print("    - Checkpoint:    checkpoints/best_model.pt")
    print("    - Norm stats:    checkpoints/norm_stats.json")
    print("    - History:       checkpoints/training_history.npz")
    print("    - Scatter plot:  results/scatter.png")
    print("    - Metrics:       results/metrics_test.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
