"""
main.py
-------
One-click entry point for the full pipeline:
  1. Check data files (npy + csv)
  2. Check train/test splits exist
  3. Train the model specified in config.py
  4. Evaluate on test set

Config-first design
~~~~~~~~~~~~~~~~~~~
  All hyperparameters default to the values in src/config.py.
  Command-line arguments are optional overrides.

Usage
~~~~~
  python main.py                              # use config.py defaults
  python main.py --epochs 100 --batch_size 8  # override specific params
  python main.py --eval_only                  # skip training, only evaluate

DDP (multi-GPU)
~~~~~~~~~~~~~~~
  Set USE_DDP=True and NUM_GPUS=4 in src/config.py, then run:
    torchrun --nproc_per_node=4 main.py
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

    if not os.path.exists(LABEL_CSV):
        print(f"  ❌ CSV not found: {LABEL_CSV}")
        sys.exit(1)
    df = pd.read_csv(LABEL_CSV)
    print(f"  ✅ CSV loaded: {len(df)} cases")

    if not os.path.isdir(DATA_DIR):
        print(f"  ❌ npy directory not found: {DATA_DIR}")
        sys.exit(1)
    npy_files = glob.glob(os.path.join(DATA_DIR, "*_topo.npy"))
    print(f"  ✅ npy files found: {len(npy_files)}")

    if len(npy_files) == 0:
        print(f"  ❌ No *_topo.npy files in {DATA_DIR}")
        sys.exit(1)

    from src.config import GLOBAL_FEATURE_COLUMNS
    missing_cols = [c for c in GLOBAL_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"  ⚠️  Missing CSV columns for global features: {missing_cols}")
    else:
        print(f"  ✅ All {len(GLOBAL_FEATURE_COLUMNS)} global feature columns present")

    required = ["ID", "split", "drop"]
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        print(f"  ❌ Missing required CSV columns: {missing_req}")
        sys.exit(1)

    return df


def check_splits(df):
    """Verify that train and test splits are present in the CSV."""
    print("\n" + "=" * 60)
    print("Step 2: Checking train/test splits")
    print("=" * 60)

    split_counts = df["split"].value_counts()
    print(f"  Current splits: {dict(split_counts)}")

    if "train" not in split_counts:
        print("  ❌ No 'train' split found in CSV.")
        sys.exit(1)
    if "test" not in split_counts:
        print("  ❌ No 'test' split found in CSV.")
        sys.exit(1)

    print("  ✅ train and test splits present")
    return df


def run_train(args):
    """Run training."""
    from src.config import MODEL_TYPE, USE_DDP, NUM_GPUS

    print("\n" + "=" * 60)
    print("Step 3: Training")
    print("=" * 60)

    base_cmd = sys.executable
    if USE_DDP:
        cmd = (
            f"torchrun --nproc_per_node={NUM_GPUS} main.py"
            f" --epochs {args.epochs}"
            f" --batch_size {args.batch_size}"
            f" --lr {args.lr}"
            f" --hidden_dim {args.hidden_dim}"
            f" --num_layers {args.num_layers}"
            f" --dropout {args.dropout}"
            f" --patience {args.patience}"
            f" --seed {args.seed}"
            f" --model_type {args.model_type}"
        )
    else:
        cmd = (
            f"{base_cmd} -m src.train"
            f" --epochs {args.epochs}"
            f" --batch_size {args.batch_size}"
            f" --lr {args.lr}"
            f" --hidden_dim {args.hidden_dim}"
            f" --num_layers {args.num_layers}"
            f" --dropout {args.dropout}"
            f" --patience {args.patience}"
            f" --seed {args.seed}"
            f" --model_type {args.model_type}"
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
    # Import config defaults so CLI defaults reflect config values
    from src.config import (
        BATCH_SIZE, DROPOUT, EPOCHS, HIDDEN_DIM, LR,
        MODEL_TYPE, NUM_LAYERS, PATIENCE, SEED,
    )

    parser = argparse.ArgumentParser(
        description="Global Pressure Drop Prediction - Full Pipeline"
    )
    # Training params — defaults from config.py
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
    # Mode
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Skip training, only run evaluation on test set",
    )
    args = parser.parse_args()

    # Step 1
    df = check_data()

    # Step 2
    df = check_splits(df)

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
    print("    - Checkpoint:            checkpoints/best_model.pt")
    print("    - Norm stats:            checkpoints/norm_stats.json")
    print("    - History:               checkpoints/training_history.npz")
    print("    - Scatter plot:          results/scatter.png")
    print("    - Metrics:               results/metrics_test.txt")
    print("    - Per-case predictions:  results/test_case_details.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
