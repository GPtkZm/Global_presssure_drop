# Global Pressure Drop Prediction

A complete, runnable pipeline for predicting **global pressure drop (Δp)** from
CAD topology data (STP → `.npy`) using configurable deep-learning models
(HeteroGNN or Transformer) built on [PyTorch Geometric](https://pyg.org/).

---

## Project Overview

Each engineering case is represented as a heterogeneous graph with three node types
extracted from the CAD topology:

| Node type | Features | Source |
|-----------|----------|--------|
| `point`   | Normalised XYZ coordinates (3-dim) | `vertex_coordinates` |
| `face`    | One-hot surface type ‖ normalised UV bounds | `face_surface_type_names`, `face_surface_uv_bounds` |
| `edge`    | Normalised edge parameter ranges (2-dim) | `edge_parameter_ranges` |

Eight edge types capture the topology relationships between nodes.

The model also takes 14-dimensional physics/geometry global features from the CSV
and outputs a single scalar: the predicted global pressure drop Δp.

---

## Repository Structure

```
├── data/
│   ├── topo/          ← place your *_topo.npy files here
│   └── labels.csv     ← place the labels CSV here (train / test splits)
├── checkpoints/       ← model checkpoints saved here
├── results/
│   ├── scatter.png               ← predicted vs true scatter plot
│   ├── metrics_test.txt          ← overall test metrics
│   └── test_case_details.csv     ← per-case predictions (sorted by rel. error)
├── src/
│   ├── config.py      ← all hyperparameters, paths, and model selection
│   ├── dataset.py     ← PyG Dataset: npy → HeteroData
│   ├── model.py       ← backward-compatibility shim
│   ├── models/
│   │   ├── __init__.py    ← build_model() factory function
│   │   ├── heterognn.py   ← HeteroGNN architecture
│   │   └── transformer.py ← Transformer architecture
│   ├── train.py       ← training loop (train/test only, DDP-capable)
│   ├── evaluate.py    ← evaluation + scatter plot + per-case table
│   └── utils.py       ← metrics (MAE, MSE, MRE, MAPE, R², RMSE), normalisation
├── main.py            ← one-click pipeline entry point
└── requirements.txt
```

---

## Data Preparation

### 1. Topology files

Copy all `*_topo.npy` files into `data/topo/`.

### 2. Labels CSV

Place `labels.csv` in `data/`.  Required columns:

| Column | Description |
|--------|-------------|
| `ID`   | Case identifier matching the npy filename prefix |
| `drop` | Global pressure drop (Pa) – the regression target |
| `split`| One of `train`, `test` |

Example:
```csv
split,ID,drop,chang,kuan,...
train,DOE001-A1-B2-C1-D1-E6-F1-G1-H1-I7,34681.02834,...
test,DOE003-A1-B4-C1-D2-E7-F1-G1-H5-I1,51234.56789,...
```

> **Note**: There is no `val` split. Early stopping is based on the **test loss**.

---

## Installation

```bash
pip install -r requirements.txt
```

> **GPU users**: install the CUDA-enabled builds of `torch-scatter` and
> `torch-sparse` following the
> [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

---

## Configuration

All hyperparameters and settings are centralised in `src/config.py`.
**Running `python main.py` without any arguments uses these values directly.**

Key settings:

```python
# Model selection
MODEL_TYPE    = "heterognn"   # "heterognn" or "transformer"

# HeteroGNN hyperparameters
HIDDEN_DIM    = 128
NUM_LAYERS    = 6
DROPOUT       = 0.1

# Transformer hyperparameters
TRANSFORMER_D_MODEL             = 256
TRANSFORMER_NHEAD               = 8
TRANSFORMER_NUM_ENCODER_LAYERS  = 4
TRANSFORMER_DIM_FEEDFORWARD     = 512
TRANSFORMER_DROPOUT             = 0.1
TRANSFORMER_POOL                = "mean"   # "mean" or "max"

# Training hyperparameters
LR            = 1e-3
EPOCHS        = 300
BATCH_SIZE    = 16
PATIENCE      = 30    # early stopping on test loss
SEED          = 42

# Multi-GPU (DDP)
USE_DDP       = False  # set True to enable DistributedDataParallel
NUM_GPUS      = 4      # number of GPUs when USE_DDP=True
```

---

## Running the Pipeline

### One-click (recommended)

```bash
python main.py
```

This runs all steps: data check → train → evaluate.  All parameters are taken
from `config.py`.  Command-line flags are optional overrides:

```bash
python main.py --epochs 100 --batch_size 8 --model_type transformer
python main.py --eval_only     # skip training, only evaluate
```

### Training only

```bash
python -m src.train
python -m src.train --epochs 200 --model_type transformer
```

Per-epoch output includes both **train** and **test** metrics:
```
Epoch 001 | train_loss=0.98  train_MAE=18423  train_MSE=3.4e8  train_MRE=32.1%  train_R²=0.07
           | test_loss=0.91   test_MAE=16821   test_MSE=2.8e8   test_MRE=29.4%   test_R²=0.09
```

After training, a **per-case prediction table** is printed and saved:

```
Case ID                        True (Pa)      Pred (Pa)  Abs Err (Pa)  Rel Err (%)
----------------------------------------------------------------------
DOE045-...                      68432.54       82341.21      13908.67        20.32%
DOE012-...                      34681.03       38219.45       3538.42        10.20%
...
```

The per-case table is also saved to `results/test_case_details.csv`.

### Evaluation only

```bash
python -m src.evaluate
python -m src.evaluate --split test
```

Output metrics:
```
  MAE:  2123.45 Pa
  MSE:  4508037.23 Pa²
  MRE:  3.87%
  MAPE: 3.87%
  R²:   0.9445
  RMSE: 2123.45 Pa
```

### Multi-GPU training (DDP)

1. Set `USE_DDP = True` and `NUM_GPUS = 4` in `src/config.py`.
2. Launch with `torchrun`:

```bash
torchrun --nproc_per_node=4 main.py
# or directly:
torchrun --nproc_per_node=4 -m src.train
```

Only rank-0 prints logs, saves checkpoints, and runs final evaluation.

---

## Outputs

| File | Description |
|------|-------------|
| `checkpoints/best_model.pt` | Best model checkpoint (lowest test loss) |
| `checkpoints/norm_stats.json` | Normalisation statistics from training set |
| `checkpoints/training_history.npz` | Per-epoch train/test metrics |
| `results/scatter.png` | Predicted vs true scatter plot |
| `results/metrics_test.txt` | Overall test metrics |
| `results/test_case_details.csv` | Per-case predictions sorted by relative error |

---

## Implementation Notes

- **No validation split**: Data is split into `train` and `test` only.
  Early stopping monitors the **test loss**.

- **Multiple models**: Select the model via `MODEL_TYPE` in `config.py`.
  Both `HeteroGNN` and `TransformerPressureDrop` accept the same `HeteroData`
  input from `PressureDropDataset`.

- **Variable graph sizes**: Every case has a different number of vertices/edges/faces.
  PyG's `DataLoader` handles batching automatically via `batch` index vectors.

- **Gradient clipping**: `max_norm=5.0` is applied per step.

- **Normalisation**: Vertex coordinates, UV bounds, edge parameters, global
  features, and target pressure drop are all z-score normalised using
  training-set statistics saved to `checkpoints/norm_stats.json`.
