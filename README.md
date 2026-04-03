# Global Pressure Drop Prediction with Heterogeneous GNN

A complete, runnable pipeline for predicting **global pressure drop (Δp)** from
CAD topology data (STP → `.npy`) using a Heterogeneous Graph Neural Network
built on [PyTorch Geometric](https://pyg.org/).

---

## Project Overview

Each engineering case is represented as a heterogeneous graph with two node
types extracted from the CAD topology:

| Node type | Features | Source |
|-----------|----------|--------|
| `point`   | Normalised XYZ coordinates (3-dim) | `vertex_coordinates` |
| `face`    | One-hot surface type ‖ normalised UV bounds | `face_surface_type_ids`, `face_surface_uv_bounds` |

Three edge types capture the topology:

| Edge type | Source matrix |
|-----------|---------------|
| `(point, to, point)` | `vertex_vertex_matrix` |
| `(face,  to, point)` | `face_vertex_matrix` |
| `(point, to, face)`  | Transpose of `face_vertex_matrix` |

The model outputs a single scalar: the normalised global pressure drop Δp.

### Model Architecture

```
Encoder
  point_encoder : MLP(3 → 128)
  face_encoder  : MLP(num_face_types + 4 → 128)

Message Passing  (6 layers)
  Each layer: HeteroConv {
    (point, to, point): SAGEConv(128, 128)
    (face,  to, point): SAGEConv(128, 128)  ← boundary-condition injection
    (point, to, face):  SAGEConv(128, 128)
  }
  + Residual connection + LayerNorm

Readout (Dual Pool)
  g_point = global_mean_pool(h_point)
  g_face  = global_mean_pool(h_face)
  g = concat([g_point, g_face])   # 256-dim

Decoder
  MLP(256 → 128 → 64 → 1)  with ReLU + Dropout(0.1)
```

---

## Repository Structure

```
├── data/
│   ├── topo/          ← place your *_topo.npy files here
│   └── labels.csv     ← place the labels CSV here
├── checkpoints/       ← model checkpoints saved here
├── results/           ← evaluation outputs saved here
├── src/
│   ├── config.py      ← all hyperparameters and paths
│   ├── dataset.py     ← PyG Dataset: npy → HeteroData
│   ├── model.py       ← HeteroGNN architecture
│   ├── train.py       ← training loop
│   ├── evaluate.py    ← evaluation + scatter plot
│   └── utils.py       ← metrics, normalisation, seeding
├── requirements.txt
└── README.md
```

---

## Data Preparation

### 1. Topology files

Copy all 793 `*_topo.npy` files into `data/topo/`:

```
data/topo/
├── DOE001-A1-B2-C1-D1-E6-F1-G1-H1-I7_topo.npy
├── DOE002-A1-B3-C2-D3-E8-F1-G1-H3-I7_topo.npy
└── ...
```

Each file is loaded as:
```python
data = np.load('DOE001-..._topo.npy', allow_pickle=True).item()
```

### 2. Labels CSV

Place `labels.csv` in `data/`.  Required columns:

| Column | Description |
|--------|-------------|
| `ID`   | Case identifier matching the npy filename prefix |
| `drop` | Global pressure drop (Pa) – the regression target |
| `split`| One of `train`, `val`, `test` |

Example:
```csv
split,ID,drop
train,DOE001-A1-B2-C1-D1-E6-F1-G1-H1-I7,34681.02834
val,DOE002-A1-B3-C2-D3-E8-F1-G1-H3-I7,68432.53918
test,DOE003-A1-B4-C1-D2-E7-F1-G1-H5-I1,51234.56789
```

Additional columns (e.g. `size`, `liuliang`, `in_v`, …) are ignored by the
pipeline but may be used in future extensions.

---

## Installation

```bash
pip install -r requirements.txt
```

> **GPU users**: install the CUDA-enabled builds of `torch-scatter` and
> `torch-sparse` following the
> [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

---

## Running the Pipeline

### Training

```bash
python -m src.train
```

Optional flags:
```
--epochs      300       Maximum training epochs
--batch_size  16        Graphs per mini-batch
--lr          1e-3      Initial learning rate
--hidden_dim  128       Hidden dimension of all layers
--num_layers  6         Number of message-passing layers
--dropout     0.1       Decoder dropout probability
--patience    30        Early-stopping patience
--seed        42        Random seed
```

Example with custom settings:
```bash
python -m src.train --epochs 200 --batch_size 8 --lr 5e-4
```

Training output example:
```
Using device: cuda
Loading training dataset and computing normalisation statistics ...
  train=635  val=79  test=79
  num_face_types=8
  point_in_dim=3  face_in_dim=12
  model parameters: 1,054,977

Epoch 001 | train_loss=0.9821  val_loss=0.9134  val_MAE=18423.21  val_MAPE=32.14%  val_R2=0.0713
...
Epoch 045 | train_loss=0.1023  val_loss=0.0981  val_MAE=2341.56   val_MAPE=4.21%   val_R2=0.9312
  ✓ Best model saved  (val_loss=0.098100)
```

The best checkpoint is saved to `checkpoints/best_model.pt`.
Normalisation statistics are saved to `checkpoints/norm_stats.json`.

### Evaluation

```bash
python -m src.evaluate
```

Optional flags:
```
--split       test      Split to evaluate (train / val / test)
--checkpoint  checkpoints/best_model.pt
--batch_size  16
```

Output:
```
============================================================
Evaluation results on 'test' split
  MAE:  2123.45 Pa
  MAPE: 3.87%
  R2:   0.9445
  RMSE: 3012.67 Pa
============================================================
Scatter plot saved to results/scatter.png
```

Metrics are also written to `results/metrics_test.txt` and the scatter plot
(predicted vs true pressure drop) is saved to `results/scatter.png`.

---

## Configuration

All hyperparameters and paths are centralised in `src/config.py`:

```python
DATA_DIR      = 'data/topo'          # directory with .npy files
LABEL_CSV     = 'data/labels.csv'    # labels CSV
HIDDEN_DIM    = 128
NUM_LAYERS    = 6
DROPOUT       = 0.1
LR            = 1e-3
EPOCHS        = 300
BATCH_SIZE    = 16
PATIENCE      = 30
SEED          = 42
```

---

## Implementation Notes

- **Variable graph sizes**: Every case has a different number of vertices/edges/faces.
  PyG's `DataLoader` handles batching automatically via `batch` index vectors.

- **Face type one-hot encoding**: All unique `face_surface_type_names` are
  collected over the entire training split before training begins, ensuring a
  consistent vocabulary across all splits.

- **Normalisation**:
  - Vertex coordinates: z-score (global mean/std over training set)
  - UV bounds: z-score (global mean/std over training set)
  - Pressure drop target: z-score (training set mean/std), de-normalised before
    metric computation. Statistics are persisted in `checkpoints/norm_stats.json`.

- **Gradient clipping**: `max_norm=5.0` is applied per step to prevent
  exploding gradients in deep networks.

- **Early stopping**: Training halts if validation loss does not improve for
  `PATIENCE` consecutive epochs.
