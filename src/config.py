"""
config.py
---------
Central configuration for the Global Pressure Drop prediction pipeline.
All hyperparameters and file paths are defined here so that every other
module can import them from a single location.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Root directory of the project (one level above src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory that contains the per-case topology .npy files
DATA_DIR = os.path.join(ROOT_DIR, "data", "topo")

# CSV file with case IDs, splits, and the pressure-drop label
LABEL_CSV = os.path.join(ROOT_DIR, "data", "labels.csv")

# Directory where model checkpoints and normalisation stats are saved
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")

# Directory where evaluation results (plots, metrics) are saved
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
# Choose which model to use: "heterognn" or "transformer"
MODEL_TYPE = "heterognn"

# ---------------------------------------------------------------------------
# Model hyperparameters — HeteroGNN
# ---------------------------------------------------------------------------
HIDDEN_DIM = 128       # Width of all hidden layers
NUM_LAYERS = 6         # Number of heterogeneous message-passing layers
DROPOUT = 0.1          # Dropout probability in the decoder MLP

# ---------------------------------------------------------------------------
# Model hyperparameters — Transformer
# ---------------------------------------------------------------------------
TRANSFORMER_D_MODEL = 256      # Transformer model dimension
TRANSFORMER_NHEAD = 8          # Number of attention heads
TRANSFORMER_NUM_ENCODER_LAYERS = 4   # Number of TransformerEncoder layers
TRANSFORMER_DIM_FEEDFORWARD = 512    # Feedforward dimension inside transformer
TRANSFORMER_DROPOUT = 0.1      # Dropout in transformer
TRANSFORMER_POOL = "mean"      # Pooling strategy: "mean" or "max"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
LR = 1e-3              # Initial learning rate for Adam
EPOCHS = 300           # Maximum number of training epochs
BATCH_SIZE = 16        # Number of graphs per mini-batch
PATIENCE = 30          # Early-stopping patience (epochs without test improvement)
SEED = 42              # Global random seed for reproducibility

# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
LR_FACTOR = 0.5        # Factor by which the LR is reduced on plateau
LR_PATIENCE = 15       # Epochs to wait before reducing LR
LR_MIN = 1e-6          # Minimum learning rate

# ---------------------------------------------------------------------------
# Physics / global features
# ---------------------------------------------------------------------------
# CSV columns used as graph-level (global) physics/geometry features
GLOBAL_FEATURE_COLUMNS = [
    "chang",        # channel length (m)
    "kuan",         # channel width (m)
    "shen",         # channel depth (m)
    "hanjiemian",   # cross-sectional area (m²)
    "liudao",       # flow path count
    "liuliang",     # flow rate (m³/s)
    "midu",         # fluid density (kg/m³)
    "niandu",       # dynamic viscosity (Pa·s)
    "ceng",         # number of layers
    "z_cut",        # z-direction cut position (m)
    "in_v",         # inlet velocity (m/s)
    "in_p",         # inlet pressure (Pa)
    "length",       # overall length (m)
    "board_length", # board/plate length (m)
]

# Hidden dimension for the physics-parameter MLP subnet
GLOBAL_MLP_DIM = 64

# ---------------------------------------------------------------------------
# Distributed training (DDP)
# ---------------------------------------------------------------------------
# Set USE_DDP=True to enable multi-GPU training with DistributedDataParallel.
# Launch with: torchrun --nproc_per_node=<NUM_GPUS> main.py
USE_DDP = False        # Enable DistributedDataParallel training
NUM_GPUS = 4           # Number of GPUs to use when USE_DDP=True

# ---------------------------------------------------------------------------
# Point cloud data (fusion model)
# ---------------------------------------------------------------------------
# Path to the single .npy file holding all 4,902 point-cloud records
CLOUD_NPY_PATH = os.path.join(ROOT_DIR, "data", "cloud", "cloud_all.npy")

# Feature keys to read from each point-cloud record (7 input channels)
CLOUD_INPUT_KEYS = ["X", "Y", "Z", "dist_re", "dist_ab", "board_re", "board_ab"]

# Number of input feature channels for the point-cloud GNN
CLOUD_INPUT_DIM = 7

# Number of nearest neighbours for KNN graph construction
CLOUD_K = 16

# Maximum number of points to keep per cloud (subsampled if larger)
MAX_CLOUD_POINTS = 4096
