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
# Model hyperparameters
# ---------------------------------------------------------------------------
HIDDEN_DIM = 128       # Width of all hidden layers
NUM_LAYERS = 6         # Number of heterogeneous message-passing layers
DROPOUT = 0.1          # Dropout probability in the decoder MLP

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
LR = 1e-3              # Initial learning rate for Adam
EPOCHS = 300           # Maximum number of training epochs
BATCH_SIZE = 16        # Number of graphs per mini-batch
PATIENCE = 30          # Early-stopping patience (epochs without val improvement)
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
