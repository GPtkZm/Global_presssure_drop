# geometry_generator

Procedural pipe/channel network geometry generator for 2-D grids.

Generates random pipe-network topologies with **guaranteed inlet→outlet
connectivity**, configurable branching, and high-quality matplotlib
visualisations.  Designed as a standalone tool that does **not** interfere
with the existing `src/` machine-learning pipeline.

---

## Quick start

```bash
# Generate 10 networks using the default config
python -m geometry_generator

# Specify a config file, override sample count and seed
python -m geometry_generator \
    --config geometry_generator/config.yaml \
    --num_samples 50 \
    --seed 42

# Save outputs to a custom directory
python -m geometry_generator --output_dir /tmp/my_networks --num_samples 5
```

After running, you will find:

```
output/
├── network_0001/
│   ├── grid.npy       ← 2-D integer array  (Ny × Nx)
│   ├── graph.json     ← node/edge graph
│   └── preview.png    ← visualisation image
├── network_0002/
│   └── …
└── summary.png        ← thumbnail montage of all networks
```

---

## Configuration (`config.yaml`)

All generation parameters live in `geometry_generator/config.yaml`.
The file is heavily commented; here is a quick reference:

| Key | Default | Description |
|-----|---------|-------------|
| `grid.Nx` / `grid.Ny` | 60 / 30 | Grid width / height in cells |
| `inlet.wall` + `inlet.pos` | left / 5 | Inlet port location |
| `outlet.wall` + `outlet.pos` | right / 25 | Outlet port location |
| `main_path.p_perturb` | 0.30 | Probability of a random detour step |
| `main_path.bias_toward_outlet` | 0.70 | Directional bias toward outlet |
| `branches.p_branch` | 0.20 | Probability of branching at a main-path cell |
| `branches.p_split` | 0.10 | Probability of recursively splitting a branch |
| `branches.max_depth` | 2 | Maximum branch recursion depth |
| `branches.max_length` | 12 | Maximum branch segment length (cells) |
| `branches.min_spacing` | 2 | Minimum distance between separate branches |
| `loops.p_loop` | 0.05 | Probability of a branch reconnecting (loop) |
| `pipe_width` | 1 | Visual pipe thickness in cells |
| `remove_dead_ends` | false | Remove degree-1 dead-end cells |
| `num_samples` | 10 | Number of topologies to generate |
| `seed` | null | Global seed (null = non-deterministic) |
| `output.dir` | output | Root output directory |
| `visualization.dpi` | 120 | PNG resolution |

---

## Output files

### `grid.npy`
NumPy integer array of shape `(Ny, Nx)`:
- `0` → empty cell
- `1` → main-path pipe cell
- `2` → branch pipe cell

Load with:
```python
import numpy as np
grid = np.load("output/network_0001/grid.npy")
```

### `graph.json`
JSON dictionary:
```json
{
  "nodes":      [[x0,y0], [x1,y1], …],
  "edges":      [[src, dst], …],
  "node_type":  [1, 2, …],
  "inlet_idx":  0,
  "outlet_idx": 42
}
```
- `node_type` mirrors the grid values (1 = main, 2 = branch).
- `edges` are undirected; each pair stored once (`src < dst`).

### `preview.png`
Colour-coded raster image:
- **Blue** cells = main path
- **Orange** cells = branches
- **Green triangle** = inlet
- **Red square** = outlet

### `summary.png`
Thumbnail montage of all generated networks for quick comparison.

---

## Module structure

```
geometry_generator/
├── __init__.py         public API
├── __main__.py         python -m geometry_generator
├── generate.py         CLI entry point & batch runner
├── config.yaml         default configuration
├── config_loader.py    YAML loading & validation
├── network.py          main-path BFS, branch growth, connectivity
├── graph.py            grid → node/edge graph
├── visualize.py        matplotlib PNG generation
└── README.md           this file
```

---

## Design notes

- **Guaranteed connectivity**: the main path uses a biased random walk that
  falls back to BFS if it gets stuck, guaranteeing the outlet is always
  reached.  After branch generation, a second BFS flood-fill removes any
  disconnected cells.
- **Branch spacing**: the Chebyshev-distance proximity check compares
  candidate cells only against cells from *other* branches, so a branch can
  grow along its own path without being rejected.
- **Reproducibility**: supply an integer `seed` in the config; each sample
  uses `seed + sample_index` so the whole batch is reproducible but each
  network is distinct.
