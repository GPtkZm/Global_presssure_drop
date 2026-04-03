"""
dataset.py
----------
PyTorch Geometric Dataset that converts each topology `.npy` file into a
`HeteroData` graph object suitable for heterogeneous GNN training.

Graph schema
~~~~~~~~~~~~
Node types:
  'point'  – CAD vertices  → features: normalised (x, y, z)
  'face'   – CAD surfaces  → features: one-hot(surface_type) ‖ normalised UV bounds
  'edge'   – CAD edges     → features: normalised edge_parameter_ranges (start, end)

Edge types:
  ('point', 'to', 'point') – from vertex_vertex_matrix  (bidirectional)
  ('face',  'to', 'point') – from face_vertex_matrix
  ('point', 'to', 'face')  – reverse of the above
  ('point', 'to', 'edge')  – from vertex_edge_matrix (or edge_vertex_pairs)
  ('edge',  'to', 'point') – reverse of the above
  ('edge',  'to', 'face')  – from edge_face_matrix
  ('face',  'to', 'edge')  – from face_edge_matrix
  ('face',  'to', 'face')  – from face_face_matrix

Edge attributes for ('point', 'to', 'point'):
  [dx, dy, dz, Euclidean_distance]  (4-dimensional)

Graph-level features:
  data.global_features – normalised 14-dim physics/geometry parameters (shape (1, 14))
  data.y               – normalised pressure drop (scalar float32, shape (1,))

Usage
~~~~~
  dataset = PressureDropDataset(root='data')
  print(dataset[0])   # HeteroData object for the first case
"""

import os
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from src.config import (
    CHECKPOINT_DIR,
    DATA_DIR,
    GLOBAL_FEATURE_COLUMNS,
    LABEL_CSV,
)
from src.utils import compute_stats, normalize, save_norm_stats


class PressureDropDataset(Dataset):
    """Dataset of heterogeneous CAD topology graphs for pressure-drop prediction.

    Parameters
    ----------
    root : str
        Root directory of the project.  `data/topo/` and `data/labels.csv`
        are expected relative to this path.
    split : str or None
        One of ``'train'``, ``'val'``, ``'test'``, or ``None`` (all cases).
    transform : callable, optional
        On-the-fly transform applied after loading a sample.
    norm_stats : dict or None
        Pre-computed normalisation statistics.  When ``None`` the stats are
        computed from the training split and saved to ``checkpoints/norm_stats.json``.
    """

    def __init__(
        self,
        root: str = ".",
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        norm_stats: Optional[dict] = None,
    ):
        # Do NOT call super().__init__() here because the PyG base class
        # would trigger its own download/process machinery, which we bypass.
        self.root = root
        self.split = split
        self._transform = transform

        # Load the label CSV
        self.labels_df = pd.read_csv(LABEL_CSV)

        # Filter by split if requested
        if split is not None:
            self.labels_df = self.labels_df[
                self.labels_df["split"] == split
            ].reset_index(drop=True)

        # Compute or reuse normalisation statistics
        if norm_stats is None:
            # First pass: collect surface types from this split (training set)
            self._surface_types: List[str] = self._collect_surface_types()
            self._type_to_idx = {t: i for i, t in enumerate(self._surface_types)}
            # _compute_norm_stats includes "surface_types" in its return value
            self._norm_stats = self._compute_norm_stats()
            save_norm_stats(
                self._norm_stats,
                os.path.join(CHECKPOINT_DIR, "norm_stats.json"),
            )
        else:
            # Reuse the vocabulary and stats from the training set so that
            # val/test samples are encoded with an identical feature space.
            self._norm_stats = norm_stats
            self._surface_types = norm_stats.get("surface_types", [])
            self._type_to_idx = {t: i for i, t in enumerate(self._surface_types)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _npy_path(self, case_id: str) -> str:
        """Return the full path to the topology .npy file for *case_id*.

        First tries an exact match ``{case_id}_topo.npy``.  If not found,
        falls back to a prefix match on the first ``-``-separated segment
        (DOE prefix), which supports filenames where the full ID suffix
        has been shortened.
        """
        exact = os.path.join(DATA_DIR, f"{case_id}_topo.npy")
        if os.path.exists(exact):
            return exact

        # Fuzzy match: try matching on the DOE prefix (segment before first '-')
        prefix = case_id.split("-")[0]
        try:
            for fname in os.listdir(DATA_DIR):
                if fname.startswith(prefix) and fname.endswith("_topo.npy"):
                    return os.path.join(DATA_DIR, fname)
        except OSError:
            pass

        return exact  # Return exact path even if missing (caller handles error)

    def _load_npy(self, case_id: str) -> dict:
        """Load and return the topology dictionary for *case_id*."""
        path = self._npy_path(case_id)
        return np.load(path, allow_pickle=True).item()

    def _collect_surface_types(self) -> List[str]:
        """Scan every .npy file in the current split to build a sorted list of
        unique surface-type names.  The list order is deterministic (sorted)."""
        type_set = set()
        for case_id in self.labels_df["ID"]:
            path = self._npy_path(case_id)
            if not os.path.exists(path):
                continue
            topo = np.load(path, allow_pickle=True).item()
            names = topo.get("face_surface_type_names", np.array([]))
            for name in names:
                type_set.add(str(name))
        return sorted(type_set)

    def _compute_norm_stats(self) -> dict:
        """Compute global normalisation statistics over the current split.

        Returns a dict with the following keys:
          coord_mean, coord_std     – vertex coordinate statistics (scalar each)
          uv_mean, uv_std           – UV-bound statistics (scalar each)
          edge_mean, edge_std       – edge parameter-range statistics (scalar each)
          global_mean, global_std   – per-column stats for the 14 CSV physics
                                       parameters (each a list of 14 floats)
          drop_mean, drop_std       – pressure-drop statistics
          surface_types             – sorted list of unique surface-type name strings
        """
        all_coords: List[np.ndarray] = []
        all_uv: List[np.ndarray] = []
        all_edges: List[np.ndarray] = []
        all_global: List[np.ndarray] = []
        all_drops: List[float] = []

        # Determine which global feature columns actually exist in the CSV
        available_global_cols = [
            c for c in GLOBAL_FEATURE_COLUMNS if c in self.labels_df.columns
        ]

        for _, row in self.labels_df.iterrows():
            case_id = row["ID"]
            path = self._npy_path(case_id)
            if not os.path.exists(path):
                continue
            topo = self._load_npy(case_id)
            all_coords.append(topo["vertex_coordinates"].astype(np.float32).reshape(-1))
            all_uv.append(topo["face_surface_uv_bounds"].astype(np.float32).reshape(-1))
            all_drops.append(float(row["drop"]))

            # Edge parameter ranges
            if "edge_parameter_ranges" in topo:
                ep = topo["edge_parameter_ranges"].astype(np.float32)
                if ep.size > 0:
                    all_edges.append(ep.reshape(-1))

            # CSV global/physics features
            if available_global_cols:
                vals = np.array([float(row[c]) for c in available_global_cols], dtype=np.float32)
                all_global.append(vals)

        coord_arr = np.concatenate(all_coords)
        uv_arr = np.concatenate(all_uv)
        drop_arr = np.array(all_drops, dtype=np.float32)

        coord_mean, coord_std = compute_stats(coord_arr)
        uv_mean, uv_std = compute_stats(uv_arr)
        drop_mean, drop_std = compute_stats(drop_arr)

        # Edge parameter stats (scalar over all values)
        if all_edges:
            edge_arr = np.concatenate(all_edges)
            edge_mean, edge_std = compute_stats(edge_arr)
        else:
            edge_mean, edge_std = 0.0, 1.0

        # Per-column stats for global/physics features
        n_global = len(GLOBAL_FEATURE_COLUMNS)
        n_avail = len(available_global_cols)
        if all_global and n_avail > 0:
            global_arr = np.stack(all_global, axis=0)  # (N, n_avail)
            g_mean_avail = [float(np.mean(global_arr[:, i])) for i in range(n_avail)]
            g_std_avail = [
                max(float(np.std(global_arr[:, i])), 1e-8) for i in range(n_avail)
            ]
            # Build full-length lists (fill absent columns with 0/1)
            col_to_mean = dict(zip(available_global_cols, g_mean_avail))
            col_to_std = dict(zip(available_global_cols, g_std_avail))
            global_mean = [col_to_mean.get(c, 0.0) for c in GLOBAL_FEATURE_COLUMNS]
            global_std = [col_to_std.get(c, 1.0) for c in GLOBAL_FEATURE_COLUMNS]
        else:
            global_mean = [0.0] * n_global
            global_std = [1.0] * n_global

        return {
            "coord_mean": coord_mean,
            "coord_std": coord_std,
            "uv_mean": uv_mean,
            "uv_std": uv_std,
            "edge_mean": edge_mean,
            "edge_std": edge_std,
            "global_mean": global_mean,
            "global_std": global_std,
            "drop_mean": drop_mean,
            "drop_std": drop_std,
            "surface_types": self._surface_types,
        }

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> HeteroData:
        return self.get(idx)

    def len(self) -> int:
        return len(self.labels_df)

    def get(self, idx: int) -> HeteroData:
        """Build and return a HeteroData object for sample *idx*."""
        row = self.labels_df.iloc[idx]
        case_id = str(row["ID"])
        pressure_drop = float(row["drop"])

        topo = self._load_npy(case_id)
        data = self._build_hetero_data(topo, pressure_drop, row)

        if self._transform is not None:
            data = self._transform(data)

        return data

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_hetero_data(self, topo: dict, pressure_drop: float, row) -> HeteroData:
        """Convert a topology dictionary to a HeteroData graph."""
        stats = self._norm_stats

        # ---- Node features -----------------------------------------------

        # 'point' nodes: normalised vertex coordinates
        coords = topo["vertex_coordinates"].astype(np.float32)          # (N_v, 3)
        coords_norm = normalize(coords, stats["coord_mean"], stats["coord_std"])
        point_x = torch.tensor(coords_norm, dtype=torch.float32)        # (N_v, 3)

        # 'face' nodes: one-hot(surface_type) ‖ normalised UV bounds
        type_names = topo["face_surface_type_names"]                     # (N_f,)
        uv_bounds = topo["face_surface_uv_bounds"].astype(np.float32)   # (N_f, 4)
        uv_norm = normalize(uv_bounds, stats["uv_mean"], stats["uv_std"])

        num_types = len(self._surface_types)
        num_faces = len(type_names)
        one_hot = np.zeros((num_faces, num_types), dtype=np.float32)
        for i, name in enumerate(type_names):
            idx = self._type_to_idx.get(str(name), 0)
            one_hot[i, idx] = 1.0

        face_x = torch.tensor(
            np.concatenate([one_hot, uv_norm], axis=1), dtype=torch.float32
        )                                                                # (N_f, num_types+4)

        # 'edge' nodes: normalised edge parameter ranges
        if "edge_parameter_ranges" in topo:
            ep = topo["edge_parameter_ranges"].astype(np.float32)       # (N_e, 2)
            ep_norm = normalize(ep, stats["edge_mean"], stats["edge_std"])
        else:
            ep_norm = np.zeros((0, 2), dtype=np.float32)
        edge_x = torch.tensor(ep_norm, dtype=torch.float32)             # (N_e, 2)

        # ---- Edge indices ------------------------------------------------

        # (point, to, point): from vertex_vertex_matrix — already symmetric,
        # np.where gives us both directions in one call.
        vv_mat = topo["vertex_vertex_matrix"]
        rows, cols = np.where(vv_mat == 1)
        pp_edge_index = torch.tensor(
            np.stack([rows, cols], axis=0), dtype=torch.long
        )                                                                # (2, E_pp)

        # Edge attributes for point-to-point: [dx, dy, dz, dist]
        if pp_edge_index.shape[1] > 0:
            src_coords = coords_norm[pp_edge_index[0].numpy()]
            dst_coords = coords_norm[pp_edge_index[1].numpy()]
            delta = dst_coords - src_coords                              # (E_pp, 3)
            dist = np.linalg.norm(delta, axis=1, keepdims=True)         # (E_pp, 1)
            pp_edge_attr = torch.tensor(
                np.concatenate([delta, dist], axis=1), dtype=torch.float32
            )                                                            # (E_pp, 4)
        else:
            pp_edge_attr = torch.zeros((0, 4), dtype=torch.float32)

        # (face, to, point): from face_vertex_matrix
        fv_mat = topo["face_vertex_matrix"]
        face_ids, vert_ids = np.where(fv_mat == 1)
        fp_edge_index = torch.tensor(
            np.stack([face_ids, vert_ids], axis=0), dtype=torch.long
        )                                                                # (2, E_fp)

        # (point, to, face): reverse of the above
        pf_edge_index = fp_edge_index[[1, 0], :]                        # (2, E_fp)

        # (point, to, edge) and (edge, to, point)
        # Prefer vertex_edge_matrix; fall back to edge_vertex_pairs
        if "vertex_edge_matrix" in topo:
            ve_mat = topo["vertex_edge_matrix"]                         # (N_v, N_e)
            v_ids, e_ids = np.where(ve_mat == 1)
            pe_edge_index = torch.tensor(
                np.stack([v_ids, e_ids], axis=0), dtype=torch.long
            )                                                            # (2, E_pe)
            ep_edge_index = pe_edge_index[[1, 0], :]                    # (2, E_pe)
        elif "edge_vertex_pairs" in topo:
            evp = topo["edge_vertex_pairs"].astype(np.int64)            # (N_e, 2)
            # Each edge has 2 endpoints; repeat edge index for each endpoint
            edge_ids_per_vertex = np.repeat(np.arange(len(evp)), 2)    # [0,0,1,1,...]
            v_flat = evp.reshape(-1)                                     # vertex ids
            pe_edge_index = torch.tensor(
                np.stack([v_flat, edge_ids_per_vertex], axis=0), dtype=torch.long
            )                                                            # (2, 2*N_e)
            ep_edge_index = pe_edge_index[[1, 0], :]                    # (2, 2*N_e)
        else:
            pe_edge_index = torch.zeros((2, 0), dtype=torch.long)
            ep_edge_index = torch.zeros((2, 0), dtype=torch.long)

        # (edge, to, face) and (face, to, edge)
        # Prefer edge_face_matrix; fall back to face_edge_matrix
        if "edge_face_matrix" in topo:
            ef_mat = topo["edge_face_matrix"]                           # (N_e, N_f)
            e_ids2, f_ids2 = np.where(ef_mat == 1)
            ef_edge_index = torch.tensor(
                np.stack([e_ids2, f_ids2], axis=0), dtype=torch.long
            )                                                            # (2, E_ef)
            fe_edge_index = ef_edge_index[[1, 0], :]                    # (2, E_ef)
        elif "face_edge_matrix" in topo:
            fe_mat = topo["face_edge_matrix"]                           # (N_f, N_e)
            f_ids3, e_ids3 = np.where(fe_mat == 1)
            fe_edge_index = torch.tensor(
                np.stack([f_ids3, e_ids3], axis=0), dtype=torch.long
            )                                                            # (2, E_fe)
            ef_edge_index = fe_edge_index[[1, 0], :]                    # (2, E_fe)
        else:
            ef_edge_index = torch.zeros((2, 0), dtype=torch.long)
            fe_edge_index = torch.zeros((2, 0), dtype=torch.long)

        # (face, to, face): from face_face_matrix — add edges in both directions
        if "face_face_matrix" in topo:
            ff_mat = topo["face_face_matrix"]                           # (N_f, N_f)
            f_src, f_dst = np.where(ff_mat != 0)
            ff_edge_index = torch.tensor(
                np.stack([f_src, f_dst], axis=0), dtype=torch.long
            )                                                            # (2, E_ff)
        else:
            ff_edge_index = torch.zeros((2, 0), dtype=torch.long)

        # ---- Global (graph-level) physics/geometry features -------------
        global_mean = np.array(stats["global_mean"], dtype=np.float32)  # (14,)
        global_std = np.array(stats["global_std"], dtype=np.float32)    # (14,)
        global_vals = np.array(
            [float(row[c]) if c in row.index else 0.0 for c in GLOBAL_FEATURE_COLUMNS],
            dtype=np.float32,
        )                                                                # (14,)
        global_norm = (global_vals - global_mean) / global_std           # (14,)
        # Shape (1, 14) so PyG DataLoader stacks batches as (B, 14)
        global_features = torch.tensor(global_norm[np.newaxis, :], dtype=torch.float32)

        # ---- Graph-level label -------------------------------------------
        drop_norm = (pressure_drop - stats["drop_mean"]) / stats["drop_std"]
        y = torch.tensor([drop_norm], dtype=torch.float32)              # (1,)

        # ---- Assemble HeteroData -----------------------------------------
        data = HeteroData()

        data["point"].x = point_x
        data["face"].x = face_x
        data["edge"].x = edge_x

        data["point", "to", "point"].edge_index = pp_edge_index
        data["point", "to", "point"].edge_attr = pp_edge_attr

        data["face", "to", "point"].edge_index = fp_edge_index
        data["point", "to", "face"].edge_index = pf_edge_index

        data["point", "to", "edge"].edge_index = pe_edge_index
        data["edge", "to", "point"].edge_index = ep_edge_index

        data["edge", "to", "face"].edge_index = ef_edge_index
        data["face", "to", "edge"].edge_index = fe_edge_index

        data["face", "to", "face"].edge_index = ff_edge_index

        data.global_features = global_features
        data.y = y

        return data

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_face_types(self) -> int:
        """Number of unique face-surface types across the dataset."""
        return len(self._surface_types)

    @property
    def norm_stats(self) -> dict:
        """Normalisation statistics dictionary."""
        return self._norm_stats
