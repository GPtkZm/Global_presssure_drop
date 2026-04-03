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

Edge types:
  ('point', 'to', 'point') – from vertex_vertex_matrix  (bidirectional)
  ('face',  'to', 'point') – from face_vertex_matrix
  ('point', 'to', 'face')  – reverse of the above

Edge attributes for ('point', 'to', 'point'):
  [dx, dy, dz, Euclidean_distance]  (4-dimensional)

Graph-level label:
  data.y – normalised pressure drop (scalar float32)

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
from torch_geometric.data import Dataset, HeteroData

from src.config import DATA_DIR, LABEL_CSV, CHECKPOINT_DIR
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
        """Return the full path to the topology .npy file for *case_id*."""
        return os.path.join(DATA_DIR, f"{case_id}_topo.npy")

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
          coord_mean, coord_std   – vertex coordinate statistics (scalar each)
          uv_mean, uv_std         – UV-bound statistics (scalar each)
          drop_mean, drop_std     – pressure-drop statistics
        """
        all_coords: List[np.ndarray] = []
        all_uv: List[np.ndarray] = []
        all_drops: List[float] = []

        for _, row in self.labels_df.iterrows():
            case_id = row["ID"]
            path = self._npy_path(case_id)
            if not os.path.exists(path):
                continue
            topo = self._load_npy(case_id)
            all_coords.append(topo["vertex_coordinates"].astype(np.float32).reshape(-1))
            all_uv.append(topo["face_surface_uv_bounds"].astype(np.float32).reshape(-1))
            all_drops.append(float(row["drop"]))

        coord_arr = np.concatenate(all_coords)
        uv_arr = np.concatenate(all_uv)
        drop_arr = np.array(all_drops, dtype=np.float32)

        coord_mean, coord_std = compute_stats(coord_arr)
        uv_mean, uv_std = compute_stats(uv_arr)
        drop_mean, drop_std = compute_stats(drop_arr)

        return {
            "coord_mean": coord_mean,
            "coord_std": coord_std,
            "uv_mean": uv_mean,
            "uv_std": uv_std,
            "drop_mean": drop_mean,
            "drop_std": drop_std,
            "surface_types": self._surface_types,
        }

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def len(self) -> int:
        return len(self.labels_df)

    def get(self, idx: int) -> HeteroData:
        """Build and return a HeteroData object for sample *idx*."""
        row = self.labels_df.iloc[idx]
        case_id = str(row["ID"])
        pressure_drop = float(row["drop"])

        topo = self._load_npy(case_id)
        data = self._build_hetero_data(topo, pressure_drop)

        if self._transform is not None:
            data = self._transform(data)

        return data

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_hetero_data(self, topo: dict, pressure_drop: float) -> HeteroData:
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

        # ---- Graph-level label -------------------------------------------
        drop_norm = (pressure_drop - stats["drop_mean"]) / stats["drop_std"]
        y = torch.tensor([drop_norm], dtype=torch.float32)              # (1,)

        # ---- Assemble HeteroData -----------------------------------------
        data = HeteroData()

        data["point"].x = point_x
        data["face"].x = face_x

        data["point", "to", "point"].edge_index = pp_edge_index
        data["point", "to", "point"].edge_attr = pp_edge_attr

        data["face", "to", "point"].edge_index = fp_edge_index
        data["point", "to", "face"].edge_index = pf_edge_index

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
