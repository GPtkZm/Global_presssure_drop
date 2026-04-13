"""
dataset_fusion.py
-----------------
Three-way fusion dataset that combines:
  1. labels.csv          – case IDs, split, global features (14-dim), pressure-drop label
  2. data/topo/*.npy     – CAD topology graphs (726 files)
  3. data/cloud/*.npy    – single big file with 4,902 point-cloud records

Only cases that exist in **all three** sources are kept.

Each sample returned by __getitem__ is a HeteroData graph that also carries:
  data.cloud_x      – (N_pts, 7)  raw normalised point-cloud features
  data.cloud_y      – (N_pts, 1)  normalised per-point pressure  (Task B label)
  data.cloud_n      – (1,)        number of points in this cloud
  data.y            – (1,)        normalised Δp scalar           (Task A label)
  data.global_features – (1, 14) normalised physics/geometry features

Usage
~~~~~
  ds_train = FusionDataset(split="train")
  norm_stats = ds_train.norm_stats
  ds_test  = FusionDataset(split="test", norm_stats=norm_stats)
"""

import os
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from src.config import (
    CHECKPOINT_DIR,
    CLOUD_INPUT_KEYS,
    CLOUD_NPY_PATH,
    DATA_DIR,
    GLOBAL_FEATURE_COLUMNS,
    LABEL_CSV,
    MAX_CLOUD_POINTS,
)
from src.utils import compute_stats, normalize, save_norm_stats


# ---------------------------------------------------------------------------
# Helper: load and index the big point-cloud npy
# ---------------------------------------------------------------------------

def _load_cloud_index(cloud_npy_path: str) -> Dict[str, dict]:
    """Load the big .npy, deduplicate by case ID, return a dict keyed by ID.

    The File_Path field looks like:
        './te_pressure/DOE213-A4-B2-C2-D3-E16-F5-G3-H5-I6.csv'
    We strip the directory prefix and '.csv' suffix to get the case ID.
    """
    if not os.path.exists(cloud_npy_path):
        return {}

    records = np.load(cloud_npy_path, allow_pickle=True)
    index: Dict[str, dict] = {}
    for rec in records:
        fp = str(rec.get("File_Path", ""))
        # Extract base name without extension
        basename = os.path.splitext(os.path.basename(fp))[0]
        if basename and basename not in index:
            index[basename] = rec
    return index


# ---------------------------------------------------------------------------
# FusionDataset
# ---------------------------------------------------------------------------

class FusionDataset(Dataset):
    """PyTorch dataset that fuses topology graphs, point clouds, and labels.

    Parameters
    ----------
    split : str or None
        One of ``'train'``, ``'test'``, or ``None`` (all cases).
    norm_stats : dict or None
        Pre-computed normalisation statistics from the training split.
        Pass ``None`` for the training set so that stats are computed here.
    transform : callable, optional
        On-the-fly transform applied to each HeteroData object.
    cloud_npy_path : str or None
        Override the path to the big cloud .npy (defaults to CLOUD_NPY_PATH).
    max_points : int
        Maximum number of cloud points to keep per sample.
    """

    def __init__(
        self,
        split: Optional[str] = None,
        norm_stats: Optional[dict] = None,
        transform: Optional[Callable] = None,
        cloud_npy_path: Optional[str] = None,
        max_points: int = MAX_CLOUD_POINTS,
    ):
        self.split = split
        self._transform = transform
        self.max_points = max_points

        cloud_path = cloud_npy_path or CLOUD_NPY_PATH

        # ---- Load label CSV ----------------------------------------------
        labels_df = pd.read_csv(LABEL_CSV)
        if split is not None:
            labels_df = labels_df[labels_df["split"] == split].reset_index(drop=True)

        # ---- Load cloud index (dedup'd) -----------------------------------
        cloud_index = _load_cloud_index(cloud_path)

        # ---- Three-way intersection by case ID ----------------------------
        topo_ids = {
            os.path.splitext(f)[0].replace("_topo", "")
            for f in os.listdir(DATA_DIR)
            if f.endswith("_topo.npy")
        }
        cloud_ids = set(cloud_index.keys())
        label_ids = set(labels_df["ID"].astype(str))

        valid_ids = label_ids & topo_ids & cloud_ids
        labels_df = labels_df[labels_df["ID"].astype(str).isin(valid_ids)].reset_index(
            drop=True
        )

        print(f"  ✅ Fusion valid samples: {len(labels_df)}")
        missing_in_topo = label_ids - topo_ids
        missing_in_cloud = label_ids - cloud_ids
        if missing_in_topo or missing_in_cloud:
            print(f"  ⚠️  Missing in topo: {len(missing_in_topo)}  |  missing in cloud: {len(missing_in_cloud)}")

        self.labels_df = labels_df
        self._cloud_index = cloud_index

        # ---- Collect surface types (needed for face encoder) --------------
        if norm_stats is None:
            self._surface_types = self._collect_surface_types()
        else:
            self._surface_types = norm_stats.get("surface_types", [])
        self._type_to_idx = {t: i for i, t in enumerate(self._surface_types)}

        # ---- Normalisation stats -----------------------------------------
        if norm_stats is None:
            self._norm_stats = self._compute_norm_stats()
            save_norm_stats(
                self._norm_stats,
                os.path.join(CHECKPOINT_DIR, "fusion_norm_stats.json"),
            )
        else:
            self._norm_stats = norm_stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _npy_path(self, case_id: str) -> str:
        exact = os.path.join(DATA_DIR, f"{case_id}_topo.npy")
        if os.path.exists(exact):
            return exact
        prefix = case_id.split("-")[0]
        try:
            for fname in os.listdir(DATA_DIR):
                if fname.startswith(prefix) and fname.endswith("_topo.npy"):
                    return os.path.join(DATA_DIR, fname)
        except OSError:
            pass
        return exact

    def _load_topo(self, case_id: str) -> dict:
        return np.load(self._npy_path(case_id), allow_pickle=True).item()

    def _collect_surface_types(self) -> List[str]:
        type_set: set = set()
        for case_id in self.labels_df["ID"]:
            path = self._npy_path(str(case_id))
            if not os.path.exists(path):
                continue
            topo = np.load(path, allow_pickle=True).item()
            for name in topo.get("face_surface_type_names", []):
                type_set.add(str(name))
        return sorted(type_set)

    def _compute_norm_stats(self) -> dict:
        all_coords: List[np.ndarray] = []
        all_uv: List[np.ndarray] = []
        all_edges: List[np.ndarray] = []
        all_global: List[np.ndarray] = []
        all_drops: List[float] = []
        # cloud: list of (N_pts, 7) arrays for features, (N_pts,) for pressure
        cloud_feat_list: List[np.ndarray] = []
        cloud_pres_list: List[np.ndarray] = []

        avail_global_cols = [c for c in GLOBAL_FEATURE_COLUMNS if c in self.labels_df.columns]

        for _, row in self.labels_df.iterrows():
            case_id = str(row["ID"])
            path = self._npy_path(case_id)
            if not os.path.exists(path):
                continue
            topo = self._load_topo(case_id)
            all_coords.append(topo["vertex_coordinates"].astype(np.float32).reshape(-1))
            all_uv.append(topo["face_surface_uv_bounds"].astype(np.float32).reshape(-1))
            all_drops.append(float(row["drop"]))

            if "edge_parameter_ranges" in topo:
                ep = topo["edge_parameter_ranges"].astype(np.float32)
                if ep.size > 0:
                    all_edges.append(ep.reshape(-1))

            if avail_global_cols:
                vals = np.array([float(row[c]) for c in avail_global_cols], dtype=np.float32)
                all_global.append(vals)

            # Cloud features
            rec = self._cloud_index.get(case_id)
            if rec is not None:
                feat = np.stack(
                    [rec[k].astype(np.float32) for k in CLOUD_INPUT_KEYS], axis=-1
                )  # (N_pts, 7)
                cloud_feat_list.append(feat)
                pres = rec["pressure"].astype(np.float32)
                cloud_pres_list.append(pres)

        # CAD topo stats
        coord_arr = np.concatenate(all_coords)
        uv_arr = np.concatenate(all_uv)
        drop_arr = np.array(all_drops, dtype=np.float32)
        coord_mean, coord_std = compute_stats(coord_arr)
        uv_mean, uv_std = compute_stats(uv_arr)
        drop_mean, drop_std = compute_stats(drop_arr)

        if all_edges:
            edge_mean, edge_std = compute_stats(np.concatenate(all_edges))
        else:
            edge_mean, edge_std = 0.0, 1.0

        n_global = len(GLOBAL_FEATURE_COLUMNS)
        n_avail = len(avail_global_cols)
        if all_global and n_avail > 0:
            g_arr = np.stack(all_global, axis=0)
            g_mean_avail = [float(np.mean(g_arr[:, i])) for i in range(n_avail)]
            g_std_avail = [max(float(np.std(g_arr[:, i])), 1e-8) for i in range(n_avail)]
            c2m = dict(zip(avail_global_cols, g_mean_avail))
            c2s = dict(zip(avail_global_cols, g_std_avail))
            global_mean = [c2m.get(c, 0.0) for c in GLOBAL_FEATURE_COLUMNS]
            global_std = [c2s.get(c, 1.0) for c in GLOBAL_FEATURE_COLUMNS]
        else:
            global_mean = [0.0] * n_global
            global_std = [1.0] * n_global

        # Cloud feature stats (per column, over all training points)
        if cloud_feat_list:
            cloud_all = np.concatenate(cloud_feat_list, axis=0)  # (N_total, 7)
            cloud_feat_mean = [float(np.mean(cloud_all[:, i])) for i in range(cloud_all.shape[1])]
            cloud_feat_std = [
                max(float(np.std(cloud_all[:, i])), 1e-8) for i in range(cloud_all.shape[1])
            ]
        else:
            cloud_feat_mean = [0.0] * len(CLOUD_INPUT_KEYS)
            cloud_feat_std = [1.0] * len(CLOUD_INPUT_KEYS)

        if cloud_pres_list:
            pres_all = np.concatenate(cloud_pres_list)
            cloud_pres_mean, cloud_pres_std = compute_stats(pres_all)
        else:
            cloud_pres_mean, cloud_pres_std = 0.0, 1.0

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
            "cloud_feat_mean": cloud_feat_mean,
            "cloud_feat_std": cloud_feat_std,
            "cloud_pres_mean": cloud_pres_mean,
            "cloud_pres_std": cloud_pres_std,
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
        row = self.labels_df.iloc[idx]
        case_id = str(row["ID"])
        pressure_drop = float(row["drop"])

        topo = self._load_topo(case_id)
        cloud_rec = self._cloud_index[case_id]

        data = self._build_graph(topo, pressure_drop, row, cloud_rec, sample_idx=idx)

        if self._transform is not None:
            data = self._transform(data)

        return data

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self, topo: dict, pressure_drop: float, row, cloud_rec: dict, sample_idx: int = 0) -> HeteroData:
        stats = self._norm_stats

        # ============================================================
        # 1. CAD Topo nodes
        # ============================================================
        coords = topo["vertex_coordinates"].astype(np.float32)
        coords_norm = normalize(coords, stats["coord_mean"], stats["coord_std"])
        point_x = torch.tensor(coords_norm, dtype=torch.float32)

        type_names = topo["face_surface_type_names"]
        uv_bounds = topo["face_surface_uv_bounds"].astype(np.float32)
        uv_norm = normalize(uv_bounds, stats["uv_mean"], stats["uv_std"])
        num_types = len(self._surface_types)
        num_faces = len(type_names)
        one_hot = np.zeros((num_faces, num_types), dtype=np.float32)
        for i, name in enumerate(type_names):
            one_hot[i, self._type_to_idx.get(str(name), 0)] = 1.0
        face_x = torch.tensor(np.concatenate([one_hot, uv_norm], axis=1), dtype=torch.float32)

        if "edge_parameter_ranges" in topo:
            ep = topo["edge_parameter_ranges"].astype(np.float32)
            ep_norm = normalize(ep, stats["edge_mean"], stats["edge_std"])
        else:
            ep_norm = np.zeros((0, 2), dtype=np.float32)
        edge_x = torch.tensor(ep_norm, dtype=torch.float32)

        # ============================================================
        # 2. CAD Topo edges
        # ============================================================
        vv_mat = topo["vertex_vertex_matrix"]
        rows_, cols_ = np.where(vv_mat == 1)
        pp_edge_index = torch.tensor(np.stack([rows_, cols_], axis=0), dtype=torch.long)

        if pp_edge_index.shape[1] > 0:
            src_c = coords_norm[pp_edge_index[0].numpy()]
            dst_c = coords_norm[pp_edge_index[1].numpy()]
            delta = dst_c - src_c
            dist = np.linalg.norm(delta, axis=1, keepdims=True)
            pp_edge_attr = torch.tensor(np.concatenate([delta, dist], axis=1), dtype=torch.float32)
        else:
            pp_edge_attr = torch.zeros((0, 4), dtype=torch.float32)

        fv_mat = topo["face_vertex_matrix"]
        face_ids, vert_ids = np.where(fv_mat == 1)
        fp_edge_index = torch.tensor(np.stack([face_ids, vert_ids], axis=0), dtype=torch.long)
        pf_edge_index = fp_edge_index[[1, 0], :]

        if "vertex_edge_matrix" in topo:
            ve_mat = topo["vertex_edge_matrix"]
            v_ids, e_ids = np.where(ve_mat == 1)
            pe_edge_index = torch.tensor(np.stack([v_ids, e_ids], axis=0), dtype=torch.long)
            ep_edge_index = pe_edge_index[[1, 0], :]
        elif "edge_vertex_pairs" in topo:
            evp = topo["edge_vertex_pairs"].astype(np.int64)
            eidx = np.repeat(np.arange(len(evp)), 2)
            vflat = evp.reshape(-1)
            pe_edge_index = torch.tensor(np.stack([vflat, eidx], axis=0), dtype=torch.long)
            ep_edge_index = pe_edge_index[[1, 0], :]
        else:
            pe_edge_index = torch.zeros((2, 0), dtype=torch.long)
            ep_edge_index = torch.zeros((2, 0), dtype=torch.long)

        if "edge_face_matrix" in topo:
            ef_mat = topo["edge_face_matrix"]
            e_ids2, f_ids2 = np.where(ef_mat == 1)
            ef_edge_index = torch.tensor(np.stack([e_ids2, f_ids2], axis=0), dtype=torch.long)
            fe_edge_index = ef_edge_index[[1, 0], :]
        elif "face_edge_matrix" in topo:
            fe_mat = topo["face_edge_matrix"]
            f_ids3, e_ids3 = np.where(fe_mat == 1)
            fe_edge_index = torch.tensor(np.stack([f_ids3, e_ids3], axis=0), dtype=torch.long)
            ef_edge_index = fe_edge_index[[1, 0], :]
        else:
            ef_edge_index = torch.zeros((2, 0), dtype=torch.long)
            fe_edge_index = torch.zeros((2, 0), dtype=torch.long)

        if "face_face_matrix" in topo:
            ff_mat = topo["face_face_matrix"]
            f_src, f_dst = np.where(ff_mat != 0)
            ff_edge_index = torch.tensor(np.stack([f_src, f_dst], axis=0), dtype=torch.long)
        else:
            ff_edge_index = torch.zeros((2, 0), dtype=torch.long)

        # ============================================================
        # 3. Global features
        # ============================================================
        gm = np.array(stats["global_mean"], dtype=np.float32)
        gs = np.array(stats["global_std"], dtype=np.float32)
        # Missing CSV columns are imputed with the column mean (0.0 before z-score,
        # which becomes 0.0 after z-score), matching the strategy in dataset.py.
        gvals = np.array(
            [float(row[c]) if c in row.index else 0.0 for c in GLOBAL_FEATURE_COLUMNS],
            dtype=np.float32,
        )
        global_norm = (gvals - gm) / gs
        global_features = torch.tensor(global_norm[np.newaxis, :], dtype=torch.float32)

        # ============================================================
        # 4. Label (Δp)
        # ============================================================
        drop_norm = (pressure_drop - stats["drop_mean"]) / stats["drop_std"]
        y = torch.tensor([drop_norm], dtype=torch.float32)

        # ============================================================
        # 5. Point cloud features + pressure label
        # ============================================================
        feat_arr = np.stack(
            [cloud_rec[k].astype(np.float32) for k in CLOUD_INPUT_KEYS], axis=-1
        )  # (N_pts, 7)
        pres_arr = cloud_rec["pressure"].astype(np.float32)  # (N_pts,)

        N_pts = feat_arr.shape[0]
        if N_pts > self.max_points:
            # Use a sample-specific RNG so subsampling is reproducible
            # regardless of worker ordering or epoch shuffling.
            rng = np.random.RandomState(sample_idx)
            chosen = rng.choice(N_pts, self.max_points, replace=False)
            feat_arr = feat_arr[chosen]
            pres_arr = pres_arr[chosen]
            N_pts = self.max_points

        # Normalise cloud features per-column
        cfm = np.array(stats["cloud_feat_mean"], dtype=np.float32)
        cfs = np.array(stats["cloud_feat_std"], dtype=np.float32)
        feat_norm = (feat_arr - cfm) / cfs  # (N_pts, 7)

        # Normalise cloud pressure
        cpm = float(stats["cloud_pres_mean"])
        cps = float(stats["cloud_pres_std"])
        pres_norm = (pres_arr - cpm) / cps  # (N_pts,)

        cloud_x = torch.tensor(feat_norm, dtype=torch.float32)      # (N_pts, 7)
        cloud_y = torch.tensor(pres_norm[:, None], dtype=torch.float32)  # (N_pts, 1)
        cloud_n = torch.tensor([N_pts], dtype=torch.long)            # (1,)

        # ============================================================
        # 6. Assemble HeteroData
        # ============================================================
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

        # Attach cloud tensors as graph-level attributes
        data.cloud_x = cloud_x    # (N_pts, 7)
        data.cloud_y = cloud_y    # (N_pts, 1)
        data.cloud_n = cloud_n    # (1,)

        return data

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def norm_stats(self) -> dict:
        return self._norm_stats

    @property
    def num_face_types(self) -> int:
        return len(self._surface_types)
