"""
models/graphgps.py
------------------
GraphGPS model for global pressure-drop regression.

Architecture overview
~~~~~~~~~~~~~~~~~~~~~
1. **Node Feature Projection** – three linear projections map each heterogeneous
   node type (point / face / edge) into a shared ``hidden_dim`` space, plus a
   learnable type embedding so the model can distinguish node origins.

2. **Unified Homogeneous Graph** – all heterogeneous edges are re-indexed into a
   single flat graph so that every GPS layer can run both a local MPNN *and*
   global self-attention over the full node set.  A learnable edge-type
   embedding (8 types) is used as the edge feature for the MPNN.

3. **GPS Layers** – each layer executes two parallel branches:
     a. *Local MPNN* – GINEConv operating along actual topological edges
        (respects CAD adjacency).
     b. *Global Attention* – multi-head self-attention over **all** nodes in the
        same graph (captures long-range inlet-to-outlet correlations).
   The two branches are summed, followed by residual + LayerNorm + FFN.

4. **Global-feature Projection** – a small MLP maps the 14-dim physics/geometry
   CSV parameters into a ``global_mlp_dim``-dim vector.

5. **Readout & Regression** – mean-pool over the node sequence → concatenate
   with the global-feature embedding → MLP decoder → scalar Δp.

Compatibility
~~~~~~~~~~~~~
Accepts the same ``HeteroData`` objects produced by ``PressureDropDataset``.
The forward() signature is identical to HeteroGNN / TransformerPressureDrop:
    model(data, global_features=None) -> Tensor of shape (B, 1)
"""

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.utils import to_dense_batch

from src.config import (
    GLOBAL_MLP_DIM,
    GPS_ATTN_DROPOUT,
    GPS_DIM_FEEDFORWARD,
    GPS_DROPOUT,
    GPS_HIDDEN_DIM,
    GPS_NHEAD,
    GPS_NUM_LAYERS,
)


# ---------------------------------------------------------------------------
# Helper: edge-type registry (must match dataset.py edge definitions)
# ---------------------------------------------------------------------------
EDGE_TYPE_LIST = [
    ("point", "to", "point"),
    ("face", "to", "point"),
    ("point", "to", "face"),
    ("point", "to", "edge"),
    ("edge", "to", "point"),
    ("edge", "to", "face"),
    ("face", "to", "edge"),
    ("face", "to", "face"),
]
NUM_EDGE_TYPES = len(EDGE_TYPE_LIST)
_EDGE_TYPE_TO_IDX = {et: i for i, et in enumerate(EDGE_TYPE_LIST)}


# ---------------------------------------------------------------------------
# GPSLayer
# ---------------------------------------------------------------------------
class GPSLayer(nn.Module):
    """One GraphGPS layer: Local MPNN ‖ Global Attention → Add → FFN.

    Parameters
    ----------
    hidden_dim : int
        Width of node representations.
    nhead : int
        Number of attention heads for the global branch.
    dim_feedforward : int
        FFN intermediate width.
    dropout : float
        Dropout in FFN and residual paths.
    attn_dropout : float
        Dropout applied to attention weights.
    """

    def __init__(
        self,
        hidden_dim: int,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()

        # ---- Local MPNN branch (GINEConv) ---------------------------------
        gin_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.local_conv = GINEConv(gin_nn, edge_dim=hidden_dim)
        self.norm_local = nn.LayerNorm(hidden_dim)

        # ---- Global Attention branch --------------------------------------
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm_attn = nn.LayerNorm(hidden_dim)

        # ---- Feed-forward network -----------------------------------------
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        sort_idx: torch.Tensor,
        unsort_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h : (N_total, hidden_dim)
            Node features (flat, all graphs concatenated).
            Nodes are in ORIGINAL order (point, face, edge concatenated).
        edge_index : (2, E_total)
            Unified edge index (original order).
        edge_attr : (E_total, hidden_dim)
            Edge features (edge-type embeddings).
        batch : (N_total,)
            Graph membership index for each node (original order).
        sort_idx : (N_total,)
            Permutation that sorts nodes by graph ID (for to_dense_batch).
        unsort_idx : (N_total,)
            Inverse permutation to restore original node order.

        Returns
        -------
        h_out : (N_total, hidden_dim)  — same order as input h.
        """
        residual = h

        # ---- Local MPNN branch --------------------------------------------
        h_local = self.local_conv(h, edge_index, edge_attr)
        h_local = self.dropout(h_local)

        # ---- Global Attention branch --------------------------------------
        # Sort nodes by graph ID so to_dense_batch works correctly.
        h_sorted = h[sort_idx]                      # (N_total, D), grouped by graph
        batch_sorted = batch[sort_idx]               # (N_total,), now 0,0,..,1,1,..,2,..

        h_dense, mask = to_dense_batch(h_sorted, batch_sorted)  # (B, max_N, D), (B, max_N)
        key_padding_mask = ~mask  # True = ignore this position

        h_attn_dense, _ = self.attn(
            h_dense, h_dense, h_dense, key_padding_mask=key_padding_mask
        )

        # Back to flat sorted, then unsort to original order
        h_attn_flat_sorted = h_attn_dense[mask]     # (N_total, D)
        h_attn_flat = h_attn_flat_sorted[unsort_idx] # (N_total, D) — original order
        h_attn_flat = self.dropout(h_attn_flat)

        # ---- Merge: residual + local + global -----------------------------
        h = self.norm_local(residual + h_local)
        h = self.norm_attn(h + h_attn_flat)

        # ---- FFN ----------------------------------------------------------
        h = self.norm_ffn(h + self.ffn(h))
        return h


# ---------------------------------------------------------------------------
# GraphGPSPressureDrop
# ---------------------------------------------------------------------------
class GraphGPSPressureDrop(nn.Module):
    """GraphGPS model for graph-level pressure-drop regression.

    Parameters
    ----------
    point_in_dim : int
        Dimensionality of 'point' node input features (3).
    face_in_dim : int
        Dimensionality of 'face' node input features (num_types + 4).
    edge_in_dim : int
        Dimensionality of 'edge' node input features (2).
    global_feature_dim : int
        Dimensionality of the physics/geometry global features (14).
    hidden_dim : int
        Unified hidden dimension for all nodes and layers.
    num_layers : int
        Number of stacked GPS layers.
    nhead : int
        Number of attention heads per GPS layer.
    dim_feedforward : int
        FFN intermediate width in GPS layers.
    dropout : float
        Dropout probability.
    attn_dropout : float
        Dropout on attention weights.
    global_mlp_dim : int
        Output dimension of the global-feature MLP.
    """

    def __init__(
        self,
        point_in_dim: int,
        face_in_dim: int,
        edge_in_dim: int = 2,
        global_feature_dim: int = 14,
        hidden_dim: int = GPS_HIDDEN_DIM,
        num_layers: int = GPS_NUM_LAYERS,
        nhead: int = GPS_NHEAD,
        dim_feedforward: int = GPS_DIM_FEEDFORWARD,
        dropout: float = GPS_DROPOUT,
        attn_dropout: float = GPS_ATTN_DROPOUT,
        global_mlp_dim: int = GLOBAL_MLP_DIM,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # ---- Node-type projections ----------------------------------------
        self.point_proj = nn.Sequential(
            nn.Linear(point_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.face_proj = nn.Sequential(
            nn.Linear(face_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Type embedding: 0=point, 1=face, 2=edge
        self.type_embed = nn.Embedding(3, hidden_dim)

        # ---- Edge-type embedding ------------------------------------------
        self.edge_type_embed = nn.Embedding(NUM_EDGE_TYPES, hidden_dim)

        # ---- GPS Layers ---------------------------------------------------
        self.gps_layers = nn.ModuleList(
            [
                GPSLayer(
                    hidden_dim=hidden_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # ---- Global-feature MLP ------------------------------------------
        self.global_mlp = nn.Sequential(
            nn.Linear(global_feature_dim, global_mlp_dim),
            nn.ReLU(),
            nn.Linear(global_mlp_dim, global_mlp_dim),
            nn.ReLU(),
            nn.Linear(global_mlp_dim, global_mlp_dim),
        )

        # ---- Fusion Decoder ----------------------------------------------
        fusion_in_dim = hidden_dim + global_mlp_dim
        self.decoder = nn.Sequential(
            nn.Linear(fusion_in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    # ------------------------------------------------------------------
    # Flatten heterogeneous graph into a single homogeneous graph
    # ------------------------------------------------------------------
    def _flatten_hetero(self, data):
        """Convert HeteroData into flat tensors suitable for GPS layers.

        Returns
        -------
        h : (N_total, hidden_dim)
            Projected + type-embedded node features.
        edge_index : (2, E_total)
            Unified edge index with re-mapped node IDs.
        edge_attr : (E_total, hidden_dim)
            Edge-type embeddings.
        batch : (N_total,)
            Graph membership for each node.
        sort_idx : (N_total,)
            Permutation that sorts nodes by graph ID.
        unsort_idx : (N_total,)
            Inverse permutation to restore original node order.
        num_graphs : int
        """
        device = data["point"].x.device

        # -- Project each node type ----------------------------------------
        h_point = self.point_proj(data["point"].x) + self.type_embed(
            torch.zeros(data["point"].x.shape[0], dtype=torch.long, device=device)
        )
        h_face = self.face_proj(data["face"].x) + self.type_embed(
            torch.ones(data["face"].x.shape[0], dtype=torch.long, device=device)
        )

        n_edge_nodes = data["edge"].x.shape[0]
        if n_edge_nodes > 0:
            h_edge = self.edge_proj(data["edge"].x) + self.type_embed(
                torch.full((n_edge_nodes,), 2, dtype=torch.long, device=device)
            )
        else:
            h_edge = torch.zeros(0, self.hidden_dim, device=device)

        # -- Concatenate all node features ---------------------------------
        h = torch.cat([h_point, h_face, h_edge], dim=0)  # (N_total, D)

        # -- Build unified batch vector ------------------------------------
        n_point = h_point.shape[0]
        n_face = h_face.shape[0]
        n_edge = h_edge.shape[0]

        batch_point = data["point"].batch
        batch_face = data["face"].batch
        if n_edge > 0:
            batch_edge = data["edge"].batch
        else:
            batch_edge = torch.zeros(0, dtype=torch.long, device=device)
        batch = torch.cat([batch_point, batch_face, batch_edge], dim=0)

        num_graphs = int(batch_point.max().item()) + 1

        # -- Compute sort/unsort indices for to_dense_batch ----------------
        # to_dense_batch requires batch vector to be sorted (non-decreasing).
        # After cat([point_batch, face_batch, edge_batch]) the batch vector
        # is NOT sorted (e.g. [0,0,1,1, 0,0,1,1, 0,1]).
        # We precompute a stable sort and its inverse once.
        sort_idx = torch.argsort(batch, stable=True)
        unsort_idx = torch.empty_like(sort_idx)
        unsort_idx[sort_idx] = torch.arange(sort_idx.shape[0], device=device)

        # -- Compute node-ID offsets per type ------------------------------
        offset_face = n_point
        offset_edge = n_point + n_face

        _TYPE_OFFSET = {
            "point": 0,
            "face": offset_face,
            "edge": offset_edge,
        }

        # -- Re-index all heterogeneous edges into the flat ID space -------
        all_edge_index = []
        all_edge_type_ids = []

        for et in EDGE_TYPE_LIST:
            src_type, _, dst_type = et
            key = (src_type, "to", dst_type)
            try:
                ei = data[key].edge_index  # (2, E_k)
            except (KeyError, AttributeError):
                continue
            if ei.shape[1] == 0:
                continue

            src_offset = _TYPE_OFFSET[src_type]
            dst_offset = _TYPE_OFFSET[dst_type]

            remapped = ei.clone()
            remapped[0] += src_offset
            remapped[1] += dst_offset
            all_edge_index.append(remapped)

            type_idx = _EDGE_TYPE_TO_IDX[et]
            all_edge_type_ids.append(
                torch.full((ei.shape[1],), type_idx, dtype=torch.long, device=device)
            )

        if all_edge_index:
            edge_index = torch.cat(all_edge_index, dim=1)  # (2, E_total)
            edge_type_ids = torch.cat(all_edge_type_ids)    # (E_total,)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            edge_type_ids = torch.zeros(0, dtype=torch.long, device=device)

        edge_attr = self.edge_type_embed(edge_type_ids)  # (E_total, hidden_dim)

        return h, edge_index, edge_attr, batch, sort_idx, unsort_idx, num_graphs

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, data, global_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        data : HeteroData
            Batched heterogeneous graph from ``PressureDropDataset``.
        global_features : torch.Tensor, optional
            Pre-extracted global features of shape (B, global_feature_dim).
            If None, they are read from ``data.global_features``.

        Returns
        -------
        torch.Tensor
            Shape (B, 1) – normalised pressure-drop prediction per graph.
        """
        if global_features is None:
            global_features = data.global_features  # (B, 14)

        # ---- Flatten heterogeneous → homogeneous --------------------------
        h, edge_index, edge_attr, batch, sort_idx, unsort_idx, num_graphs = (
            self._flatten_hetero(data)
        )

        # ---- GPS Layers ---------------------------------------------------
        for layer in self.gps_layers:
            h = layer(h, edge_index, edge_attr, batch, sort_idx, unsort_idx)

        # ---- Readout: mean pool over all nodes ----------------------------
        graph_emb = global_mean_pool(h, batch, size=num_graphs)  # (B, hidden_dim)

        # ---- Global-feature MLP ------------------------------------------
        g_physics = self.global_mlp(global_features)  # (B, global_mlp_dim)

        # ---- Fusion + Decode ---------------------------------------------
        g_fused = torch.cat([graph_emb, g_physics], dim=-1)
        out = self.decoder(g_fused)  # (B, 1)
        return out
