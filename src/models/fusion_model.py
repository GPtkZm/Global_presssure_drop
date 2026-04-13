"""
models/fusion_model.py
----------------------
Three-way fusion model for pressure-drop and node-level pressure prediction.

Architecture
~~~~~~~~~~~~
                   ┌──────────────────────┐
 Point cloud  ──►  │  PointCloudGNN       │ → h_cloud (N, 128)
 (XYZ + feats)     │  KNN + 6×GINEConv    │
                   └──────────────────────┘
                            │
                   ┌────────▼─────────────┐
 CAD topology ──►  │  CADTopoEncoder      │ → topo_emb (B, 192)
 (HeteroGraph)     │  2-layer HeteroConv  │
                   └──────────────────────┘
                            │
                   ┌────────▼─────────────┐
 Global feats ──►  │  GlobalFeatureMLP    │ → global_emb (B, 64)
 (14-dim CSV)      │  3-layer MLP         │
                   └──────────────────────┘
                            │
                   context = cat(topo_emb, global_emb)  → (B, 256)

Task A (drop):
    cloud_pool = mean_pool(h_cloud)        → (B, 128)
    graph_input = cat(cloud_pool, context) → (B, 384)
    MLP 384→256→128→1                      → (B, 1)

Task B (node_pressure):
    ctx_proj(context)[cloud_batch]         → (N, 128)
    node_input = cat(h_cloud, ctx_broad)   → (N, 256)
    MLP 256→128→64→1                       → (N, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv,
    HeteroConv,
    SAGEConv,
    global_mean_pool,
    knn_graph,
)

from src.config import (
    CLOUD_INPUT_DIM,
    CLOUD_K,
    DROPOUT,
    GLOBAL_FEATURE_COLUMNS,
    GLOBAL_MLP_DIM,
)


# ---------------------------------------------------------------------------
# Shared MLP helper (same as in heterognn.py)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple multi-layer perceptron with ReLU activations and optional dropout."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_layers >= 1
        layers = []
        cur = in_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(cur, hidden_dim), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            cur = hidden_dim
        layers.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Point Cloud GNN (6-layer GINEConv with residuals + LayerNorm)
# ---------------------------------------------------------------------------

class PointCloudGNN(nn.Module):
    """KNN-based GNN over raw point clouds.

    Input: (N, cloud_in_dim) – normalised XYZ + 4 extra features per point.
    Output: (N, hidden_dim)  – per-point embeddings.
    """

    def __init__(
        self,
        cloud_in_dim: int = CLOUD_INPUT_DIM,
        hidden_dim: int = 128,
        num_layers: int = 6,
        k: int = CLOUD_K,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.k = k
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Project input node features to hidden_dim
        self.node_encoder = MLP(cloud_in_dim, hidden_dim, hidden_dim, num_layers=2)

        # Edge features are [dx, dy, dz, euclidean_dist] → 4-dim
        self.edge_encoder = MLP(4, hidden_dim, hidden_dim, num_layers=2)

        # 6 × GINEConv layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            gin_nn = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=2)
            self.convs.append(GINEConv(gin_nn, edge_dim=hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        cloud_x: torch.Tensor,
        cloud_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        cloud_x : (N, cloud_in_dim)  – stacked node features for all points in batch
        cloud_batch : (N,)           – graph index for each node

        Returns
        -------
        h : (N, hidden_dim)
        """
        # Build KNN graph from raw XYZ (first 3 dims)
        xyz = cloud_x[:, :3]
        edge_index = knn_graph(xyz, k=self.k, batch=cloud_batch, loop=False)

        # Edge features: [dx, dy, dz, dist]
        src, dst = edge_index
        delta = xyz[dst] - xyz[src]              # (E, 3)
        dist = delta.norm(dim=-1, keepdim=True)  # (E, 1)
        raw_edge_attr = torch.cat([delta, dist], dim=-1)  # (E, 4)
        edge_attr = self.edge_encoder(raw_edge_attr)       # (E, hidden_dim)

        h = self.node_encoder(cloud_x)  # (N, hidden_dim)

        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index, edge_attr)
            h_new = self.dropout(h_new)
            h = self.norms[i](h + h_new)

        return h  # (N, hidden_dim)


# ---------------------------------------------------------------------------
# CAD Topology Encoder (2-layer HeteroConv, 8 edge types, hidden_dim=64)
# ---------------------------------------------------------------------------

class CADTopoEncoder(nn.Module):
    """Lightweight 2-layer heterogeneous GNN for CAD topology graphs.

    Output: concatenation of mean-pooled point / edge / face embeddings
            → (B, 3 * hidden_dim).
    """

    def __init__(
        self,
        point_in_dim: int,
        face_in_dim: int,
        edge_in_dim: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.point_encoder = MLP(point_in_dim, hidden_dim, hidden_dim, num_layers=2)
        self.edge_encoder = MLP(edge_in_dim, hidden_dim, hidden_dim, num_layers=2)
        self.face_encoder = MLP(face_in_dim, hidden_dim, hidden_dim, num_layers=2)

        self.convs = nn.ModuleList()
        self.point_norms = nn.ModuleList()
        self.edge_norms = nn.ModuleList()
        self.face_norms = nn.ModuleList()

        for _ in range(2):
            conv = HeteroConv(
                {
                    ("point", "to", "point"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    ("face", "to", "point"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    ("edge", "to", "point"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    ("point", "to", "face"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    ("edge", "to", "face"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    ("face", "to", "face"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    ("point", "to", "edge"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    ("face", "to", "edge"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                },
                aggr="sum",
            )
            self.convs.append(conv)
            self.point_norms.append(nn.LayerNorm(hidden_dim))
            self.edge_norms.append(nn.LayerNorm(hidden_dim))
            self.face_norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, data) -> torch.Tensor:
        """
        Parameters
        ----------
        data : HeteroData batch

        Returns
        -------
        topo_emb : (B, 3 * hidden_dim)
        """
        h_point = self.point_encoder(data["point"].x)
        h_edge = self.edge_encoder(data["edge"].x)
        h_face = self.face_encoder(data["face"].x)

        for i, conv in enumerate(self.convs):
            x_dict = {"point": h_point, "edge": h_edge, "face": h_face}
            ei_dict = {
                ("point", "to", "point"): data["point", "to", "point"].edge_index,
                ("face", "to", "point"): data["face", "to", "point"].edge_index,
                ("edge", "to", "point"): data["edge", "to", "point"].edge_index,
                ("point", "to", "face"): data["point", "to", "face"].edge_index,
                ("edge", "to", "face"): data["edge", "to", "face"].edge_index,
                ("face", "to", "face"): data["face", "to", "face"].edge_index,
                ("point", "to", "edge"): data["point", "to", "edge"].edge_index,
                ("face", "to", "edge"): data["face", "to", "edge"].edge_index,
            }
            out = conv(x_dict, ei_dict)
            h_point = self.point_norms[i](h_point + F.relu(self.dropout(out["point"])))
            h_face = self.face_norms[i](h_face + F.relu(self.dropout(out["face"])))
            if h_edge.shape[0] > 0:
                # out['edge'] may be absent if no edge-type messages were aggregated;
                # fall back to zeros (identity residual) to avoid a silent skip.
                h_edge = self.edge_norms[i](
                    h_edge + F.relu(self.dropout(out.get("edge", torch.zeros_like(h_edge))))
                )

        num_graphs = int(data["point"].batch.max().item()) + 1
        g_point = global_mean_pool(h_point, data["point"].batch, size=num_graphs)
        g_face = global_mean_pool(h_face, data["face"].batch, size=num_graphs)
        if h_edge.shape[0] > 0:
            g_edge = global_mean_pool(h_edge, data["edge"].batch, size=num_graphs)
        else:
            g_edge = torch.zeros(num_graphs, self.hidden_dim, device=h_point.device)

        return torch.cat([g_point, g_edge, g_face], dim=-1)  # (B, 3*hidden_dim)


# ---------------------------------------------------------------------------
# FusionModel
# ---------------------------------------------------------------------------

class FusionModel(nn.Module):
    """Three-way fusion model.

    Combines:
    - PointCloudGNN   (cloud_hidden_dim=128, 6 layers)
    - CADTopoEncoder  (topo_hidden_dim=64, 2 layers) → (B, 192)
    - GlobalMLP       (14→64→64→64)                  → (B, 64)
    → context = cat(topo_emb, global_emb)            → (B, 256)

    Task A head (drop):    (B, 128+256=384) → MLP → (B, 1)
    Task B head (node_p):  (N, 128+128=256) → MLP → (N, 1)
    """

    def __init__(
        self,
        point_in_dim: int,
        face_in_dim: int,
        edge_in_dim: int = 2,
        global_feature_dim: int = 14,
        cloud_in_dim: int = CLOUD_INPUT_DIM,
        cloud_hidden_dim: int = 128,
        cloud_num_layers: int = 6,
        cloud_k: int = CLOUD_K,
        topo_hidden_dim: int = 64,
        global_mlp_dim: int = GLOBAL_MLP_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.cloud_hidden_dim = cloud_hidden_dim
        self.topo_hidden_dim = topo_hidden_dim
        self.global_mlp_dim = global_mlp_dim

        # ---- Sub-modules -------------------------------------------------
        self.cloud_gnn = PointCloudGNN(
            cloud_in_dim=cloud_in_dim,
            hidden_dim=cloud_hidden_dim,
            num_layers=cloud_num_layers,
            k=cloud_k,
            dropout=dropout,
        )

        self.topo_encoder = CADTopoEncoder(
            point_in_dim=point_in_dim,
            face_in_dim=face_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=topo_hidden_dim,
            dropout=dropout,
        )

        self.global_mlp = MLP(
            in_dim=global_feature_dim,
            hidden_dim=global_mlp_dim,
            out_dim=global_mlp_dim,
            num_layers=3,
            dropout=0.0,
        )

        # context_dim = 3 * topo_hidden_dim + global_mlp_dim = 192 + 64 = 256
        context_dim = 3 * topo_hidden_dim + global_mlp_dim

        # ---- Task A: global Δp -------------------------------------------
        self.drop_head = MLP(
            in_dim=cloud_hidden_dim + context_dim,   # 128 + 256 = 384
            hidden_dim=256,
            out_dim=1,
            num_layers=3,
            dropout=dropout,
        )

        # ---- Task B: node-level pressure ---------------------------------
        # Project context → cloud_hidden_dim so we can concatenate with h_cloud
        self.ctx_proj = nn.Linear(context_dim, cloud_hidden_dim)  # 256 → 128
        self.node_pressure_head = MLP(
            in_dim=cloud_hidden_dim + cloud_hidden_dim,  # 128 + 128 = 256
            hidden_dim=128,
            out_dim=1,
            num_layers=3,
            dropout=dropout,
        )

    # ------------------------------------------------------------------

    def forward(self, data, task: str = "drop"):
        """
        Parameters
        ----------
        data : HeteroData batch from FusionDataset
        task : ``'drop'`` or ``'node_pressure'``

        Returns
        -------
        For task='drop':
            (dp_pred, None, None)
            dp_pred : (B, 1)
        For task='node_pressure':
            (None, p_node_pred, cloud_batch)
            p_node_pred : (N, 1)
            cloud_batch : (N,)
        """
        # ---- Build cloud batch vector from data.cloud_n ------------------
        cloud_n = data.cloud_n.view(-1)  # (B,)
        cloud_batch = torch.arange(cloud_n.shape[0], device=cloud_n.device).repeat_interleave(
            cloud_n
        )  # (N,)

        # ---- Point cloud GNN --------------------------------------------
        h_cloud = self.cloud_gnn(data.cloud_x, cloud_batch)  # (N, cloud_hidden_dim)

        # ---- Topo encoder -----------------------------------------------
        topo_emb = self.topo_encoder(data)  # (B, 192)

        # ---- Global MLP -------------------------------------------------
        global_emb = self.global_mlp(data.global_features)  # (B, 64)

        # ---- Context fusion ---------------------------------------------
        context = torch.cat([topo_emb, global_emb], dim=-1)  # (B, 256)

        if task == "drop":
            # Mean-pool cloud features
            cloud_pool = global_mean_pool(h_cloud, cloud_batch)  # (B, cloud_hidden_dim)
            graph_input = torch.cat([cloud_pool, context], dim=-1)  # (B, 384)
            dp_pred = self.drop_head(graph_input)  # (B, 1)
            return dp_pred, None, None

        if task == "node_pressure":
            # Broadcast context to every cloud node
            ctx_broad = self.ctx_proj(context)[cloud_batch]  # (N, 128)
            node_input = torch.cat([h_cloud, ctx_broad], dim=-1)  # (N, 256)
            p_node_pred = self.node_pressure_head(node_input)  # (N, 1)
            return None, p_node_pred, cloud_batch

        raise ValueError(f"Unknown task '{task}'. Use 'drop' or 'node_pressure'.")
