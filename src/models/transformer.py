"""
models/transformer.py
---------------------
Transformer-based model for global pressure-drop regression.

Architecture overview
~~~~~~~~~~~~~~~~~~~~~
1. **Node Feature Projection** – three linear projections map each node type
   (point / face / edge) from their respective input dimensions into a common
   ``d_model`` space.

2. **Global-feature Projection** – a small MLP maps the 14-dim physics/
   geometry CSV parameters into a ``d_model``-dim vector.

3. **Transformer Encoder** – the projected node feature vectors are treated as
   a sequence (optionally with a learnable CLS token) and processed by
   ``num_encoder_layers`` standard TransformerEncoder layers.

4. **Pooling** – the output sequence is pooled (mean or max over the sequence
   dimension) to produce a single ``d_model``-dim graph embedding.

5. **Fusion & Regression** – the pooled GNN embedding is concatenated with the
   global-feature embedding and fed through a small MLP to produce a single
   scalar pressure-drop prediction.

Compatibility
~~~~~~~~~~~~~
Accepts the same ``HeteroData`` objects produced by ``PressureDropDataset``.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool

from src.config import (
    GLOBAL_MLP_DIM,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_DIM_FEEDFORWARD,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_ENCODER_LAYERS,
    TRANSFORMER_POOL,
)


class TransformerPressureDrop(nn.Module):
    """Transformer encoder model for graph-level pressure-drop regression.

    Parameters
    ----------
    point_in_dim : int
        Dimensionality of 'point' node input features.
    face_in_dim : int
        Dimensionality of 'face' node input features.
    edge_in_dim : int
        Dimensionality of 'edge' node input features.
    global_feature_dim : int
        Dimensionality of the physics/geometry global features (default 14).
    d_model : int
        Transformer model dimension.
    nhead : int
        Number of attention heads.
    num_encoder_layers : int
        Number of TransformerEncoder layers.
    dim_feedforward : int
        Feedforward dimension inside transformer.
    dropout : float
        Dropout probability.
    pool : str
        Pooling strategy over the node sequence: ``"mean"`` or ``"max"``.
    global_mlp_dim : int
        Output dimension of the global-feature MLP.
    """

    def __init__(
        self,
        point_in_dim: int,
        face_in_dim: int,
        edge_in_dim: int = 2,
        global_feature_dim: int = 14,
        d_model: int = TRANSFORMER_D_MODEL,
        nhead: int = TRANSFORMER_NHEAD,
        num_encoder_layers: int = TRANSFORMER_NUM_ENCODER_LAYERS,
        dim_feedforward: int = TRANSFORMER_DIM_FEEDFORWARD,
        dropout: float = TRANSFORMER_DROPOUT,
        pool: str = TRANSFORMER_POOL,
        global_mlp_dim: int = GLOBAL_MLP_DIM,
    ):
        super().__init__()

        self.d_model = d_model
        self.pool = pool

        # ---- Node-type projections ----------------------------------------
        self.point_proj = nn.Linear(point_in_dim, d_model)
        self.face_proj = nn.Linear(face_in_dim, d_model)
        self.edge_proj = nn.Linear(edge_in_dim, d_model)

        # Type embeddings (one per node type) so the transformer can distinguish types
        self.type_embed = nn.Embedding(3, d_model)  # 0=point, 1=face, 2=edge

        # ---- Global-feature MLP ------------------------------------------
        self.global_mlp = nn.Sequential(
            nn.Linear(global_feature_dim, global_mlp_dim),
            nn.ReLU(),
            nn.Linear(global_mlp_dim, global_mlp_dim),
            nn.ReLU(),
            nn.Linear(global_mlp_dim, global_mlp_dim),
        )

        # ---- Transformer Encoder -----------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # ---- Fusion Decoder ----------------------------------------------
        fusion_in_dim = d_model + global_mlp_dim
        self.decoder = nn.Sequential(
            nn.Linear(fusion_in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def _pool_nodes(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Mean- or max-pool node features to graph level."""
        if self.pool == "max":
            return global_max_pool(h, batch, size=num_graphs)
        return global_mean_pool(h, batch, size=num_graphs)

    def forward(self, data, global_features: torch.Tensor = None) -> torch.Tensor:
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
            global_features = data.global_features  # (B, global_feature_dim)

        num_graphs = int(data["point"].batch.max().item()) + 1

        # ---- Project each node type to d_model ---------------------------
        # Use new_zeros / new_full to create index tensors on the correct device
        # without a separate allocation step.
        point_x = data["point"].x
        face_x = data["face"].x
        h_edge_nodes = data["edge"].x

        h_point = self.point_proj(point_x) + self.type_embed(
            point_x.new_zeros(point_x.shape[0], dtype=torch.long)
        )
        h_face = self.face_proj(face_x) + self.type_embed(
            face_x.new_ones(face_x.shape[0], dtype=torch.long)
        )
        if h_edge_nodes.shape[0] > 0:
            h_edge = self.edge_proj(h_edge_nodes) + self.type_embed(
                h_edge_nodes.new_full((h_edge_nodes.shape[0],), 2, dtype=torch.long)
            )
        else:
            h_edge = torch.zeros(0, self.d_model, device=point_x.device)

        # ---- Pool each node type to graph level --------------------------
        g_point = self._pool_nodes(h_point, data["point"].batch, num_graphs)  # (B, d_model)
        g_face = self._pool_nodes(h_face, data["face"].batch, num_graphs)     # (B, d_model)
        if h_edge.shape[0] > 0:
            g_edge = self._pool_nodes(h_edge, data["edge"].batch, num_graphs) # (B, d_model)
        else:
            g_edge = torch.zeros(num_graphs, self.d_model, device=data["point"].x.device)

        # ---- Build sequence: [point_pool, edge_pool, face_pool] ----------
        # Shape: (B, 3, d_model)
        seq = torch.stack([g_point, g_edge, g_face], dim=1)

        # ---- Transformer Encoder -----------------------------------------
        encoded = self.transformer_encoder(seq)  # (B, 3, d_model)

        # ---- Aggregate over the 3-token sequence -------------------------
        if self.pool == "max":
            graph_emb = encoded.max(dim=1).values   # (B, d_model)
        else:
            graph_emb = encoded.mean(dim=1)          # (B, d_model)

        # ---- Global-feature MLP ------------------------------------------
        g_physics = self.global_mlp(global_features)  # (B, global_mlp_dim)

        # ---- Fusion + Decode ---------------------------------------------
        g_fused = torch.cat([graph_emb, g_physics], dim=-1)  # (B, d_model + global_mlp_dim)
        out = self.decoder(g_fused)                           # (B, 1)
        return out
