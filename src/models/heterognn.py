"""
models/heterognn.py
-------------------
Dual-subnet heterogeneous GNN for global pressure-drop regression.

Architecture overview
~~~~~~~~~~~~~~~~~~~~~
1. **GNN Subnet** – processes the topology (.npy) data:

   a. *Encoders* – three lightweight MLPs that project each node type into a
      shared hidden space of dimension ``hidden_dim``:
        - point_encoder : MLP(3          → hidden_dim)
        - edge_encoder  : MLP(2          → hidden_dim)
        - face_encoder  : MLP(face_in_dim → hidden_dim)

   b. *Message passing* – ``num_layers`` stacked HeteroConv layers, each
      containing one SAGEConv per edge type (8 total).  Residual connections
      and LayerNorm are applied to all three node types after every layer.

   c. *Readout* – triple global mean pooling (point + edge + face), then
      concatenation → vector of size ``3 * hidden_dim``.

2. **MLP Subnet** – processes the CSV physics/geometry parameters:
   - Input  : 14-dim normalised global features
   - Layers : MLP(14 → 64 → 64 → 64)
   - Output : 64-dim vector

3. **Fusion decoder** – concatenates GNN output (3 * hidden_dim) and MLP
   output (64) then passes through a three-layer MLP:
     (3 * hidden_dim + 64) → 256 → 128 → 1

All hyperparameters have sensible defaults but can be overridden via the
constructor or by editing config.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool

from src.config import GLOBAL_MLP_DIM


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
        assert num_layers >= 1, "num_layers must be at least 1"

        layers = []
        current = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current = hidden_dim
        layers.append(nn.Linear(current, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HeteroGNN(nn.Module):
    """Dual-subnet heterogeneous GNN for graph-level pressure-drop regression."""

    def __init__(
        self,
        point_in_dim: int,
        face_in_dim: int,
        edge_in_dim: int = 2,
        global_feature_dim: int = 14,
        hidden_dim: int = 128,
        num_layers: int = 6,
        dropout: float = 0.1,
        global_mlp_dim: int = GLOBAL_MLP_DIM,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.global_feature_dim = global_feature_dim
        self.global_mlp_dim = global_mlp_dim

        # ---- GNN Subnet: Encoders ----------------------------------------
        self.point_encoder = MLP(point_in_dim, hidden_dim, hidden_dim, num_layers=2)
        self.edge_encoder = MLP(edge_in_dim, hidden_dim, hidden_dim, num_layers=2)
        self.face_encoder = MLP(face_in_dim, hidden_dim, hidden_dim, num_layers=2)

        # ---- GNN Subnet: Message-passing layers --------------------------
        self.convs = nn.ModuleList()
        self.point_norms = nn.ModuleList()
        self.edge_norms = nn.ModuleList()
        self.face_norms = nn.ModuleList()

        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ("point", "to", "point"): SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim
                    ),
                    ("face", "to", "point"): SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim
                    ),
                    ("edge", "to", "point"): SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim
                    ),
                    ("point", "to", "face"): SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim
                    ),
                    ("edge", "to", "face"): SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim
                    ),
                    ("face", "to", "face"): SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim
                    ),
                    ("point", "to", "edge"): SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim
                    ),
                    ("face", "to", "edge"): SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)
            self.point_norms.append(nn.LayerNorm(hidden_dim))
            self.edge_norms.append(nn.LayerNorm(hidden_dim))
            self.face_norms.append(nn.LayerNorm(hidden_dim))

        # ---- MLP Subnet (physics/geometry global features) ---------------
        self.global_mlp = MLP(
            in_dim=global_feature_dim,
            hidden_dim=global_mlp_dim,
            out_dim=global_mlp_dim,
            num_layers=3,
            dropout=0.0,
        )

        # ---- Fusion Decoder ----------------------------------------------
        fusion_in_dim = 3 * hidden_dim + global_mlp_dim
        self.decoder = MLP(
            in_dim=fusion_in_dim,
            hidden_dim=256,
            out_dim=1,
            num_layers=3,
            dropout=dropout,
        )

    def forward(self, data, global_features: torch.Tensor = None) -> torch.Tensor:
        """Forward pass."""
        if global_features is None:
            global_features = data.global_features

        h_point = self.point_encoder(data["point"].x)
        h_edge = self.edge_encoder(data["edge"].x)
        h_face = self.face_encoder(data["face"].x)

        for i, conv in enumerate(self.convs):
            x_dict = {"point": h_point, "edge": h_edge, "face": h_face}

            edge_index_dict = {
                ("point", "to", "point"): data["point", "to", "point"].edge_index,
                ("face", "to", "point"): data["face", "to", "point"].edge_index,
                ("edge", "to", "point"): data["edge", "to", "point"].edge_index,
                ("point", "to", "face"): data["point", "to", "face"].edge_index,
                ("edge", "to", "face"): data["edge", "to", "face"].edge_index,
                ("face", "to", "face"): data["face", "to", "face"].edge_index,
                ("point", "to", "edge"): data["point", "to", "edge"].edge_index,
                ("face", "to", "edge"): data["face", "to", "edge"].edge_index,
            }

            out_dict = conv(x_dict, edge_index_dict)

            h_point = self.point_norms[i](h_point + F.relu(out_dict["point"]))
            h_face = self.face_norms[i](h_face + F.relu(out_dict["face"]))
            if h_edge.shape[0] > 0:
                edge_out = out_dict.get("edge", torch.zeros_like(h_edge))
                h_edge = self.edge_norms[i](h_edge + F.relu(edge_out))

        num_graphs = int(data["point"].batch.max().item()) + 1
        g_point = global_mean_pool(h_point, data["point"].batch, size=num_graphs)
        g_face = global_mean_pool(h_face, data["face"].batch, size=num_graphs)

        if h_edge.shape[0] > 0:
            g_edge = global_mean_pool(h_edge, data["edge"].batch, size=num_graphs)
        else:
            g_edge = torch.zeros(num_graphs, self.hidden_dim, device=h_point.device)

        g_gnn = torch.cat([g_point, g_edge, g_face], dim=-1)

        g_physics = self.global_mlp(global_features)

        g_fused = torch.cat([g_gnn, g_physics], dim=-1)
        out = self.decoder(g_fused)
        return out
