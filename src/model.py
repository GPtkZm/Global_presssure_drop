"""
model.py
--------
Heterogeneous Graph Neural Network for global pressure-drop prediction.

Architecture overview
~~~~~~~~~~~~~~~~~~~~~
1. **Encoder** – lightweight MLPs that project each node type into a shared
   hidden space of dimension *hidden_dim*.

2. **Message passing** – *num_layers* stacked HeteroConv layers.  Each layer
   contains three SAGEConv operators (one per edge type) and applies residual
   connections + LayerNorm to improve gradient flow in deep networks.

3. **Readout** – dual global mean pooling over 'point' and 'face' nodes,
   then concatenation.

4. **Decoder** – a three-layer MLP that maps the pooled graph embedding to a
   single scalar (the normalised pressure drop).

All hyperparameters have sensible defaults but can be overridden via the
constructor or by editing config.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool


class MLP(nn.Module):
    """Simple multi-layer perceptron with ReLU activations and optional dropout.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Width of intermediate layers.
    out_dim : int
        Output dimension.
    num_layers : int
        Total number of linear layers (≥ 1).
    dropout : float
        Dropout probability applied after every hidden layer.
    """

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
    """Heterogeneous GNN for graph-level pressure-drop regression.

    Parameters
    ----------
    point_in_dim : int
        Dimensionality of 'point' node input features (e.g. 3 for XYZ).
    face_in_dim : int
        Dimensionality of 'face' node input features (one-hot + UV bounds).
    hidden_dim : int
        Width of all hidden layers (default 128).
    num_layers : int
        Number of heterogeneous message-passing layers (default 6).
    dropout : float
        Dropout probability in the decoder MLP (default 0.1).
    """

    def __init__(
        self,
        point_in_dim: int,
        face_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ---- Encoders ----------------------------------------------------
        # Project each node type into the shared hidden space
        self.point_encoder = MLP(point_in_dim, hidden_dim, hidden_dim, num_layers=2)
        self.face_encoder = MLP(face_in_dim, hidden_dim, hidden_dim, num_layers=2)

        # ---- Message-passing layers --------------------------------------
        self.convs = nn.ModuleList()
        self.point_norms = nn.ModuleList()
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
                    ("point", "to", "face"): SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)
            self.point_norms.append(nn.LayerNorm(hidden_dim))
            self.face_norms.append(nn.LayerNorm(hidden_dim))

        # ---- Decoder -----------------------------------------------------
        # Input: concatenation of pooled point and face embeddings → 2*hidden_dim
        self.decoder = MLP(
            in_dim=2 * hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_layers=3,
            dropout=dropout,
        )

    # ------------------------------------------------------------------

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        data : HeteroData
            A (possibly batched) heterogeneous graph containing:
              - data['point'].x            : (N_v, point_in_dim)
              - data['face'].x             : (N_f, face_in_dim)
              - data['point','to','point'].edge_index
              - data['face','to','point'].edge_index
              - data['point','to','face'].edge_index
              - data['point'].batch        : (N_v,) batch assignment
              - data['face'].batch         : (N_f,) batch assignment

        Returns
        -------
        torch.Tensor
            Shape (B, 1) – normalised pressure-drop prediction per graph.
        """
        # ---- Encode node features ----------------------------------------
        h_point = self.point_encoder(data["point"].x)   # (N_v, hidden)
        h_face = self.face_encoder(data["face"].x)       # (N_f, hidden)

        # ---- Message passing with residual connections -------------------
        for i, conv in enumerate(self.convs):
            # Build the x_dict and edge_index_dict expected by HeteroConv
            x_dict = {"point": h_point, "face": h_face}

            edge_index_dict = {
                ("point", "to", "point"): data["point", "to", "point"].edge_index,
                ("face", "to", "point"): data["face", "to", "point"].edge_index,
                ("point", "to", "face"): data["point", "to", "face"].edge_index,
            }

            out_dict = conv(x_dict, edge_index_dict)

            # Residual connection + LayerNorm
            h_point = self.point_norms[i](h_point + F.relu(out_dict["point"]))
            h_face = self.face_norms[i](h_face + F.relu(out_dict["face"]))

        # ---- Dual global mean pooling ------------------------------------
        batch_point = data["point"].batch    # (N_v,)
        batch_face = data["face"].batch      # (N_f,)

        g_point = global_mean_pool(h_point, batch_point)   # (B, hidden)
        g_face = global_mean_pool(h_face, batch_face)       # (B, hidden)

        g = torch.cat([g_point, g_face], dim=-1)            # (B, 2*hidden)

        # ---- Decode to scalar -------------------------------------------
        out = self.decoder(g)   # (B, 1)
        return out
