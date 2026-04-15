"""
models/__init__.py
------------------
Model registry and factory function for the pressure-drop prediction pipeline.

Usage
~~~~~
    from src.models import build_model

    model = build_model(cfg)
"""

from src.config import (
    DROPOUT,
    GLOBAL_FEATURE_COLUMNS,
    GLOBAL_MLP_DIM,
    HIDDEN_DIM,
    MODEL_TYPE,
    NUM_LAYERS,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_DIM_FEEDFORWARD,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_ENCODER_LAYERS,
    TRANSFORMER_POOL,
    GPS_ATTN_DROPOUT,
    GPS_DIM_FEEDFORWARD,
    GPS_DROPOUT,
    GPS_HIDDEN_DIM,
    GPS_NHEAD,
    GPS_NUM_LAYERS,
)
from src.models.heterognn import HeteroGNN
from src.models.transformer import TransformerPressureDrop
from src.models.graphgps import GraphGPSPressureDrop

__all__ = ["HeteroGNN", "TransformerPressureDrop", "GraphGPSPressureDrop", "build_model"]


def build_model(cfg: dict) -> "torch.nn.Module":
    """Factory function: build and return the model specified in *cfg*.

    Parameters
    ----------
    cfg : dict
        Must contain keys:
          - ``model_type``        : ``"heterognn"``, ``"transformer"``, or ``"graphgps"``
          - ``point_in_dim``      : int
          - ``face_in_dim``       : int
          - ``edge_in_dim``       : int
          - ``global_feature_dim``: int
        May also contain model-specific hyperparameter overrides.

    Returns
    -------
    torch.nn.Module
        The constructed (un-trained) model.

    Raises
    ------
    ValueError
        If ``cfg["model_type"]`` is not a recognised model name.
    """
    model_type = cfg.get("model_type", MODEL_TYPE).lower()
    point_in_dim = cfg["point_in_dim"]
    face_in_dim = cfg["face_in_dim"]
    edge_in_dim = cfg.get("edge_in_dim", 2)
    global_feature_dim = cfg.get("global_feature_dim", len(GLOBAL_FEATURE_COLUMNS))

    if model_type == "heterognn":
        return HeteroGNN(
            point_in_dim=point_in_dim,
            face_in_dim=face_in_dim,
            edge_in_dim=edge_in_dim,
            global_feature_dim=global_feature_dim,
            hidden_dim=cfg.get("hidden_dim", HIDDEN_DIM),
            num_layers=cfg.get("num_layers", NUM_LAYERS),
            dropout=cfg.get("dropout", DROPOUT),
            global_mlp_dim=cfg.get("global_mlp_dim", GLOBAL_MLP_DIM),
        )

    if model_type == "transformer":
        return TransformerPressureDrop(
            point_in_dim=point_in_dim,
            face_in_dim=face_in_dim,
            edge_in_dim=edge_in_dim,
            global_feature_dim=global_feature_dim,
            d_model=cfg.get("transformer_d_model", TRANSFORMER_D_MODEL),
            nhead=cfg.get("transformer_nhead", TRANSFORMER_NHEAD),
            num_encoder_layers=cfg.get(
                "transformer_num_encoder_layers", TRANSFORMER_NUM_ENCODER_LAYERS
            ),
            dim_feedforward=cfg.get(
                "transformer_dim_feedforward", TRANSFORMER_DIM_FEEDFORWARD
            ),
            dropout=cfg.get("transformer_dropout", TRANSFORMER_DROPOUT),
            pool=cfg.get("transformer_pool", TRANSFORMER_POOL),
            global_mlp_dim=cfg.get("global_mlp_dim", GLOBAL_MLP_DIM),
        )

    if model_type == "graphgps":
        return GraphGPSPressureDrop(
            point_in_dim=point_in_dim,
            face_in_dim=face_in_dim,
            edge_in_dim=edge_in_dim,
            global_feature_dim=global_feature_dim,
            hidden_dim=cfg.get("hidden_dim", GPS_HIDDEN_DIM),
            num_layers=cfg.get("num_layers", GPS_NUM_LAYERS),
            nhead=cfg.get("gps_nhead", GPS_NHEAD),
            dim_feedforward=cfg.get("gps_dim_feedforward", GPS_DIM_FEEDFORWARD),
            dropout=cfg.get("dropout", GPS_DROPOUT),
            attn_dropout=cfg.get("gps_attn_dropout", GPS_ATTN_DROPOUT),
            global_mlp_dim=cfg.get("global_mlp_dim", GLOBAL_MLP_DIM),
        )

    raise ValueError(
        f"Unknown model_type '{model_type}'. "
        f"Supported types: 'heterognn', 'transformer', 'graphgps'."
    )
