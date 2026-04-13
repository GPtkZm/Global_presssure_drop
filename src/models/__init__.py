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
    CLOUD_INPUT_DIM,
    CLOUD_K,
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
)
from src.models.fusion_model import FusionModel
from src.models.heterognn import HeteroGNN
from src.models.transformer import TransformerPressureDrop

__all__ = ["HeteroGNN", "TransformerPressureDrop", "FusionModel", "build_model", "build_fusion_model"]


def build_model(cfg: dict) -> "torch.nn.Module":
    """Factory function: build and return the model specified in *cfg*.

    Parameters
    ----------
    cfg : dict
        Must contain keys:
          - ``model_type``        : ``"heterognn"`` or ``"transformer"``
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

    if model_type == "fusion":
        return build_fusion_model(cfg)

    raise ValueError(
        f"Unknown model_type '{model_type}'. "
        f"Supported types: 'heterognn', 'transformer', 'fusion'."
    )


def build_fusion_model(cfg: dict) -> "FusionModel":
    """Convenience factory for FusionModel.

    Parameters
    ----------
    cfg : dict
        Must contain: ``point_in_dim``, ``face_in_dim``.
        Optional overrides: ``edge_in_dim``, ``global_feature_dim``,
        ``cloud_in_dim``, ``cloud_hidden_dim``, ``cloud_num_layers``,
        ``cloud_k``, ``topo_hidden_dim``, ``global_mlp_dim``, ``dropout``.
    """
    return FusionModel(
        point_in_dim=cfg["point_in_dim"],
        face_in_dim=cfg["face_in_dim"],
        edge_in_dim=cfg.get("edge_in_dim", 2),
        global_feature_dim=cfg.get("global_feature_dim", len(GLOBAL_FEATURE_COLUMNS)),
        cloud_in_dim=cfg.get("cloud_in_dim", CLOUD_INPUT_DIM),
        cloud_hidden_dim=cfg.get("cloud_hidden_dim", 128),
        cloud_num_layers=cfg.get("cloud_num_layers", 6),
        cloud_k=cfg.get("cloud_k", CLOUD_K),
        topo_hidden_dim=cfg.get("topo_hidden_dim", 64),
        global_mlp_dim=cfg.get("global_mlp_dim", GLOBAL_MLP_DIM),
        dropout=cfg.get("dropout", DROPOUT),
    )
