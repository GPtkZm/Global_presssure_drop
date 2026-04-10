"""
model.py
--------
Backward-compatibility shim.  The actual model implementations now live in
``src/models/``.  Import from there directly, or use ``build_model(cfg)``
from ``src.models``.
"""

from src.models.heterognn import HeteroGNN, MLP  # noqa: F401
from src.models import build_model  # noqa: F401

__all__ = ["HeteroGNN", "MLP", "build_model"]
