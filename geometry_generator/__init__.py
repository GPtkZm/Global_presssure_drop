"""
geometry_generator
==================
Procedural pipe/channel network geometry generator for 2D grids.

Generates random pipe network topologies with guaranteed inlet→outlet
connectivity, configurable branching, and matplotlib visualizations.

Usage
-----
    python -m geometry_generator --config geometry_generator/config.yaml
    python -m geometry_generator --num_samples 20 --seed 42
"""

from .config_loader import load_config
from .network import generate_network
from .graph import build_graph
from .visualize import visualize_network, build_summary_image

__all__ = [
    "load_config",
    "generate_network",
    "build_graph",
    "visualize_network",
    "build_summary_image",
]
