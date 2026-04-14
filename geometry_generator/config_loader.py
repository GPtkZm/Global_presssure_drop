"""
config_loader.py
================
Load and validate the YAML configuration for geometry_generator.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


# ── Defaults (used if a key is absent from the user's YAML) ──────────────────

_DEFAULTS: dict[str, Any] = {
    "grid": {"Nx": 60, "Ny": 30},
    "inlet": {"wall": "left", "pos": 5},
    "outlet": {"wall": "left", "pos": 25},
    # ── Backbone generation ────────────────────────────────
    "backbone": {
        "num_backbones": 6,
        "type": "serpentine",
        "spacing": 4,
        "p_perturb": 0.10,
        "max_perturb": 1,
    },
    # ── Vertical connectors ────────────────────────────────
    "connectors": {
        "density": 0.15,
        "min_length": 2,
        "p_prune": 0.30,
    },
    # ── Coverage ──────────────────────────────────────────
    "coverage": {
        "min_coverage": 0.30,
        "target_coverage": [0.40, 0.70],
        "subregion_cols": 12,
        "subregion_rows": 6,
    },
    # ── Manufacturing constraints ─────────────────────────
    "manufacturing": {
        "min_spacing": 2,
        "max_consecutive_turns": 3,
        "channel_width_mm": 12,
    },
    "protection_radius": 4,
    "allow_dead_ends": False,
    "loops": {"p_loop": 0.20},
    "pipe_width": 1,
    "num_samples": 10,
    "seed": None,
    "output": {
        "dir": "output",
        "save_grid": True,
        "save_graph": True,
        "save_image": True,
        "save_summary": True,
    },
    "visualization": {
        "dpi": 120,
        "figsize": [10, 5],
        "color_empty": "#f5f5f5",
        "color_main": "#1f77b4",
        "color_branch": "#ff7f0e",
        "color_inlet": "#2ca02c",
        "color_outlet": "#d62728",
        "color_protection": "#ffe0e0",
        "show_grid_lines": True,
        "show_inlet_outlet": True,
        "show_protection_zone": True,
        "summary_cols": 5,
        "summary_dpi": 80,
        "summary_figsize": None,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from a YAML file and merge with defaults.

    Parameters
    ----------
    path:
        Path to the YAML config file.  If *None*, the default
        ``config.yaml`` bundled with this package is used.

    Returns
    -------
    dict
        Validated configuration dictionary.
    """
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        user_cfg = yaml.safe_load(fh) or {}

    cfg = _deep_merge(_DEFAULTS, user_cfg)
    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    """Raise *ValueError* if *cfg* contains invalid values."""
    Nx: int = cfg["grid"]["Nx"]
    Ny: int = cfg["grid"]["Ny"]
    if Nx < 4 or Ny < 4:
        raise ValueError("Grid dimensions must be at least 4×4.")

    for port_name in ("inlet", "outlet"):
        port = cfg[port_name]
        wall = port["wall"]
        pos = port["pos"]
        if wall not in ("left", "right", "top", "bottom"):
            raise ValueError(
                f"{port_name}.wall must be one of left/right/top/bottom, "
                f"got {wall!r}"
            )
        max_pos = (Ny - 1) if wall in ("left", "right") else (Nx - 1)
        if not (0 <= pos <= max_pos):
            raise ValueError(
                f"{port_name}.pos={pos} is out of range for wall={wall!r} "
                f"(0..{max_pos})"
            )

    # Backbone validation
    bb = cfg["backbone"]
    if bb["num_backbones"] < 2:
        raise ValueError("backbone.num_backbones must be ≥ 2.")
    for prob_key in (("backbone", "p_perturb"),):
        val = cfg[prob_key[0]][prob_key[1]]
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{'.'.join(prob_key)} must be in [0, 1], got {val}")

    # Connector validation
    conn = cfg["connectors"]
    if not (0.0 <= conn["density"] <= 1.0):
        raise ValueError("connectors.density must be in [0, 1].")
    if not (0.0 <= conn["p_prune"] <= 1.0):
        raise ValueError("connectors.p_prune must be in [0, 1].")

    # Coverage validation
    cov = cfg["coverage"]
    if not (0.0 <= cov["min_coverage"] <= 1.0):
        raise ValueError("coverage.min_coverage must be in [0, 1].")
    if cov["subregion_cols"] < 1:
        raise ValueError("coverage.subregion_cols must be ≥ 1.")
    if cov["subregion_rows"] < 1:
        raise ValueError("coverage.subregion_rows must be ≥ 1.")

    # Manufacturing validation
    mfg = cfg["manufacturing"]
    if mfg["min_spacing"] < 1:
        raise ValueError("manufacturing.min_spacing must be ≥ 1.")
    if mfg["max_consecutive_turns"] < 1:
        raise ValueError("manufacturing.max_consecutive_turns must be ≥ 1.")

    # Protection radius
    if cfg["protection_radius"] < 0:
        raise ValueError("protection_radius must be ≥ 0.")

    # Loop probability
    val = cfg["loops"]["p_loop"]
    if not (0.0 <= val <= 1.0):
        raise ValueError(f"loops.p_loop must be in [0, 1], got {val}")

    # Batch / output
    if cfg["num_samples"] < 1:
        raise ValueError("num_samples must be ≥ 1.")
    if cfg["pipe_width"] < 1:
        raise ValueError("pipe_width must be ≥ 1.")

