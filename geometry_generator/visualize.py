"""
visualize.py
============
Matplotlib-based visualisation for pipe-network geometries.

Public functions
----------------
* :func:`visualize_network`   – save a single-network PNG
* :func:`build_summary_image` – save a thumbnail montage of all networks
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ── Colour helpers ────────────────────────────────────────────────────────────

def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _build_rgb_image(
    grid: np.ndarray,
    color_empty: str,
    color_main: str,
    color_branch: str,
) -> np.ndarray:
    """Convert the integer grid to an RGB image array for imshow."""
    Ny, Nx = grid.shape
    img = np.ones((Ny, Nx, 3), dtype=float)

    ce = _hex_to_rgb(color_empty)
    cm = _hex_to_rgb(color_main)
    cb = _hex_to_rgb(color_branch)

    for y in range(Ny):
        for x in range(Nx):
            v = grid[y, x]
            if v == 0:
                img[y, x] = ce
            elif v == 1:
                img[y, x] = cm
            else:
                img[y, x] = cb

    return img


# ── Single-network visualisation ─────────────────────────────────────────────

def visualize_network(
    network: dict[str, Any],
    sample_id: str,
    cfg: dict[str, Any],
    save_path: str | Path,
) -> None:
    """Render a single pipe network and save it as a PNG.

    Parameters
    ----------
    network:
        Dictionary returned by :func:`~geometry_generator.network.generate_network`.
    sample_id:
        Human-readable identifier, e.g. ``"network_0001"``.
    cfg:
        Full configuration dictionary.
    save_path:
        Destination PNG file path.
    """
    vis_cfg = cfg["visualization"]
    grid = network["grid_vis"]          # dilated grid for display
    Ny, Nx = grid.shape
    inlet: tuple[int, int] = network["inlet"]
    outlet: tuple[int, int] = network["outlet"]

    figsize = tuple(vis_cfg["figsize"])
    dpi = vis_cfg["dpi"]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # ── Raster image ──────────────────────────────────────────────────────
    img = _build_rgb_image(
        grid,
        color_empty=vis_cfg["color_empty"],
        color_main=vis_cfg["color_main"],
        color_branch=vis_cfg["color_branch"],
    )
    ax.imshow(img, origin="upper", aspect="equal", interpolation="nearest")

    # ── Grid lines ────────────────────────────────────────────────────────
    if vis_cfg.get("show_grid_lines", True):
        ax.set_xticks(np.arange(-0.5, Nx, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, Ny, 1), minor=True)
        ax.grid(which="minor", color="#cccccc", linewidth=0.3)
        ax.tick_params(which="minor", length=0)

    ax.set_xticks(np.arange(0, Nx, max(1, Nx // 10)))
    ax.set_yticks(np.arange(0, Ny, max(1, Ny // 5)))

    # ── Inlet / outlet markers ────────────────────────────────────────────
    if vis_cfg.get("show_inlet_outlet", True):
        ix, iy = inlet
        ox, oy = outlet
        c_in = vis_cfg["color_inlet"]
        c_out = vis_cfg["color_outlet"]
        marker_kw = dict(s=120, zorder=5, linewidths=1.5, edgecolors="white")
        ax.scatter(ix, iy, color=c_in, marker=">", **marker_kw)
        ax.scatter(ox, oy, color=c_out, marker="s", **marker_kw)

    # ── Legend ────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=vis_cfg["color_main"], label="Main path"),
        mpatches.Patch(color=vis_cfg["color_branch"], label="Branch"),
        mpatches.Patch(color=vis_cfg["color_inlet"], label="Inlet"),
        mpatches.Patch(color=vis_cfg["color_outlet"], label="Outlet"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.8)

    # ── Title ─────────────────────────────────────────────────────────────
    br_cfg = cfg["branches"]
    title = (
        f"{sample_id}  |  {Nx}×{Ny} grid  |  "
        f"p_branch={br_cfg['p_branch']:.2f}  "
        f"p_split={br_cfg['p_split']:.2f}  "
        f"max_depth={br_cfg['max_depth']}"
    )
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── Summary montage ──────────────────────────────────────────────────────────

def build_summary_image(
    image_paths: list[Path],
    sample_ids: list[str],
    cfg: dict[str, Any],
    save_path: str | Path,
) -> None:
    """Create a thumbnail montage of all generated networks.

    Parameters
    ----------
    image_paths:
        Paths to the individual preview PNGs (in order).
    sample_ids:
        Corresponding sample IDs for sub-titles.
    cfg:
        Full configuration dictionary.
    save_path:
        Destination PNG for the montage.
    """
    if not image_paths:
        return

    vis_cfg = cfg["visualization"]
    n_cols: int = vis_cfg.get("summary_cols", 5)
    n_rows = math.ceil(len(image_paths) / n_cols)

    fig_size = vis_cfg.get("summary_figsize") or (n_cols * 4, n_rows * 2.5)
    dpi = vis_cfg.get("summary_dpi", 80)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=fig_size,
        dpi=dpi,
        squeeze=False,
    )

    for idx, (img_path, sid) in enumerate(zip(image_paths, sample_ids)):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        try:
            img = plt.imread(str(img_path))
            ax.imshow(img, aspect="auto")
        except Exception:
            ax.text(0.5, 0.5, "Error", ha="center", va="center")
        ax.set_title(sid, fontsize=6, pad=2)
        ax.axis("off")

    # Hide unused axes
    for idx in range(len(image_paths), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].axis("off")

    fig.suptitle("Generated Pipe Networks – Summary", fontsize=10, y=1.01)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
