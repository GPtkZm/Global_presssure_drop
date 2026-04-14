"""
graph.py
========
Build a node/edge graph representation from a 2-D pipe-network grid.

Graph convention
----------------
* Each non-zero grid cell is a **node** with coordinate ``(x, y)``.
* Two nodes share an **edge** iff they are 4-connected neighbours
  (up / down / left / right) and both are pipe cells.

Output format (JSON-serialisable dict)::

    {
        "nodes":          [[x0, y0], [x1, y1], ...],
        "edges":          [[src_idx, dst_idx], ...],
        "node_features":  [[norm_x, norm_y, degree, is_junction, is_bend,
                            is_inlet, is_outlet, dist_to_inlet], ...],
        "edge_features":  [[direction, channel_width_mm], ...],
        "node_type":      [1, 2, ...],   # 1=backbone, 2=connector
        "inlet_idx":      <int>,
        "outlet_idx":     <int>,
    }

Node features
-------------
* ``norm_x``          – x / (Nx - 1)
* ``norm_y``          – y / (Ny - 1)
* ``degree``          – number of pipe-cell neighbours (1–4)
* ``is_junction``     – 1 if degree ≥ 3, else 0
* ``is_bend``         – 1 if degree == 2 with perpendicular neighbours, else 0
* ``is_inlet``        – 1 if this is the inlet cell, else 0
* ``is_outlet``       – 1 if this is the outlet cell, else 0
* ``dist_to_inlet``   – BFS distance from inlet, normalised by (Nx + Ny)

Edge features
-------------
* ``direction``       – 0 = horizontal, 1 = vertical
* ``channel_width_mm``– from manufacturing.channel_width_mm in config
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


_DIRS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


# ── BFS distance from a source on the pipe grid ──────────────────────────────

def _bfs_dist(
    source: tuple[int, int],
    grid: np.ndarray,
    Nx: int,
    Ny: int,
) -> dict[tuple[int, int], int]:
    """Return a dict mapping each reachable pipe cell to its BFS distance
    from *source* (traversing only pipe cells)."""
    dist: dict[tuple[int, int], int] = {source: 0}
    queue: deque[tuple[int, int]] = deque([source])
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in _DIRS_4:
            nb = (cx + dx, cy + dy)
            nx_c, ny_c = nb
            if (
                0 <= nx_c < Nx
                and 0 <= ny_c < Ny
                and grid[ny_c, nx_c] != 0
                and nb not in dist
            ):
                dist[nb] = dist[(cx, cy)] + 1
                queue.append(nb)
    return dist


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(network: dict[str, Any]) -> dict[str, Any]:
    """Construct a GNN-compatible graph from :func:`~geometry_generator.network.generate_network` output.

    Parameters
    ----------
    network:
        Dictionary returned by :func:`~geometry_generator.network.generate_network`.

    Returns
    -------
    dict
        Keys: ``nodes``, ``edges``, ``node_features``, ``edge_features``,
        ``node_type``, ``inlet_idx``, ``outlet_idx``.
    """
    grid: np.ndarray = network["grid"]
    Ny, Nx = grid.shape
    inlet: tuple[int, int] = network["inlet"]
    outlet: tuple[int, int] = network["outlet"]

    # Resolve channel_width_mm: prefer explicit value injected by generate.py,
    # fall back to the config embedded in the network dict, then default to 12.
    _cw = network.get("channel_width_mm")
    channel_width_mm: float = float(_cw) if _cw is not None else 12.0

    # ── Build node list ───────────────────────────────────────────────────
    coord_to_idx: dict[tuple[int, int], int] = {}
    nodes: list[list[int]] = []
    node_type: list[int] = []

    for y in range(Ny):
        for x in range(Nx):
            if grid[y, x] != 0:
                idx = len(nodes)
                coord_to_idx[(x, y)] = idx
                nodes.append([x, y])
                node_type.append(int(grid[y, x]))  # 1=backbone, 2=connector

    # ── Build edge list (each undirected pair stored once) ─────────────────
    edges: list[list[int]] = []
    edge_features: list[list[float]] = []

    for y in range(Ny):
        for x in range(Nx):
            if grid[y, x] == 0:
                continue
            src_idx = coord_to_idx[(x, y)]
            for dx, dy in _DIRS_4:
                nx_c, ny_c = x + dx, y + dy
                if 0 <= nx_c < Nx and 0 <= ny_c < Ny and grid[ny_c, nx_c] != 0:
                    dst_idx = coord_to_idx[(nx_c, ny_c)]
                    if src_idx < dst_idx:
                        edges.append([src_idx, dst_idx])
                        # direction: 0 = horizontal (dy==0), 1 = vertical (dx==0)
                        direction = 0 if dy == 0 else 1
                        edge_features.append([direction, channel_width_mm])

    # ── Inlet / outlet indices ────────────────────────────────────────────
    inlet_idx = coord_to_idx.get(inlet, -1)
    outlet_idx = coord_to_idx.get(outlet, -1)

    # ── Node degree ───────────────────────────────────────────────────────
    degree: list[int] = [0] * len(nodes)
    for src, dst in edges:
        degree[src] += 1
        degree[dst] += 1

    # ── BFS distance from inlet ───────────────────────────────────────────
    bfs_dist_map = _bfs_dist(inlet, grid, Nx, Ny)
    # Use the actual maximum BFS distance in the network for normalisation
    # (not Nx+Ny which underestimates for serpentine paths)
    max_dist = max(bfs_dist_map.values()) if bfs_dist_map else 1

    # ── Compute per-node features ─────────────────────────────────────────
    norm_denom_x = max(1, Nx - 1)
    norm_denom_y = max(1, Ny - 1)

    node_features: list[list[float]] = []
    for idx, (x, y) in enumerate(nodes):
        deg = degree[idx]

        # is_bend: degree == 2 AND the two neighbours are perpendicular
        is_bend = 0
        if deg == 2:
            nbs = [
                (x + dx, y + dy)
                for dx, dy in _DIRS_4
                if 0 <= x + dx < Nx
                and 0 <= y + dy < Ny
                and grid[y + dy, x + dx] != 0
            ]
            if len(nbs) == 2:
                dx1 = nbs[0][0] - x
                dy1 = nbs[0][1] - y
                dx2 = nbs[1][0] - x
                dy2 = nbs[1][1] - y
                # Perpendicular: dot product == 0
                if dx1 * dx2 + dy1 * dy2 == 0:
                    is_bend = 1

        raw_dist = bfs_dist_map.get((x, y), max_dist)
        node_features.append([
            x / norm_denom_x,           # norm_x
            y / norm_denom_y,           # norm_y
            float(deg),                 # degree
            float(1 if deg >= 3 else 0),  # is_junction
            float(is_bend),             # is_bend
            float(1 if (x, y) == inlet else 0),   # is_inlet
            float(1 if (x, y) == outlet else 0),  # is_outlet
            raw_dist / max_dist,        # dist_to_inlet (normalised)
        ])

    return {
        "nodes": nodes,
        "edges": edges,
        "node_features": node_features,
        "edge_features": edge_features,
        "node_type": node_type,
        "inlet_idx": inlet_idx,
        "outlet_idx": outlet_idx,
    }

