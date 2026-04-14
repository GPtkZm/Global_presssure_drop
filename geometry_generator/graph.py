"""
graph.py
========
Build a node/edge graph representation from a 2-D pipe-network grid.

Graph convention
----------------
* Each non-zero grid cell is a **node** with coordinate ``(x, y)``.
* Two nodes share an **edge** iff they are 4-connected neighbours
  (up / down / left / right) and both are pipe cells.

The graph is stored as a plain ``dict`` for easy JSON serialisation::

    {
        "nodes": [[x0, y0], [x1, y1], ...],
        "edges": [[src_idx, dst_idx], ...],
        "node_type": [1, 2, ...],   # 1 = main path, 2 = branch
        "inlet_idx":  <int>,
        "outlet_idx": <int>,
    }
"""

from __future__ import annotations

from typing import Any

import numpy as np


_DIRS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def build_graph(network: dict[str, Any]) -> dict[str, Any]:
    """Construct a graph from the output of :func:`network.generate_network`.

    Parameters
    ----------
    network:
        Dictionary returned by :func:`~geometry_generator.network.generate_network`.

    Returns
    -------
    dict
        Graph with keys ``nodes``, ``edges``, ``node_type``,
        ``inlet_idx``, ``outlet_idx``.
    """
    grid: np.ndarray = network["grid"]
    Ny, Nx = grid.shape
    inlet: tuple[int, int] = network["inlet"]
    outlet: tuple[int, int] = network["outlet"]

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
                node_type.append(int(grid[y, x]))  # 1 = main, 2 = branch

    # ── Build edge list (undirected, each pair stored once) ───────────────
    edges: list[list[int]] = []

    for y in range(Ny):
        for x in range(Nx):
            if grid[y, x] == 0:
                continue
            src_idx = coord_to_idx[(x, y)]
            for dx, dy in _DIRS_4:
                nx, ny = x + dx, y + dy
                if 0 <= nx < Nx and 0 <= ny < Ny and grid[ny, nx] != 0:
                    dst_idx = coord_to_idx[(nx, ny)]
                    # Store each undirected edge only once (src < dst)
                    if src_idx < dst_idx:
                        edges.append([src_idx, dst_idx])

    # ── Inlet / outlet indices ────────────────────────────────────────────
    inlet_idx = coord_to_idx.get(inlet, -1)
    outlet_idx = coord_to_idx.get(outlet, -1)

    return {
        "nodes": nodes,
        "edges": edges,
        "node_type": node_type,
        "inlet_idx": inlet_idx,
        "outlet_idx": outlet_idx,
    }
