"""
network.py
==========
Pipe/channel network generation on a 2-D integer grid.

Grid convention
---------------
* ``grid[y, x] == 0``  → empty cell
* ``grid[y, x] == 1``  → main-path pipe cell
* ``grid[y, x] == 2``  → branch pipe cell

The module provides:
* :func:`resolve_port`        – convert wall/pos config to (x, y) grid coord
* :func:`generate_network`    – produce a full network given a config dict
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────────────

_ALL_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def resolve_port(wall: str, pos: int, Nx: int, Ny: int) -> tuple[int, int]:
    """Convert a wall/pos specification to an ``(x, y)`` coordinate.

    Parameters
    ----------
    wall:
        One of ``"left"``, ``"right"``, ``"top"``, ``"bottom"``.
    pos:
        Row index for left/right walls; column index for top/bottom walls.
    Nx, Ny:
        Grid dimensions.

    Returns
    -------
    tuple[int, int]
        ``(x, y)`` coordinate inside the grid boundary.
    """
    if wall == "left":
        return (0, pos)
    elif wall == "right":
        return (Nx - 1, pos)
    elif wall == "top":
        return (pos, 0)
    elif wall == "bottom":
        return (pos, Ny - 1)
    else:
        raise ValueError(f"Unknown wall: {wall!r}")


def _in_bounds(x: int, y: int, Nx: int, Ny: int) -> bool:
    return 0 <= x < Nx and 0 <= y < Ny


# ── BFS shortest path ────────────────────────────────────────────────────────

def _bfs_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    Nx: int,
    Ny: int,
) -> list[tuple[int, int]]:
    """Return a shortest path from *start* to *goal* on an unconstrained grid.

    Uses 4-connectivity BFS.  Returns an empty list if the grid is fully
    disconnected (should not happen on an open grid).
    """
    if start == goal:
        return [start]

    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue: deque[tuple[int, int]] = deque([start])

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        cx, cy = current
        for dx, dy in _ALL_DIRS:
            nb = (cx + dx, cy + dy)
            if _in_bounds(*nb, Nx, Ny) and nb not in parent:
                parent[nb] = current
                queue.append(nb)

    if goal not in parent:
        return []

    # Reconstruct path
    path: list[tuple[int, int]] = []
    node: tuple[int, int] | None = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


# ── Main-path generation ─────────────────────────────────────────────────────

def _generate_main_path(
    inlet: tuple[int, int],
    outlet: tuple[int, int],
    Nx: int,
    Ny: int,
    p_perturb: float,
    bias_toward_outlet: float,
    max_steps: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Generate a wiggly main path from *inlet* to *outlet*.

    Strategy
    --------
    At each step, with probability ``bias_toward_outlet`` move one step in the
    direction that reduces Manhattan distance to the outlet; otherwise pick a
    random neighbour.  A perturbation step (90° detour) is taken with
    probability ``p_perturb``.  If the walk gets stuck or exceeds
    ``max_steps``, BFS supplies the remaining segment.
    """
    visited: set[tuple[int, int]] = set()
    path: list[tuple[int, int]] = [inlet]
    visited.add(inlet)
    current = inlet

    for _ in range(max_steps):
        if current == outlet:
            break
        cx, cy = current
        ox, oy = outlet

        # Preferred direction toward outlet
        pref_dx = (1 if ox > cx else -1) if ox != cx else 0
        pref_dy = (1 if oy > cy else -1) if oy != cy else 0

        # Build candidate list: biased toward outlet
        candidates: list[tuple[int, int]] = []

        if rng.random() < bias_toward_outlet:
            # Choose one of the two "toward outlet" moves
            toward = []
            if pref_dx != 0:
                nb = (cx + pref_dx, cy)
                if _in_bounds(*nb, Nx, Ny):
                    toward.append(nb)
            if pref_dy != 0:
                nb = (cx, cy + pref_dy)
                if _in_bounds(*nb, Nx, Ny):
                    toward.append(nb)
            if toward:
                candidates = toward

        if not candidates or rng.random() < p_perturb:
            # Perturbation: add all valid neighbours
            candidates = [
                (cx + dx, cy + dy)
                for dx, dy in _ALL_DIRS
                if _in_bounds(cx + dx, cy + dy, Nx, Ny)
                and (cx + dx, cy + dy) not in visited
            ]

        # Remove already-visited to avoid tight loops
        candidates = [c for c in candidates if c not in visited]

        if not candidates:
            break  # stuck – fall through to BFS

        next_cell = rng.choice(candidates)
        path.append(next_cell)
        visited.add(next_cell)
        current = next_cell

    if current != outlet:
        # Fallback: stitch remaining gap with BFS
        rest = _bfs_path(current, outlet, Nx, Ny)
        for cell in rest[1:]:  # skip 'current', already in path
            if cell not in visited:
                path.append(cell)
                visited.add(cell)
        # Ensure outlet is the last cell
        if path[-1] != outlet:
            path.append(outlet)

    return path


# ── Branch generation ────────────────────────────────────────────────────────

def _too_close_to_others(
    candidate: tuple[int, int],
    own_cells: set[tuple[int, int]],
    grid: np.ndarray,
    min_spacing: int,
    Nx: int,
    Ny: int,
) -> bool:
    """Return True if *candidate* is within *min_spacing* of a pipe cell
    that does not belong to this branch's own cells.

    The spacing is measured as Chebyshev distance (king-moves), which
    naturally penalises diagonal proximity.
    """
    cx, cy = candidate
    r = min_spacing - 1  # neighbourhood radius
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            xx, yy = cx + dx, cy + dy
            if not _in_bounds(xx, yy, Nx, Ny):
                continue
            if grid[yy, xx] != 0 and (xx, yy) not in own_cells:
                return True
    return False


def _grow_branch(
    start: tuple[int, int],
    grid: np.ndarray,
    depth: int,
    config: dict[str, Any],
    Nx: int,
    Ny: int,
    rng: random.Random,
    p_loop: float,
) -> list[tuple[int, int]]:
    """Recursively grow a branch starting at *start*.

    Parameters
    ----------
    start:
        Starting cell (must already be a pipe cell).
    grid:
        Current grid state (used for collision / spacing checks).
    depth:
        Current recursion depth.
    config:
        ``branches`` sub-dict from the loaded config.
    Nx, Ny:
        Grid dimensions.
    rng:
        Seeded random instance.
    p_loop:
        Probability of allowing a step onto an existing pipe cell to
        form a loop.

    Returns
    -------
    list[tuple[int, int]]
        New cells added by this branch (does **not** include *start*).
    """
    if depth > config["max_depth"]:
        return []

    branch_cells: list[tuple[int, int]] = []
    own_cells: set[tuple[int, int]] = {start}
    current = start
    dirs = [tuple(d) for d in config["directions"]]
    min_spacing: int = config["min_spacing"]

    for _ in range(config["max_length"]):
        cx, cy = current
        rng.shuffle(dirs)  # type: ignore[arg-type]

        moved = False
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if not _in_bounds(nx, ny, Nx, Ny):
                continue

            # Loop reconnection: allow stepping onto existing pipe
            if grid[ny, nx] != 0:
                if rng.random() < p_loop:
                    # Create a loop – stop growing after reconnection
                    return branch_cells
                continue

            # Spacing check – only against cells not owned by this branch
            if _too_close_to_others(
                (nx, ny), own_cells, grid, min_spacing, Nx, Ny
            ):
                continue

            # Valid step
            branch_cells.append((nx, ny))
            own_cells.add((nx, ny))
            # Temporarily mark grid so sub-branches respect this branch
            grid[ny, nx] = 2
            current = (nx, ny)
            moved = True
            break

        if not moved:
            break

        # Recursive split
        if rng.random() < config["p_split"]:
            sub = _grow_branch(
                current, grid, depth + 1, config, Nx, Ny, rng, p_loop
            )
            branch_cells.extend(sub)

    return branch_cells


# ── Connectivity check ───────────────────────────────────────────────────────

def _reachable_from(
    start: tuple[int, int],
    grid: np.ndarray,
    Nx: int,
    Ny: int,
) -> set[tuple[int, int]]:
    """BFS flood-fill of all pipe cells reachable from *start*."""
    reachable: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque([start])
    reachable.add(start)
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in _ALL_DIRS:
            nb = (cx + dx, cy + dy)
            if _in_bounds(*nb, Nx, Ny) and grid[nb[1], nb[0]] != 0 and nb not in reachable:
                reachable.add(nb)
                queue.append(nb)
    return reachable


def _remove_unreachable(
    grid: np.ndarray,
    inlet: tuple[int, int],
    Nx: int,
    Ny: int,
) -> np.ndarray:
    """Zero-out any pipe cells not reachable from *inlet*."""
    reachable = _reachable_from(inlet, grid, Nx, Ny)
    result = grid.copy()
    for y in range(Ny):
        for x in range(Nx):
            if result[y, x] != 0 and (x, y) not in reachable:
                result[y, x] = 0
    return result


# ── Dead-end removal ─────────────────────────────────────────────────────────

def _remove_dead_ends(
    grid: np.ndarray,
    protected: set[tuple[int, int]],
    Nx: int,
    Ny: int,
) -> np.ndarray:
    """Iteratively remove degree-1 pipe cells that are not *protected*.

    Parameters
    ----------
    grid:
        Input grid (modified in-place on a copy).
    protected:
        Cells that must never be removed (inlet, outlet).
    Nx, Ny:
        Grid dimensions.
    """
    g = grid.copy()
    changed = True
    while changed:
        changed = False
        for y in range(Ny):
            for x in range(Nx):
                if g[y, x] == 0 or (x, y) in protected:
                    continue
                degree = sum(
                    1
                    for dx, dy in _ALL_DIRS
                    if _in_bounds(x + dx, y + dy, Nx, Ny)
                    and g[y + dy, x + dx] != 0
                )
                if degree <= 1:
                    g[y, x] = 0
                    changed = True
    return g


# ── Pipe dilation (width > 1) ────────────────────────────────────────────────

def _dilate(grid: np.ndarray, width: int, Nx: int, Ny: int) -> np.ndarray:
    """Dilate the pipe mask so each cell expands to a square of *width* cells.

    The dilation is applied only to the *mask* used for visualisation; the
    topology (node/edge) data is derived from the undilated grid.
    """
    if width <= 1:
        return grid
    from scipy.ndimage import binary_dilation  # type: ignore

    struct = np.ones((width, width), dtype=bool)
    mask = binary_dilation(grid != 0, structure=struct)
    result = np.where(mask, np.where(grid != 0, grid, 2), 0).astype(int)
    return result


# ── Public API ───────────────────────────────────────────────────────────────

def generate_network(
    cfg: dict[str, Any],
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Generate a single pipe-network topology.

    Parameters
    ----------
    cfg:
        Loaded configuration dictionary (from :func:`config_loader.load_config`).
    rng:
        Optional seeded :class:`random.Random` instance.  If *None*, a new
        instance is created (non-deterministic).

    Returns
    -------
    dict with keys:

    * ``"grid"``       – ``np.ndarray`` shape ``(Ny, Nx)`` int, undilated
    * ``"grid_vis"``   – ``np.ndarray`` shape ``(Ny, Nx)`` int, dilated for viz
    * ``"main_path"``  – list of ``(x, y)`` tuples
    * ``"inlet"``      – ``(x, y)``
    * ``"outlet"``     – ``(x, y)``
    * ``"Nx"``, ``"Ny"``
    """
    if rng is None:
        rng = random.Random()

    Nx: int = cfg["grid"]["Nx"]
    Ny: int = cfg["grid"]["Ny"]

    inlet = resolve_port(
        cfg["inlet"]["wall"], cfg["inlet"]["pos"], Nx, Ny
    )
    outlet = resolve_port(
        cfg["outlet"]["wall"], cfg["outlet"]["pos"], Nx, Ny
    )

    mp_cfg = cfg["main_path"]
    br_cfg = cfg["branches"]
    p_loop: float = cfg["loops"]["p_loop"]

    # ── 1. Main path ──────────────────────────────────────────────────────
    main_path = _generate_main_path(
        inlet,
        outlet,
        Nx,
        Ny,
        p_perturb=mp_cfg["p_perturb"],
        bias_toward_outlet=mp_cfg["bias_toward_outlet"],
        max_steps=mp_cfg["max_steps"],
        rng=rng,
    )

    grid = np.zeros((Ny, Nx), dtype=int)
    for x, y in main_path:
        grid[y, x] = 1

    # ── 2. Branches ───────────────────────────────────────────────────────
    main_path_set: set[tuple[int, int]] = set(main_path)

    for cell in list(main_path):
        if rng.random() < br_cfg["p_branch"]:
            branch_cells = _grow_branch(
                cell, grid, depth=1,
                config=br_cfg, Nx=Nx, Ny=Ny, rng=rng,
                p_loop=p_loop,
            )
            for x, y in branch_cells:
                if grid[y, x] == 0:
                    grid[y, x] = 2  # branch cell (may have been set already)

    # ── 3. Connectivity: remove cells unreachable from inlet ──────────────
    grid = _remove_unreachable(grid, inlet, Nx, Ny)

    # ── 4. Optional dead-end removal ─────────────────────────────────────
    if cfg.get("remove_dead_ends", False):
        protected = {inlet, outlet}
        grid = _remove_dead_ends(grid, protected, Nx, Ny)

    # ── 5. Ensure inlet / outlet are still pipe cells ────────────────────
    ix, iy = inlet
    ox, oy = outlet
    if grid[iy, ix] == 0:
        grid[iy, ix] = 1
    if grid[oy, ox] == 0:
        grid[oy, ox] = 1

    # Recompute main_path to only include cells still in grid
    main_path_clean = [c for c in main_path if grid[c[1], c[0]] != 0]

    # ── 6. Optional pipe dilation for visualisation ───────────────────────
    pipe_width: int = cfg.get("pipe_width", 1)
    grid_vis = _dilate(grid, pipe_width, Nx, Ny) if pipe_width > 1 else grid.copy()

    return {
        "grid": grid,
        "grid_vis": grid_vis,
        "main_path": main_path_clean,
        "inlet": inlet,
        "outlet": outlet,
        "Nx": Nx,
        "Ny": Ny,
    }
