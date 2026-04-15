"""
network.py
==========
Multi-stage pipe/channel network generation for EV battery
liquid cooling channel networks on a 2-D integer grid.

Grid convention
---------------
* ``grid[y, x] == 0``  → empty cell
* ``grid[y, x] == 1``  → backbone pipe cell
* ``grid[y, x] == 2``  → connector pipe cell

Physical setup
--------------
* Plate: 1200mm × 600mm at 20mm resolution → 60 × 30 grid
* Inlet:  (0, 5)  — left wall, row 5
* Outlet: (0, 25) — left wall, row 25
* Both ports on the LEFT wall → serpentine flow required

Generation stages
-----------------
1. Generate horizontal backbone paths (serpentine)
2. Add vertical connectors between adjacent backbones
3. Random pruning while maintaining inlet→outlet connectivity
4. Ensure global connectivity (BFS from inlet, remove islands)
5. Coverage and subregion uniformity check (12×6 subregions)
6. Manufacturing constraint validation

The module provides:
* :func:`resolve_port`      – convert wall/pos config to (x, y) grid coord
* :func:`generate_network`  – produce a full network given a config dict
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

_ALL_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


# ── Utility helpers ───────────────────────────────────────────────────────────

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
        ``(x, y)`` coordinate on the grid boundary.
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


def _protection_zone(
    inlet: tuple[int, int],
    outlet: tuple[int, int],
    radius: int,
    Nx: int,
    Ny: int,
) -> set[tuple[int, int]]:
    """Return the set of cells within Manhattan distance *radius* of inlet or outlet."""
    protected: set[tuple[int, int]] = set()
    for port in (inlet, outlet):
        px, py = port
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dx) + abs(dy) <= radius:
                    x, y = px + dx, py + dy
                    if _in_bounds(x, y, Nx, Ny):
                        protected.add((x, y))
    return protected


def _reachable_from(
    start: tuple[int, int],
    grid: np.ndarray,
    Nx: int,
    Ny: int,
) -> set[tuple[int, int]]:
    """BFS flood-fill: all pipe cells reachable from *start*."""
    if grid[start[1], start[0]] == 0:
        return set()
    reachable: set[tuple[int, int]] = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in _ALL_DIRS:
            nb = (cx + dx, cy + dy)
            if (
                _in_bounds(*nb, Nx, Ny)
                and grid[nb[1], nb[0]] != 0
                and nb not in reachable
            ):
                reachable.add(nb)
                queue.append(nb)
    return reachable


def _bfs_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    Nx: int,
    Ny: int,
    grid: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Shortest path from *start* to *goal* using 4-connectivity BFS.

    If *grid* is provided, only traverses existing pipe cells (``!= 0``).
    Otherwise all in-bounds cells are reachable.
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
            if not _in_bounds(*nb, Nx, Ny) or nb in parent:
                continue
            if grid is not None and grid[nb[1], nb[0]] == 0:
                continue
            parent[nb] = current
            queue.append(nb)
    if goal not in parent:
        return []
    path: list[tuple[int, int]] = []
    node: tuple[int, int] | None = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


# ── Stage 1: Backbone generation ─────────────────────────────────────────────

def _generate_single_backbone(
    x_start: int,
    y_nominal: int,
    x_end: int,
    Nx: int,
    Ny: int,
    p_perturb: float,
    max_perturb: int,
    rng: random.Random,
    protection_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Generate a mostly-horizontal backbone path.

    The path goes from ``(x_start, y_nominal)`` to ``(x_end, y_nominal)``.
    In the middle section it may deviate vertically by up to *max_perturb*
    cells with probability *p_perturb*.  The first and last ``snap_margin``
    steps are kept straight to ensure clean serpentine connections.

    4-connectivity is maintained: when the y-level changes, a vertical
    intermediate cell is inserted before the horizontal move.
    """
    direction = 1 if x_end > x_start else -1
    total_steps = abs(x_end - x_start)
    snap_margin = max(4, total_steps // 10)

    cells: list[tuple[int, int]] = []
    y = y_nominal
    x = x_start
    cells.append((x, y))

    for step in range(1, total_steps + 1):
        next_x = x + direction
        dist_to_end = total_steps - step
        in_snap_zone = step < snap_margin or dist_to_end < snap_margin

        target_y = y

        # Perturbation: try to shift y by ±1
        if not in_snap_zone and rng.random() < p_perturb:
            dy = rng.choice([-1, 1])
            new_y = y + dy
            if (
                0 <= new_y < Ny
                and abs(new_y - y_nominal) <= max_perturb
                and (next_x, new_y) not in protection_cells
            ):
                target_y = new_y

        # Snap back toward y_nominal when near the end
        if in_snap_zone and y != y_nominal and dist_to_end > 0:
            if abs(y - y_nominal) <= dist_to_end:
                dy_snap = 1 if y_nominal > y else -1
                target_y = y + dy_snap

        # Insert vertical intermediate cell to maintain 4-connectivity
        if target_y != y:
            cells.append((x, target_y))
            y = target_y

        # Horizontal step
        x = next_x
        cells.append((x, y))

    # Ensure the path ends exactly at (x_end, y_nominal)
    if y != y_nominal:
        dy = 1 if y_nominal > y else -1
        while y != y_nominal:
            y += dy
            cells.append((x_end, y))

    return cells


def _generate_backbones(
    cfg: dict[str, Any],
    Nx: int,
    Ny: int,
    inlet: tuple[int, int],
    outlet: tuple[int, int],
    rng: random.Random,
    protection_cells: set[tuple[int, int]],
) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[int]]:
    """Stage 1: Generate horizontal backbone paths in a serpentine pattern.

    Since both inlet and outlet are on the LEFT wall, the pattern is:

    * BB0 (even → left→right): starts at inlet  ``(0, inlet_y)``
    * BB1 (odd  → right→left): starts at        ``(Nx-1, y1)``
    * BB2 (even → left→right): starts at        ``(0, y2)``
    * ...
    * BBn (last): ends at outlet ``(0, outlet_y)``

    Adjacent backbones are joined by vertical "turn" segments at the
    appropriate left or right edge of the grid.

    Returns
    -------
    grid:
        ``(Ny, Nx)`` array; backbone cells have value 1.
    backbones:
        List of cell lists for each backbone (excluding turn segments).
    y_levels:
        Nominal y-coordinate for each backbone.
    """
    bb_cfg = cfg["backbone"]
    num_bb: int = bb_cfg["num_backbones"]
    p_perturb: float = bb_cfg["p_perturb"]
    max_perturb: int = bb_cfg["max_perturb"]

    inlet_y = inlet[1]
    outlet_y = outlet[1]

    # Evenly space y-levels between inlet and outlet
    if num_bb >= 2:
        y_levels = [
            round(inlet_y + (outlet_y - inlet_y) * i / (num_bb - 1))
            for i in range(num_bb)
        ]
    else:
        y_levels = [inlet_y]

    grid = np.zeros((Ny, Nx), dtype=int)
    backbones: list[list[tuple[int, int]]] = []
    # Track actual endpoint y for each backbone (used for serpentine turns)
    endpoint_y: list[int] = []

    for i, y_level in enumerate(y_levels):
        # Determine direction: even = left→right, odd = right→left
        if i % 2 == 0:
            x_start, x_end = 0, Nx - 1
        else:
            x_start, x_end = Nx - 1, 0

        cells = _generate_single_backbone(
            x_start, y_level, x_end, Nx, Ny,
            p_perturb, max_perturb, rng, protection_cells,
        )
        backbones.append(cells)
        for x, y in cells:
            if _in_bounds(x, y, Nx, Ny):
                grid[y, x] = 1

        # Track the actual y at the endpoint (last cell should be at y_level
        # due to snap, but record it explicitly)
        actual_end_y = cells[-1][1] if cells else y_level
        endpoint_y.append(actual_end_y)

        # Add vertical "turn" segment connecting this backbone to the next
        if i < num_bb - 1:
            next_y_nom = y_levels[i + 1]
            turn_x = x_end  # turn at the endpoint side of this backbone
            y_from = actual_end_y
            y_to = next_y_nom

            y_step = 1 if y_to > y_from else -1
            for vy in range(y_from + y_step, y_to + y_step, y_step):
                if _in_bounds(turn_x, vy, Nx, Ny):
                    grid[vy, turn_x] = 1

    return grid, backbones, y_levels


# ── Stage 2: Vertical connectors ─────────────────────────────────────────────

def _add_connectors(
    grid: np.ndarray,
    y_levels: list[int],
    backbones: list[list[tuple[int, int]]],
    cfg: dict[str, Any],
    Nx: int,
    Ny: int,
    rng: random.Random,
    protection_cells: set[tuple[int, int]],
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    """Stage 2: Add vertical connectors between adjacent horizontal backbones.

    Returns
    -------
    grid:
        Updated grid; connector cells have value 2.
    connectors:
        List of cell lists for each placed connector (cells between backbones,
        not including the backbone cells themselves).
    """
    conn_cfg = cfg["connectors"]
    density: float = conn_cfg["density"]
    min_length: int = conn_cfg.get("min_length", 2)

    # Build lookup: backbone_y_at_x[i][x] = actual y of backbone i at column x
    backbone_y_at_x: list[dict[int, int]] = []
    for cells in backbones:
        y_at_x: dict[int, int] = {}
        for x, y in cells:
            y_at_x[x] = y
        backbone_y_at_x.append(y_at_x)

    connectors: list[list[tuple[int, int]]] = []

    for pair_idx in range(len(y_levels) - 1):
        y_top_nom = y_levels[pair_idx]
        y_bot_nom = y_levels[pair_idx + 1]

        for x in range(1, Nx - 1):  # skip edges (used for serpentine turns)
            if rng.random() >= density:
                continue

            # Get actual backbone y at this column (fallback to nominal)
            actual_y_top = backbone_y_at_x[pair_idx].get(x, y_top_nom)
            actual_y_bot = backbone_y_at_x[pair_idx + 1].get(x, y_bot_nom)

            y_min = min(actual_y_top, actual_y_bot)
            y_max = max(actual_y_top, actual_y_bot)

            # Connector cells are strictly between the two backbone cells
            connector_cells: list[tuple[int, int]] = []
            for y in range(y_min + 1, y_max):
                if _in_bounds(x, y, Nx, Ny) and (x, y) not in protection_cells:
                    connector_cells.append((x, y))

            # Skip connectors that are too short or in the protection zone
            if len(connector_cells) < max(1, min_length - 2):
                continue

            # Skip if any column cell is in the protection zone
            if any((x, y) in protection_cells for y in range(y_min, y_max + 1)):
                continue

            connectors.append(connector_cells)
            for cx, cy in connector_cells:
                if grid[cy, cx] == 0:  # don't overwrite backbone cells
                    grid[cy, cx] = 2

    return grid, connectors


# ── Stage 3: Pruning ──────────────────────────────────────────────────────────

def _prune_connections(
    grid: np.ndarray,
    connectors: list[list[tuple[int, int]]],
    cfg: dict[str, Any],
    Nx: int,
    Ny: int,
    inlet: tuple[int, int],
    outlet: tuple[int, int],
    rng: random.Random,
) -> np.ndarray:
    """Stage 3: Randomly prune connectors while maintaining connectivity.

    Each connector is considered for removal in a random order.
    A connector is removed only if inlet→outlet connectivity is preserved.
    """
    p_prune: float = cfg["connectors"]["p_prune"]

    # Shuffle for random pruning order
    indices = list(range(len(connectors)))
    rng.shuffle(indices)

    for idx in indices:
        if rng.random() >= p_prune:
            continue

        cells = connectors[idx]

        # Temporarily remove this connector (only cells that are still value 2)
        removed: list[tuple[int, int]] = []
        for x, y in cells:
            if grid[y, x] == 2:
                grid[y, x] = 0
                removed.append((x, y))

        if not removed:
            continue

        # Check connectivity
        reachable = _reachable_from(inlet, grid, Nx, Ny)
        if outlet not in reachable:
            # Restore: removing this connector breaks connectivity
            for x, y in removed:
                grid[y, x] = 2

    return grid


# ── Stage 4: Ensure global connectivity ──────────────────────────────────────

def _ensure_connectivity(
    grid: np.ndarray,
    inlet: tuple[int, int],
    outlet: tuple[int, int],
    Nx: int,
    Ny: int,
) -> np.ndarray:
    """Stage 4: Remove cells unreachable from inlet; verify outlet is reachable.

    Also ensures inlet and outlet cells are always present.
    """
    reachable = _reachable_from(inlet, grid, Nx, Ny)
    result = grid.copy()

    for y in range(Ny):
        for x in range(Nx):
            if result[y, x] != 0 and (x, y) not in reachable:
                result[y, x] = 0

    # Always keep inlet and outlet as pipe cells
    result[inlet[1], inlet[0]] = 1
    result[outlet[1], outlet[0]] = 1

    # If outlet is not reachable, force a BFS path from inlet to outlet
    reachable2 = _reachable_from(inlet, result, Nx, Ny)
    if outlet not in reachable2:
        path = _bfs_path(inlet, outlet, Nx, Ny)
        for x, y in path:
            result[y, x] = 1

    return result


# ── Stage 5: Coverage check ───────────────────────────────────────────────────

def _fill_subregion(
    grid: np.ndarray,
    x_start: int,
    y_start: int,
    x_end: int,
    y_end: int,
    Nx: int,
    Ny: int,
) -> None:
    """Add a vertical connector through an empty subregion and connect to grid.

    Finds the nearest pipe cell above and below in the center column,
    then fills the gap.  The fill cells are marked as value 2 (connector).
    Modifies *grid* in-place.
    """
    mid_x = (x_start + x_end) // 2

    # Search upward for nearest pipe cell
    y_above: int | None = None
    for y in range(y_start - 1, -1, -1):
        if grid[y, mid_x] != 0:
            y_above = y
            break

    # Search downward for nearest pipe cell
    y_below: int | None = None
    for y in range(y_end, Ny):
        if grid[y, mid_x] != 0:
            y_below = y
            break

    # Fill between nearest pipe cells, passing through the subregion
    if y_above is not None and y_below is not None:
        for y in range(y_above + 1, y_below):
            if _in_bounds(mid_x, y, Nx, Ny) and grid[y, mid_x] == 0:
                grid[y, mid_x] = 2
    elif y_above is not None:
        # Extend downward from nearest pipe cell to bottom of subregion
        for y in range(y_above + 1, y_end):
            if _in_bounds(mid_x, y, Nx, Ny) and grid[y, mid_x] == 0:
                grid[y, mid_x] = 2
    elif y_below is not None:
        # Extend upward from nearest pipe cell to top of subregion
        for y in range(y_start, y_below):
            if _in_bounds(mid_x, y, Nx, Ny) and grid[y, mid_x] == 0:
                grid[y, mid_x] = 2
    else:
        # No pipe cells in column at all: try horizontal connection
        mid_y = (y_start + y_end) // 2
        # Find nearest pipe cell to the left
        for x in range(mid_x - 1, -1, -1):
            if _in_bounds(x, mid_y, Nx, Ny) and grid[mid_y, x] != 0:
                for fill_x in range(x + 1, mid_x + 1):
                    if grid[mid_y, fill_x] == 0:
                        grid[mid_y, fill_x] = 2
                break


def _check_coverage(
    grid: np.ndarray,
    cfg: dict[str, Any],
    Nx: int,
    Ny: int,
    inlet: tuple[int, int],
    outlet: tuple[int, int],
    rng: random.Random,
) -> np.ndarray:
    """Stage 5: Ensure subregion coverage and minimum overall coverage.

    Divides the grid into ``subregion_cols × subregion_rows`` blocks.
    Each block must contain at least one pipe cell.  If a block is empty,
    a vertical connector stub is added to fill it.

    Also checks overall coverage against ``min_coverage`` and adds more
    connectors if needed.
    """
    cov_cfg = cfg["coverage"]
    sub_cols: int = cov_cfg["subregion_cols"]
    sub_rows: int = cov_cfg["subregion_rows"]
    min_coverage: float = cov_cfg["min_coverage"]

    sub_width = max(1, Nx // sub_cols)
    sub_height = max(1, Ny // sub_rows)

    # ── Per-subregion check ───────────────────────────────────────────────
    for sr in range(sub_rows):
        for sc in range(sub_cols):
            x0 = sc * sub_width
            x1 = min(x0 + sub_width, Nx)
            y0 = sr * sub_height
            y1 = min(y0 + sub_height, Ny)

            has_cell = bool(np.any(grid[y0:y1, x0:x1] != 0))
            if not has_cell:
                _fill_subregion(grid, x0, y0, x1, y1, Nx, Ny)

    # ── Overall coverage check ────────────────────────────────────────────
    pipe_cells = int(np.sum(grid != 0))
    total_cells = Nx * Ny
    coverage = pipe_cells / total_cells

    if coverage < min_coverage:
        target = int(min_coverage * total_cells)
        # Add connectors by extending existing pipe cells into empty neighbors
        # Iterate in a fixed random order to be reproducible
        candidates: list[tuple[int, int]] = []
        for y in range(1, Ny - 1):
            for x in range(1, Nx - 1):
                if grid[y, x] == 0:
                    has_nb = any(
                        _in_bounds(x + dx, y + dy, Nx, Ny)
                        and grid[y + dy, x + dx] != 0
                        for dx, dy in _ALL_DIRS
                    )
                    if has_nb:
                        candidates.append((x, y))

        rng.shuffle(candidates)
        for cx, cy in candidates:
            if pipe_cells >= target:
                break
            if grid[cy, cx] == 0:
                grid[cy, cx] = 2
                pipe_cells += 1

    return grid


# ── Stage 6: Manufacturing validation ────────────────────────────────────────

def _validate_manufacturing(
    grid: np.ndarray,
    cfg: dict[str, Any],
    Nx: int,
    Ny: int,
) -> tuple[bool, list[str]]:
    """Stage 6: Check manufacturing constraints.

    Checks:
    * Minimum spacing between SEPARATE parallel horizontal channels
      (two different backbones whose rows are too close).
      A single backbone that perturbs across adjacent rows is NOT flagged.
    * Excessive consecutive junction density along backbone rows.

    Returns
    -------
    (all_ok, warnings)
        *all_ok* is ``True`` if no constraint is violated.
        *warnings* is a list of human-readable warning strings.
    """
    mfg_cfg = cfg.get("manufacturing", {})
    min_spacing: int = mfg_cfg.get("min_spacing", 2)
    max_turns: int = mfg_cfg.get("max_consecutive_turns", 3)

    warnings_list: list[str] = []

    # ── Identify "horizontal channel" rows ────────────────────────────────
    # A row qualifies if it has a run of ≥ 3 adjacent pipe cells.
    def _is_horizontal_channel_row(y: int) -> bool:
        run = 0
        for x in range(Nx):
            if grid[y, x] != 0:
                run += 1
                if run >= 3:
                    return True
            else:
                run = 0
        return False

    horizontal_rows = [y for y in range(Ny) if _is_horizontal_channel_row(y)]

    # ── Spacing check ─────────────────────────────────────────────────────
    # Two horizontal-channel rows are "separate parallel channels" only if
    # the pipe cells in those two rows are NOT part of the same connected
    # component (i.e., they belong to different backbones).
    for i in range(len(horizontal_rows) - 1):
        y1 = horizontal_rows[i]
        y2 = horizontal_rows[i + 1]
        gap = y2 - y1
        if gap >= min_spacing:
            continue

        # Find any pipe cell in y1 and any pipe cell in y2
        cell_y1: tuple[int, int] | None = None
        cell_y2: tuple[int, int] | None = None
        for x in range(Nx):
            if grid[y1, x] != 0 and cell_y1 is None:
                cell_y1 = (x, y1)
            if grid[y2, x] != 0 and cell_y2 is None:
                cell_y2 = (x, y2)

        if cell_y1 is None or cell_y2 is None:
            continue

        # Check connectivity: if y1 and y2 cells are in the same component,
        # they belong to the same (perturbed) backbone → skip
        path = _bfs_path(cell_y1, cell_y2, Nx, Ny, grid)
        if path:
            # Same connected component (same backbone, just perturbed)
            continue

        warnings_list.append(
            f"Separate horizontal channels at rows {y1} and {y2} "
            f"have spacing {gap} < {min_spacing}"
        )

    # ── Consecutive-turns check ────────────────────────────────────────────
    for y in horizontal_rows:
        consecutive = 0
        for x in range(1, Nx - 1):
            if grid[y, x] == 0:
                consecutive = 0
                continue
            has_vert_nb = any(
                _in_bounds(x, y + dy, Nx, Ny) and grid[y + dy, x] != 0
                for dy in (-1, 1)
            )
            if has_vert_nb:
                consecutive += 1
                if consecutive > max_turns:
                    warnings_list.append(
                        f"Excessive branch density at row {y} around x={x}: "
                        f"{consecutive} consecutive junction points > {max_turns}"
                    )
                    break
            else:
                consecutive = 0

    return len(warnings_list) == 0, warnings_list


# ── Dead-end removal ─────────────────────────────────────────────────────────

def _remove_dead_ends(
    grid: np.ndarray,
    protected: set[tuple[int, int]],
    Nx: int,
    Ny: int,
) -> np.ndarray:
    """Iteratively remove degree-1 pipe cells that are not in *protected*.

    A degree-1 pipe cell has exactly one pipe-cell neighbour (it is a dead
    end).  Removal is repeated until no more dead ends remain.
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
    """Dilate the pipe mask so each centre-line cell expands by *width* cells.

    Only used for the visualisation grid; topology data comes from the
    undilated grid.
    """
    if width <= 1:
        return grid
    from scipy.ndimage import binary_dilation  # type: ignore

    struct = np.ones((width, width), dtype=bool)
    mask = binary_dilation(grid != 0, structure=struct)
    result = np.where(mask, np.where(grid != 0, grid, 2), 0).astype(int)
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def generate_network(
    cfg: dict[str, Any],
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Generate a single EV battery cooling channel network topology.

    Parameters
    ----------
    cfg:
        Loaded configuration dictionary (from :func:`config_loader.load_config`).
    rng:
        Optional seeded :class:`random.Random` instance.

    Returns
    -------
    dict with keys:

    * ``"grid"``                  – ``np.ndarray`` ``(Ny, Nx)`` int (undilated)
    * ``"grid_vis"``              – ``np.ndarray`` ``(Ny, Nx)`` int (dilated)
    * ``"main_path"``             – list of ``(x, y)`` backbone cells
    * ``"inlet"``                 – ``(x, y)``
    * ``"outlet"``                – ``(x, y)``
    * ``"Nx"``, ``"Ny"``
    * ``"coverage"``              – fraction of cells that are pipe cells
    * ``"manufacturing_ok"``      – bool, True if all constraints satisfied
    * ``"manufacturing_warnings"``– list of warning strings
    * ``"protection_cells"``      – set of cells in the protection zone
    """
    if rng is None:
        rng = random.Random()

    Nx: int = cfg["grid"]["Nx"]
    Ny: int = cfg["grid"]["Ny"]

    inlet = resolve_port(cfg["inlet"]["wall"], cfg["inlet"]["pos"], Nx, Ny)
    outlet = resolve_port(cfg["outlet"]["wall"], cfg["outlet"]["pos"], Nx, Ny)

    protection_radius: int = cfg.get("protection_radius", 4)
    protection_cells = _protection_zone(inlet, outlet, protection_radius, Nx, Ny)

    # ── Stage 1: Backbone generation ─────────────────────────────────────
    grid, backbones, y_levels = _generate_backbones(
        cfg, Nx, Ny, inlet, outlet, rng, protection_cells
    )

    # ── Stage 2: Vertical connectors ─────────────────────────────────────
    grid, connectors = _add_connectors(
        grid, y_levels, backbones, cfg, Nx, Ny, rng, protection_cells
    )

    # ── Stage 3: Pruning (while maintaining connectivity) ─────────────────
    grid = _prune_connections(
        grid, connectors, cfg, Nx, Ny, inlet, outlet, rng
    )

    # ── Stage 4: Ensure global connectivity ──────────────────────────────
    grid = _ensure_connectivity(grid, inlet, outlet, Nx, Ny)

    # ── Optional: remove dead ends before coverage fill ───────────────────
    allow_dead_ends: bool = cfg.get("allow_dead_ends", True)
    if not allow_dead_ends:
        protected_nodes = {inlet, outlet}
        grid = _remove_dead_ends(grid, protected_nodes, Nx, Ny)
        # Re-ensure inlet/outlet after dead-end removal
        grid[inlet[1], inlet[0]] = 1
        grid[outlet[1], outlet[0]] = 1

    # ── Stage 5: Coverage check (runs after dead-end removal) ─────────────
    # Coverage stubs added here are NOT removed (dead-end removal already ran)
    grid = _check_coverage(grid, cfg, Nx, Ny, inlet, outlet, rng)

    # Re-verify connectivity after coverage fill
    grid = _ensure_connectivity(grid, inlet, outlet, Nx, Ny)

    # ── Stage 6: Manufacturing validation ────────────────────────────────
    mfg_ok, mfg_warnings = _validate_manufacturing(grid, cfg, Nx, Ny)

    # ── Compute final statistics ──────────────────────────────────────────
    coverage = float(np.sum(grid != 0)) / (Nx * Ny)

    # ── Optional pipe dilation for visualisation ─────────────────────────
    pipe_width: int = cfg.get("pipe_width", 1)
    grid_vis = _dilate(grid, pipe_width, Nx, Ny) if pipe_width > 1 else grid.copy()

    # Flatten backbone cells as "main_path" for backward compatibility
    main_path = [
        cell
        for backbone in backbones
        for cell in backbone
        if grid[cell[1], cell[0]] != 0
    ]

    return {
        "grid": grid,
        "grid_vis": grid_vis,
        "main_path": main_path,
        "inlet": inlet,
        "outlet": outlet,
        "Nx": Nx,
        "Ny": Ny,
        "coverage": coverage,
        "manufacturing_ok": mfg_ok,
        "manufacturing_warnings": mfg_warnings,
        "protection_cells": protection_cells,
    }

