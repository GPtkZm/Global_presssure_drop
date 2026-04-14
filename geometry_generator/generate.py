"""
generate.py
===========
Main entry point and CLI for the geometry_generator package.

Usage
-----
    python -m geometry_generator [OPTIONS]

    Options:
      --config PATH       Path to YAML config file
                          (default: geometry_generator/config.yaml)
      --num_samples INT   Override num_samples from config
      --seed INT          Override random seed from config
      --output_dir PATH   Override output directory from config
      --no_summary        Skip generating the summary montage image
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="geometry_generator",
        description="Procedurally generate pipe/channel network geometries.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to the YAML configuration file. "
            "Defaults to geometry_generator/config.yaml."
        ),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of random topologies to generate (overrides config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global random seed for reproducibility (overrides config).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root output directory (overrides config).",
    )
    parser.add_argument(
        "--no_summary",
        action="store_true",
        help="Skip generating the summary montage image.",
    )
    return parser.parse_args(argv)


def _apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Mutate *cfg* to apply command-line overrides."""
    if args.num_samples is not None:
        cfg["num_samples"] = args.num_samples
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.output_dir is not None:
        cfg["output"]["dir"] = args.output_dir


def _make_rng(cfg: dict[str, Any], sample_index: int) -> random.Random:
    """Create a seeded :class:`random.Random` for *sample_index*.

    If ``cfg["seed"]`` is *None*, uses the system clock (non-deterministic).
    Otherwise uses ``seed + sample_index`` so each sample is distinct but
    the whole batch is reproducible.
    """
    base_seed = cfg.get("seed")
    if base_seed is None:
        return random.Random()
    return random.Random(base_seed + sample_index)


def run(cfg: dict[str, Any], *, skip_summary: bool = False) -> None:
    """Execute batch generation using the provided config dictionary.

    Parameters
    ----------
    cfg:
        Validated configuration dictionary.
    skip_summary:
        If *True*, the summary montage is not generated even if
        ``cfg["output"]["save_summary"]`` is *True*.
    """
    # Lazy imports so the module loads quickly for ``--help``
    from geometry_generator.network import generate_network
    from geometry_generator.graph import build_graph
    from geometry_generator.visualize import visualize_network, build_summary_image

    out_cfg = cfg["output"]
    output_dir = Path(out_cfg["dir"])
    num_samples: int = cfg["num_samples"]

    image_paths: list[Path] = []
    sample_ids: list[str] = []

    print(f"[geometry_generator] Generating {num_samples} network(s)…")
    print(f"[geometry_generator] Output directory: {output_dir.resolve()}")

    t0 = time.time()

    for i in range(num_samples):
        sample_id = f"network_{i + 1:04d}"
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        rng = _make_rng(cfg, i)

        # ── Generate network ──────────────────────────────────────────────
        network = generate_network(cfg, rng=rng)

        # ── Save grid.npy ─────────────────────────────────────────────────
        if out_cfg.get("save_grid", True):
            np.save(sample_dir / "grid.npy", network["grid"])

        # ── Save graph.json ───────────────────────────────────────────────
        if out_cfg.get("save_graph", True):
            graph = build_graph(network)
            with open(sample_dir / "graph.json", "w", encoding="utf-8") as fh:
                json.dump(graph, fh, separators=(",", ":"))

        # ── Save preview.png ──────────────────────────────────────────────
        preview_path = sample_dir / "preview.png"
        if out_cfg.get("save_image", True):
            visualize_network(network, sample_id, cfg, preview_path)
            image_paths.append(preview_path)
            sample_ids.append(sample_id)

        elapsed = time.time() - t0
        print(
            f"  [{i + 1:>{len(str(num_samples))}}/{num_samples}] "
            f"{sample_id}  —  "
            f"nodes={sum(1 for row in network['grid'] for v in row if v != 0)}  "
            f"({elapsed:.1f}s elapsed)"
        )

    # ── Summary montage ───────────────────────────────────────────────────
    if (
        not skip_summary
        and out_cfg.get("save_summary", True)
        and image_paths
    ):
        summary_path = output_dir / "summary.png"
        print(f"[geometry_generator] Building summary image → {summary_path}")
        build_summary_image(image_paths, sample_ids, cfg, summary_path)

    total = time.time() - t0
    print(f"[geometry_generator] Done in {total:.1f}s.  Output: {output_dir.resolve()}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)

    # Resolve config path
    if args.config is not None:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).parent / "config.yaml"

    # Import here to keep startup fast
    from geometry_generator.config_loader import load_config

    try:
        cfg = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[geometry_generator] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    _apply_overrides(cfg, args)

    run(cfg, skip_summary=args.no_summary)


if __name__ == "__main__":
    main()
