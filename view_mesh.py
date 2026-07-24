#!/usr/bin/env python3
"""View a mesh using trimesh.

Usage:
    python view_mesh.py --mesh path/to/mesh.ply

The script loads either a single mesh or a scene and opens the interactive
trimesh viewer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a mesh in the trimesh viewer")
    parser.add_argument("--mesh", required=True, help="Path to the mesh file")
    return parser.parse_args()


def load_geometry(path: Path):
    """Load a mesh or scene from disk."""
    loaded = trimesh.load(path)

    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"No geometry found in '{path}'")
        return loaded

    if isinstance(loaded, trimesh.Trimesh):
        if loaded.is_empty:
            raise ValueError(f"Mesh '{path}' is empty")
        return loaded

    raise TypeError(f"Unsupported geometry loaded from '{path}'")


def main() -> int:
    args = parse_args()
    mesh_path = Path(args.mesh)

    if not mesh_path.exists():
        print(f"Error: mesh file '{mesh_path}' not found", file=sys.stderr)
        return 1

    try:
        geometry = load_geometry(mesh_path)
        print(f"Loaded {mesh_path}")
        print("Opening viewer...")
        geometry.show()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())