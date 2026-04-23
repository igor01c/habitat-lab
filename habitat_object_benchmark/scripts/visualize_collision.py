#!/usr/bin/env python3
"""Visualize convex hull vs VHACD collision decomposition side by side.

Usage:
    python -m habitat_object_benchmark.scripts.visualize_collision \
        --mesh data/datasets/amara-spatial-10k/extracted/<name>.glb \
        [--out /tmp/collision_comparison.png] \
        [--vhacd-parts 32]
"""

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import coacd
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ── Mesh loading ─────────────────────────────────────────────────────────────

def load_mesh(path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(str(path), force="scene")
    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene
    return mesh


# ── Collision generation ─────────────────────────────────────────────────────

def make_convex_hull(mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
    return [mesh.convex_hull]


def make_vhacd(mesh: trimesh.Trimesh, max_convex_hull: int = 32) -> List[trimesh.Trimesh]:
    import os
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout, saved_stderr = os.dup(1), os.dup(2)
    os.dup2(devnull_fd, 1); os.dup2(devnull_fd, 2)
    try:
        parts = coacd.run_coacd(coacd_mesh, max_convex_hull=max_convex_hull)
    finally:
        os.dup2(saved_stdout, 1); os.dup2(saved_stderr, 2)
        os.close(saved_stdout); os.close(saved_stderr); os.close(devnull_fd)
    return [trimesh.Trimesh(vertices=np.array(v), faces=np.array(f)) for v, f in parts]


# ── Rendering ────────────────────────────────────────────────────────────────

# Distinct colours for VHACD parts
PART_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4",
    "#8bc34a", "#ff5722", "#607d8b", "#795548", "#ffc107",
    "#673ab7", "#009688", "#ff9800", "#03a9f4", "#cddc39",
]


def _faces_to_triangles(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return (N, 3, 3) array of triangle vertex positions."""
    return mesh.vertices[mesh.faces]


def render_meshes(ax, meshes: List[trimesh.Trimesh], title: str,
                  single_color: Optional[str] = None, alpha: float = 0.55):
    all_verts = np.vstack([m.vertices for m in meshes])
    cx, cy, cz = all_verts.mean(axis=0)
    span = (all_verts.max(axis=0) - all_verts.min(axis=0)).max() / 2 * 1.1

    for i, mesh in enumerate(meshes):
        color = single_color if single_color else PART_COLORS[i % len(PART_COLORS)]
        tris = _faces_to_triangles(mesh)
        poly = Poly3DCollection(tris, alpha=alpha, linewidths=0.1)
        poly.set_facecolor(color)
        poly.set_edgecolor("#00000033")
        ax.add_collection3d(poly)

    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_zlim(cz - span, cz + span)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_box_aspect([1, 1, 1])


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize convex hull vs VHACD collision shapes")
    parser.add_argument("--mesh",        required=True, type=Path, help="Input GLB/OBJ mesh")
    parser.add_argument("--out",         type=Path, default=None,  help="Output PNG path (default: show interactively)")
    parser.add_argument("--vhacd-parts", type=int, default=32,     help="Max convex parts for VHACD")
    parser.add_argument("--elev",        type=float, default=25,   help="Camera elevation angle")
    parser.add_argument("--azim",        type=float, default=45,   help="Camera azimuth angle")
    args = parser.parse_args()

    print(f"Loading {args.mesh.name}...")
    mesh = load_mesh(args.mesh)
    print(f"  {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")

    print("Computing convex hull...")
    hull_parts = make_convex_hull(mesh)
    print(f"  1 part, {len(hull_parts[0].vertices)} vertices")

    print(f"Running VHACD (max {args.vhacd_parts} parts)...")
    vhacd_parts = make_vhacd(mesh, max_convex_hull=args.vhacd_parts)
    print(f"  {len(vhacd_parts)} parts, "
          f"{sum(len(p.vertices) for p in vhacd_parts)} total vertices")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 7), facecolor="#111111")
    fig.suptitle(args.mesh.stem, color="#dddddd", fontsize=10, y=0.98)

    # Subsample render mesh faces for display (matplotlib struggles with >5k faces)
    if len(mesh.faces) > 4000:
        idx = np.random.choice(len(mesh.faces), 4000, replace=False)
        display_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[idx], process=False)
    else:
        display_mesh = mesh

    panels = [
        ("Original mesh",                    [display_mesh], "#888888"),
        ("Convex hull (1 part)",             hull_parts,     "#3498db"),
        (f"VHACD ({len(vhacd_parts)} parts)", vhacd_parts,   None),
    ]

    for col, (title, parts, color) in enumerate(panels):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d",
                             facecolor="#1a1a1a")
        ax.tick_params(colors="#555555", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333333")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#2a2a2a")
        ax.yaxis.pane.set_edgecolor("#2a2a2a")
        ax.zaxis.pane.set_edgecolor("#2a2a2a")
        ax.view_init(elev=args.elev, azim=args.azim)
        render_meshes(ax, parts, title, single_color=color)
        ax.title.set_color("#cccccc")

    plt.tight_layout()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
