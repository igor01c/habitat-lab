#!/usr/bin/env python3
"""Visualize render + collision + vhacd meshes for an amara-spatial-10k asset.

Opens an interactive Plotly 3D viewer showing all three meshes with their
bounding boxes and centres so you can compare alignment.

Usage:
    python3 habitat_object_benchmark/scripts/visualize_meshes.py \
        --config data/datasets/amara-spatial-10k/configs/<asset>.object_config.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import trimesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.geometry.values())
        if not meshes:
            raise ValueError(f"Empty scene: {path}")
        loaded = trimesh.util.concatenate(meshes)
    return loaded


def bbox_lines(bounds, color):
    """Return scatter3d traces for the 12 edges of an AABB."""
    mn, mx = bounds
    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ])
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # bottom face
        (4,5),(5,6),(6,7),(7,4),  # top face
        (0,4),(1,5),(2,6),(3,7),  # verticals
    ]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [corners[a][0], corners[b][0], None]
        ys += [corners[a][1], corners[b][1], None]
        zs += [corners[a][2], corners[b][2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                        line=dict(color=color, width=3), showlegend=False)


def mesh_trace(mesh: trimesh.Trimesh, name: str, color: str, opacity: float = 0.4):
    v = mesh.vertices
    f = mesh.faces
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color=color, opacity=opacity, name=name,
        showlegend=True,
    )


def centre_trace(centre, name, color, size=8):
    return go.Scatter3d(
        x=[centre[0]], y=[centre[1]], z=[centre[2]],
        mode="markers",
        marker=dict(size=size, color=color, symbol="cross"),
        name=f"{name} centre",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to .object_config.json")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    config_dir = args.config.parent

    render_path    = (config_dir / cfg["render_asset"]).resolve()
    collision_path = (config_dir / cfg["collision_asset"]).resolve()
    vhacd_path     = Path(str(render_path).replace(".glb", ".vhacd.glb"))

    import warnings
    warnings.filterwarnings("ignore")
    import logging
    logging.disable(logging.CRITICAL)

    print(f"Asset:     {args.config.stem}")
    print(f"Render:    {render_path.name}  ({render_path.stat().st_size // 1024} KB)")
    print(f"Collision: {collision_path.name}  ({collision_path.stat().st_size // 1024} KB)")
    if vhacd_path.exists():
        print(f"VHACD:     {vhacd_path.name}  ({vhacd_path.stat().st_size // 1024} KB)")
    else:
        print(f"VHACD:     not found")

    import os, sys
    devnull = open(os.devnull, "w")
    old_stderr = sys.stderr
    sys.stderr = devnull
    print("\nLoading meshes...")
    render_mesh    = load_mesh(render_path)
    collision_mesh = load_mesh(collision_path)
    sys.stderr = old_stderr

    def info(name, m):
        print(f"\n{name}:")
        print(f"  vertices: {len(m.vertices)}  faces: {len(m.faces)}")
        print(f"  bounds min: {m.bounds[0].round(4)}")
        print(f"  bounds max: {m.bounds[1].round(4)}")
        print(f"  extents:    {m.extents.round(4)}")
        print(f"  centre:     {m.centroid.round(4)}")

    info("Render mesh", render_mesh)
    info("Collision mesh", collision_mesh)

    traces = [
        mesh_trace(render_mesh,    "render",    "#3498db", 0.25),
        bbox_lines(render_mesh.bounds,    "#3498db"),
        centre_trace(render_mesh.centroid,    "render",    "#3498db"),
        mesh_trace(collision_mesh, "collision", "#e74c3c", 0.5),
        bbox_lines(collision_mesh.bounds, "#e74c3c"),
        centre_trace(collision_mesh.centroid, "collision", "#e74c3c"),
    ]

    if vhacd_path.exists():
        vhacd_mesh = load_mesh(vhacd_path)
        info("VHACD mesh", vhacd_mesh)
        traces += [
            mesh_trace(vhacd_mesh, "vhacd", "#2ecc71", 0.35),
            bbox_lines(vhacd_mesh.bounds, "#2ecc71"),
            centre_trace(vhacd_mesh.centroid, "vhacd", "#2ecc71"),
        ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=args.config.stem,
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
        ),
        legend=dict(x=0, y=1),
    )
    out = Path("/tmp/mesh_viz.html")
    fig.write_html(str(out))
    print(f"\nViewer saved to: {out}")
    print("Open it in your browser.")


if __name__ == "__main__":
    main()
