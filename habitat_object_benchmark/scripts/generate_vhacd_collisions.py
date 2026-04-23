#!/usr/bin/env python3
"""Generate VHACD collision GLBs for all assets in the filtered manifest.

For each asset, runs CoACD on the render mesh and exports the convex parts
as a multi-primitive GLB. Habitat-sim loads each primitive as a separate
convex hull (compound shape) when join_collision_meshes=false.

Output: <extracted-dir>/<name>.vhacd.glb  (alongside existing .collision.glb)

Usage:
    python -m habitat_object_benchmark.scripts.generate_vhacd_collisions \
        --manifest   data/datasets/amara-spatial-10k/filtered_manifest.parquet \
        --extracted-dir data/datasets/amara-spatial-10k/extracted \
        --workers 8 \
        [--max-parts 32] \
        [--threshold 0.05] \
        [--limit 20]
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ── Per-asset worker ─────────────────────────────────────────────────────────

def _process_one(args: Tuple) -> dict:
    render_path, out_path, max_parts, threshold = args

    result = {"path": str(render_path), "parts": None, "error": None, "skipped": False}

    if out_path.exists():
        result["skipped"] = True
        return result

    try:
        import coacd
        import trimesh

        # Load render mesh
        scene = trimesh.load(str(render_path), force="scene")
        if isinstance(scene, trimesh.Scene):
            meshes = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh)]
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = scene

        if len(mesh.faces) == 0:
            result["error"] = "empty mesh"
            return result

        # Run CoACD — suppress C++ spdlog output by redirecting stderr
        coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stdout, saved_stderr = os.dup(1), os.dup(2)
        os.dup2(devnull_fd, 1); os.dup2(devnull_fd, 2)
        try:
            parts = coacd.run_coacd(
                coacd_mesh,
                threshold=threshold,
                max_convex_hull=max_parts,
                preprocess_mode="auto",
            )
        finally:
            os.dup2(saved_stdout, 1); os.dup2(saved_stderr, 2)
            os.close(saved_stdout); os.close(saved_stderr); os.close(devnull_fd)

        # Export as multi-primitive GLB: one trimesh geometry per part
        export_scene = trimesh.Scene()
        for i, (verts, faces) in enumerate(parts):
            part_mesh = trimesh.Trimesh(
                vertices=np.array(verts, dtype=np.float32),
                faces=np.array(faces, dtype=np.int32),
                process=False,
            )
            export_scene.add_geometry(part_mesh, node_name=f"part_{i:03d}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_scene.export(str(out_path))
        result["parts"] = len(parts)

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate VHACD collision GLBs")
    parser.add_argument("--manifest",      required=True, type=Path,
                        help="filtered_manifest.parquet from filter_assets.py")
    parser.add_argument("--extracted-dir", required=True, type=Path,
                        help="Directory containing extracted render GLBs")
    parser.add_argument("--workers",       type=int, default=4)
    parser.add_argument("--max-parts",     type=int, default=32,
                        help="Max convex parts per asset (CoACD max_convex_hull)")
    parser.add_argument("--threshold",     type=float, default=0.05,
                        help="CoACD concavity threshold (lower = more parts, more accurate)")
    parser.add_argument("--limit",         type=int, default=None,
                        help="Process only first N assets (for testing)")
    args = parser.parse_args()

    df = pd.read_parquet(args.manifest)
    df = df[df["subset"] != "excluded"].reset_index(drop=True)
    if args.limit:
        df = df.iloc[:args.limit]
        print(f"(--limit {args.limit}: processing first {len(df)} assets)")

    # Build work list
    tasks: List[Tuple] = []
    missing = []
    for _, row in df.iterrows():
        render_glb = args.extracted_dir / row["mesh_path"]
        if not render_glb.exists():
            missing.append(row["asset_id"])
            continue
        out_path = render_glb.with_suffix("").with_suffix(".vhacd.glb")
        tasks.append((render_glb, out_path, args.max_parts, args.threshold))

    if missing:
        print(f"Warning: {len(missing)} assets have missing render GLBs — skipped.")

    already_done = sum(1 for _, out, *_ in tasks if out.exists())
    todo_tasks = [(r, o, mp, th) for r, o, mp, th in tasks if not o.exists()]
    print(f"{len(tasks)} assets total — {already_done} already done, {len(todo_tasks)} to process")
    print(f"Workers: {args.workers}  max-parts: {args.max_parts}  threshold: {args.threshold}")

    if not todo_tasks:
        print("Nothing to do.")
        return

    # Silence CoACD logs in workers
    os.environ.setdefault("COACD_LOG_LEVEL", "warning")

    errors, skipped, processed, total_parts = [], 0, 0, 0

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        for r in tqdm(pool.imap_unordered(_process_one, todo_tasks), total=len(todo_tasks)):
            if r["error"]:
                errors.append((r["path"], r["error"]))
            else:
                processed += 1
                total_parts += r["parts"]

    print(f"\nDone: {processed} generated, {len(errors)} errors")
    if processed:
        print(f"Average parts per asset: {total_parts / processed:.1f}")
    if errors:
        print(f"\nFirst 10 errors:")
        for path, err in errors[:10]:
            print(f"  {Path(path).stem}: {err}")


if __name__ == "__main__":
    main()
