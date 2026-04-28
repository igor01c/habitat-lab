#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def build_config(render_glb: Path, collision_glb: Path, out_dir: Path, scale: float) -> Path:
    rel_render = Path(os.path.relpath(render_glb, out_dir))
    rel_collision = Path(os.path.relpath(collision_glb, out_dir))

    cfg = {
        "render_asset": str(rel_render),
        "collision_asset": str(rel_collision),
        "join_collision_meshes": False,
        "friction_coefficient": 0.5,
        "requires_lighting": True,
        "up": [0.0, 1.0, 0.0],
        "front": [0.0, 0.0, -1.0],
    }

    out_path = out_dir / (render_glb.stem + ".object_config.json")
    out_path.write_text(json.dumps(cfg, indent=2))
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate habitat-sim .object_config.json files from filtered manifest"
    )
    parser.add_argument("--manifest",       required=True, type=Path)
    parser.add_argument("--extracted-dir",  required=True, type=Path)
    parser.add_argument("--out-dir",        required=True, type=Path)
    parser.add_argument("--dry-run",        action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.manifest)
    df = df[df["subset"] != "excluded"].reset_index(drop=True)
    print(f"Generating configs for {len(df)} assets...")

    missing, written = [], 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        render_glb    = args.extracted_dir / row["mesh_path"]
        collision_glb = args.extracted_dir / row["collision_path"]

        if not render_glb.exists() or not collision_glb.exists():
            missing.append(row["asset_id"])
            continue

        # The GLB meshes are normalized (max_dim ≈ 1.0).
        # Real-world scale = max dimension from manifest aabb.
        aabb  = np.array(row["aabb"], dtype=float)
        scale = float(np.max(aabb))

        if args.dry_run:
            print(f"  {row['asset_id']}  scale={scale:.4f}")
        else:
            build_config(render_glb, collision_glb, args.out_dir, scale)
            written += 1

    if missing:
        print(f"\nWarning: {len(missing)} assets had missing GLBs — skipped.")
        (args.out_dir / "missing_glbs.txt").write_text("\n".join(missing))

    if not args.dry_run:
        print(f"Written {written} configs to {args.out_dir}")


if __name__ == "__main__":
    main()
