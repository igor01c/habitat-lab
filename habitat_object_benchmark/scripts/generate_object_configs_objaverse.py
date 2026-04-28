#!/usr/bin/env python3
"""Generate habitat-sim .object_config.json files for Objaverse assets.

Objaverse has a single GLB per asset (used as both render and collision mesh),
unlike Amara which ships separate render + collision GLBs.

Usage:
    python -m habitat_object_benchmark.scripts.generate_object_configs_objaverse \
        --manifest   data/datasets/objaverse/filtered_manifest.parquet \
        --meshes-dir data/datasets/objaverse/meshes \
        --out-dir    data/datasets/objaverse/configs
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def build_config(glb: Path, out_dir: Path) -> Path:
    rel = Path(os.path.relpath(glb.resolve(), out_dir.resolve()))
    cfg = {
        "render_asset":        str(rel),
        "collision_asset":     str(rel),
        "join_collision_meshes": False,
        "friction_coefficient": 0.5,
        "requires_lighting":   True,
        "up":    [0.0, 1.0, 0.0],
        "front": [0.0, 0.0, -1.0],
    }
    out_path = out_dir / (glb.stem + ".object_config.json")
    out_path.write_text(json.dumps(cfg, indent=2))
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate object configs for Objaverse assets"
    )
    parser.add_argument("--manifest",   required=True, type=Path)
    parser.add_argument("--meshes-dir", required=True, type=Path)
    parser.add_argument("--out-dir",    required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.manifest)
    df = df[df["subset"] != "excluded"].reset_index(drop=True)
    print(f"Generating configs for {len(df)} assets...")

    missing, written = [], 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        glb = args.meshes_dir / f"{row['asset_id']}.glb"
        if not glb.exists():
            missing.append(row["asset_id"])
            continue
        build_config(glb, args.out_dir)
        written += 1

    if missing:
        print(f"Warning: {len(missing)} GLBs not found — skipped.")
        (args.out_dir / "missing_glbs.txt").write_text("\n".join(missing))

    print(f"Written {written} configs to {args.out_dir}")


if __name__ == "__main__":
    main()
