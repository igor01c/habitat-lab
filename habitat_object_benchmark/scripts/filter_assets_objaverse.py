#!/usr/bin/env python3
"""Filter Objaverse assets by category and bounding-box size.

Equivalent to filter_assets.py for the Amara dataset, but adapted for the
Objaverse metadata format (metadata.json + per-asset .glb files).

Outputs a filtered_manifest.parquet with columns:
    asset_id, name, categories, max_dim, subset (manipulation|obstacle|excluded)

Usage:
    python -m habitat_object_benchmark.scripts.filter_assets_objaverse \
        --dataset-dir data/datasets/objaverse \
        --out         data/datasets/objaverse/filtered_manifest.parquet \
        [--workers 8]
"""

import argparse
import json
import multiprocessing as mp
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Objaverse categories that map to Amara's indoor-robotics whitelist:
#   furniture-home  → Kitchen, Household Items, Bathroom, Living Room, Bedroom,
#                     Dining, Study Room, Domestic Furniture, Hotel, Library,
#                     Bookstore, Classroom, Pub, Wine Cellar, Spa, Workshop
#   food-drink      → Food, Food and Fruits, Restaurant, Coffee shop
#   music           → Musical Instruments
#   electronics-gadgets → Workshop / Study Room items
#   sports-fitness  → Toys and Games
CATEGORY_WHITELIST = {
    "furniture-home",
    "food-drink",
    "music",
    "electronics-gadgets",
    "sports-fitness",
}

MANIPULATION_MAX_DIM = 0.5   # metres — same as Amara
OBSTACLE_MAX_DIM     = 2.0   # metres — same as Amara
MAX_FILE_MB          = 50.0  # exclude heavy photogrammetry/scan GLBs

_MESHES_DIR: Path = None  # set in worker init


def _worker_init(meshes_dir: str):
    global _MESHES_DIR
    _MESHES_DIR = Path(meshes_dir)


def _compute_max_dim(asset_id: str):
    """Load GLB, return (asset_id, max_dim) or (asset_id, None) on error."""
    import trimesh
    glb = _MESHES_DIR / f"{asset_id}.glb"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scene = trimesh.load(str(glb), force="scene")
        bounds = scene.bounds  # shape (2, 3)
        if bounds is None:
            return asset_id, None
        max_dim = float((bounds[1] - bounds[0]).max())
        return asset_id, max_dim
    except Exception as e:
        return asset_id, None


def classify_subset(max_dim, file_mb, in_whitelist: bool) -> str:
    if not in_whitelist or max_dim is None or max_dim > OBSTACLE_MAX_DIM:
        return "excluded"
    if file_mb is not None and file_mb > MAX_FILE_MB:
        return "excluded"
    if max_dim <= MANIPULATION_MAX_DIM:
        return "manipulation"
    return "obstacle"


def main():
    parser = argparse.ArgumentParser(
        description="Filter Objaverse assets by category and bounding-box size"
    )
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    meshes_dir = args.dataset_dir / "meshes"
    meta_path  = args.dataset_dir / "metadata.json"

    print(f"Loading metadata from {meta_path}")
    with open(meta_path) as f:
        metadata = json.load(f)

    rows = []
    for asset_id, info in metadata.items():
        cats = info.get("categories", [])
        in_whitelist = bool(set(cats) & CATEGORY_WHITELIST)
        rows.append({
            "asset_id":    asset_id,
            "name":        info.get("name", ""),
            "categories":  cats,
            "in_whitelist": in_whitelist,
        })
    df = pd.DataFrame(rows)

    total = len(df)
    whitelisted = df["in_whitelist"].sum()
    print(f"Total assets:       {total}")
    print(f"Category-whitelisted: {whitelisted} ({100*whitelisted/total:.1f}%)")
    print(f"Computing bounding boxes with {args.workers} workers...")

    ctx = mp.get_context("spawn")
    asset_ids = df["asset_id"].tolist()
    max_dims = {}

    with ctx.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(str(meshes_dir),),
    ) as pool:
        for asset_id, max_dim in tqdm(
            pool.imap_unordered(_compute_max_dim, asset_ids, chunksize=50),
            total=len(asset_ids),
            desc="AABB",
        ):
            max_dims[asset_id] = max_dim

    df["max_dim"] = df["asset_id"].map(max_dims)
    df["file_mb"] = df["asset_id"].apply(
        lambda aid: (meshes_dir / f"{aid}.glb").stat().st_size / 1e6
        if (meshes_dir / f"{aid}.glb").exists() else None
    )
    df["subset"]  = df.apply(
        lambda r: classify_subset(r["max_dim"], r["file_mb"], r["in_whitelist"]), axis=1
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    kept = df[df["subset"] != "excluded"]
    manip = (df["subset"] == "manipulation").sum()
    obs   = (df["subset"] == "obstacle").sum()
    errs  = df["max_dim"].isna().sum()
    print(f"\nResults:")
    print(f"  Excluded:            {(df['subset'] == 'excluded').sum()}")
    print(f"  Kept:                {len(kept)}")
    print(f"    manipulation (<{MANIPULATION_MAX_DIM}m): {manip}")
    print(f"    obstacle     (<{OBSTACLE_MAX_DIM}m):  {obs}")
    print(f"  AABB errors:         {errs}")
    heavy = df[df["file_mb"] > MAX_FILE_MB]["file_mb"].count()
    print(f"  Excluded (>{MAX_FILE_MB:.0f} MB):   {heavy}")

    print("\nPer-category breakdown (kept only):")
    cat_counts = {}
    for _, row in kept.iterrows():
        for cat in row["categories"]:
            if cat in CATEGORY_WHITELIST:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:5d}  {cat}")

    print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()
