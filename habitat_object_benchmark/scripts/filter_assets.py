#!/usr/bin/env python3

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORY_WHITELIST = {
    "Kitchen",
    "Houshold Items",
    "Food and Fruits",
    "Food",
    "Toys and Games",
    "Bathroom",
    "Study Room",
    "Domestic Furniture LP",
    "Living Room",
    "Bedroom",
    "Dining",
    "Restaurant",
    "Coffee shop",
    "Medical",
    "Wine Cellar",
    "Workshop",
    "Musical Instruments",
    "Bookstore",
    "Classroom",
    "Hotel and Hostel",
    "Pub",
    "Library",
    "Spa and Wellness",
}

MANIPULATION_MAX_DIM = 0.5  # metres
OBSTACLE_MAX_DIM = 2.0      # metres


def classify_subset(max_dim: float, in_whitelist: bool) -> str:
    if not in_whitelist or max_dim > OBSTACLE_MAX_DIM:
        return "excluded"
    if max_dim <= MANIPULATION_MAX_DIM:
        return "manipulation"
    return "obstacle"


def main():
    parser = argparse.ArgumentParser(description="Filter assets by category and size")
    parser.add_argument("--metadata-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    top_categories = json.loads(
        (args.metadata_dir.parent / "top_categories.json").read_text()
    )

    parquet_files = sorted(args.metadata_dir.glob("train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {args.metadata_dir}")

    # Exclude image columns — each contains raw PNG bytes and would exhaust RAM at 10k scale
    image_cols = {"seed_image", "render_perspective", "render_front", "render_back", "render_left", "render_right"}
    import pyarrow.parquet as pq
    all_cols = pq.read_schema(parquet_files[0]).names
    keep_cols = [c for c in all_cols if c not in image_cols]
    df = pd.concat(
        [pd.read_parquet(f, columns=keep_cols) for f in parquet_files],
        ignore_index=True,
    )

    df["top_name"] = df["top_category"].apply(lambda x: top_categories[x])
    df["max_dim"] = df["aabb"].apply(lambda x: float(np.max(x)))
    df["in_whitelist"] = df["top_name"].isin(CATEGORY_WHITELIST)
    df["subset"] = df.apply(
        lambda r: classify_subset(r["max_dim"], r["in_whitelist"]), axis=1
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    total = len(df)
    kept = df[df["subset"] != "excluded"]
    print(f"Total assets:        {total}")
    print(f"Excluded:            {(df['subset'] == 'excluded').sum()}")
    print(f"Kept:                {len(kept)}")
    print(f"  manipulation (<{MANIPULATION_MAX_DIM}m): {(df['subset'] == 'manipulation').sum()}")
    print(f"  obstacle     (<{OBSTACLE_MAX_DIM}m):  {(df['subset'] == 'obstacle').sum()}")
    print()
    print("Per-category breakdown (kept only):")
    print(
        kept.groupby("top_name")["subset"]
        .value_counts()
        .unstack(fill_value=0)
        .to_string()
    )
    print(f"\nSaved filtered manifest to: {args.out}")


if __name__ == "__main__":
    main()
