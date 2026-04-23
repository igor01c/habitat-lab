#!/usr/bin/env python3

import argparse
import csv
import multiprocessing as mp
import os

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["GLOG_minloglevel"] = "5"

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from habitat_object_benchmark.checks import physics_check
from habitat_object_benchmark.scripts.build_results_json import build as build_results_json
from habitat_object_benchmark.utils.sim_factory import make_sim

FIELDNAMES = [
    "asset_id",
    "collision_mode",
    "physics_settles",
    "displacement_m",
    "flies_away",
    "final_y_offset_m",
    "sinks_permanently",
    "min_y_offset_m",
    "sinks_below_floor",
    "contact_points_at_rest",
    "error",
]

# Per-process simulator (initialised once per worker)
_sim = None
_modes = None
_images_dirs = None


def _worker_init(config_dir, scene_path, save_images, modes, out_dir):
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    os.environ["GLOG_minloglevel"] = "5"

    global _sim, _modes, _images_dirs
    _modes = modes
    _sim = make_sim(scene_path=scene_path, with_renderer=save_images, simple_floor=True)
    _sim.get_object_template_manager().load_configs(config_dir)

    _images_dirs = {}
    for mode in modes:
        if save_images:
            d = Path(out_dir) / "images" / mode
            d.mkdir(parents=True, exist_ok=True)
            _images_dirs[mode] = str(d)
        else:
            _images_dirs[mode] = None


def _process_asset(asset_id):
    otm = _sim.get_object_template_manager()
    handles = otm.get_template_handles(asset_id)
    rows = []
    if not handles:
        for mode in _modes:
            rows.append({"asset_id": asset_id, "collision_mode": mode,
                         "physics_settles": False, "displacement_m": None,
                         "flies_away": None, "final_y_offset_m": None,
                         "sinks_permanently": None, "min_y_offset_m": None,
                         "sinks_below_floor": None,
                         "contact_points_at_rest": None, "error": "handle not found"})
        return rows
    for mode in _modes:
        result = physics_check.run(
            _sim, handles[0], collision_mode=mode,
            save_dir=_images_dirs[mode], asset_id=asset_id,
        )
        rows.append({"asset_id": asset_id, **result})
    return rows


def _run_batch(asset_ids_batch, init_args, workers, csv_file, writer):
    """Process one batch inside a fresh Pool, then destroy it to free all memory."""
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=init_args,
    ) as pool:
        for rows in pool.imap_unordered(_process_asset, asset_ids_batch):
            for row in rows:
                writer.writerow(row)
            csv_file.flush()


def main():
    parser = argparse.ArgumentParser(description="Run physics stability check on all assets")
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--collision-mode", choices=["convex_hull", "vhacd", "both"], default="both")
    parser.add_argument("--scene", default="data/scene_datasets/habitat-test-scenes/apartment_1.glb")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Assets per Pool batch — pool is destroyed and recreated between "
                             "batches to release habitat-sim mesh cache memory (default: 100)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip assets already present in the output CSV")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = sorted(args.config_dir.glob("*.object_config.json"))
    if not configs:
        raise FileNotFoundError(f"No .object_config.json files in {args.config_dir}")
    if args.limit is not None:
        configs = configs[: args.limit]
        print(f"(--limit {args.limit}: testing on first {len(configs)} assets)")

    modes = ["convex_hull", "vhacd"] if args.collision_mode == "both" else [args.collision_mode]
    asset_ids = [c.stem.replace(".object_config", "") for c in configs]
    csv_path = args.out_dir / "physics_results.csv"

    # Resume: skip assets already in the CSV
    done: set = set()
    write_header = True
    if args.resume and csv_path.exists():
        existing = pd.read_csv(csv_path)
        # An asset is fully done only if all modes are present
        counts = existing.groupby("asset_id")["collision_mode"].count()
        done = set(counts[counts >= len(modes)].index)
        write_header = False
        print(f"Resuming — {len(done)} assets already done, {len(asset_ids) - len(done)} remaining.")

    asset_ids = [a for a in asset_ids if a not in done]
    if not asset_ids:
        print("Nothing to do.")
        return

    print(f"Running physics check on {len(asset_ids)} assets "
          f"(modes: {modes}, workers: {args.workers}, batch_size: {args.batch_size})...")

    init_args = (
        str(args.config_dir), args.scene, args.save_images,
        modes, str(args.out_dir),
    )

    open_mode = "a" if args.resume else "w"
    batches = [asset_ids[i:i + args.batch_size] for i in range(0, len(asset_ids), args.batch_size)]

    with open(csv_path, open_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        overall = tqdm(total=len(asset_ids), desc="assets")
        for batch_idx, batch in enumerate(batches):
            tqdm.write(f"Batch {batch_idx + 1}/{len(batches)} ({len(batch)} assets) — spawning fresh pool...")
            _run_batch(batch, init_args, args.workers, f, writer)
            overall.update(len(batch))
        overall.close()

    # Summary
    df = pd.read_csv(csv_path)
    for mode in modes:
        m = df[df["collision_mode"] == mode]
        settled = m["physics_settles"].sum()
        flies   = m["flies_away"].sum()
        perm    = m["sinks_permanently"].sum()
        traj    = m["sinks_below_floor"].sum()
        print(f"\n[{mode}]  settled: {settled}/{len(m)} ({100*settled/len(m):.1f}%)")
        print(f"  flies_away:          {flies}")
        print(f"  sinks_permanently:   {perm}  (final_y < -0.10 m)")
        print(f"  sinks_below_floor:   {traj}  (min_y < -0.15 m)")
        print(f"  mean displacement_m:  {m['displacement_m'].mean():.4f}")
        print(f"  mean final_y_offset:  {m['final_y_offset_m'].mean():.4f}")
        print(f"  mean min_y_offset:    {m['min_y_offset_m'].mean():.4f}")
        print(f"  errors:               {m['error'].notna().sum()}")

    print(f"\nFull results: {csv_path}")

    # Rebuild results.json so the inspector reflects the latest run
    results_root = args.out_dir.parent
    build_results_json(results_root, results_root / "results.json")


if __name__ == "__main__":
    main()
