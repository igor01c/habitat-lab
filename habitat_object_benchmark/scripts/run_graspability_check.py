#!/usr/bin/env python3

import argparse
import csv
import multiprocessing as mp
import os

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["GLOG_minloglevel"] = "5"

from pathlib import Path

from tqdm import tqdm

from habitat_object_benchmark.checks import graspability_check
from habitat_object_benchmark.scripts.build_results_json import build as build_results_json
from habitat_object_benchmark.checks.graspability_check import FetchIKSolver
from habitat_object_benchmark.utils.sim_factory import make_sim, load_fetch_robot, FETCH_URDF

FIELDNAMES = [
    "asset_id",
    "collision_mode",
    "grasp_success_rate",
    "grasp_successes",
    "grasp_trials",
    "mean_grasp_width_m",
    "error",
]

_sim       = None
_robot     = None
_ik_solver = None
_modes     = None
_images_dirs = None


def _worker_init(config_dir, scene_path, save_images, modes, out_dir):
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    os.environ["GLOG_minloglevel"] = "5"

    global _sim, _robot, _ik_solver, _modes, _images_dirs
    _modes     = modes
    _sim       = make_sim(scene_path=scene_path, with_renderer=save_images, simple_floor=True)
    _sim.get_object_template_manager().load_configs(config_dir)
    _robot     = load_fetch_robot(_sim)
    _ik_solver = FetchIKSolver(FETCH_URDF)

    _images_dirs = {}
    for mode in modes:
        if save_images:
            d = Path(out_dir) / "images" / mode
            d.mkdir(parents=True, exist_ok=True)
            _images_dirs[mode] = str(d)
        else:
            _images_dirs[mode] = None


def _process_asset(asset_id):
    otm    = _sim.get_object_template_manager()
    handles = otm.get_template_handles(asset_id)
    rows   = []
    if not handles:
        for mode in _modes:
            rows.append({
                "asset_id": asset_id, "collision_mode": mode,
                "grasp_success_rate": None, "grasp_successes": None,
                "grasp_trials": None, "mean_grasp_width_m": None,
                "error": "handle not found",
            })
        return rows
    for mode in _modes:
        result = graspability_check.run(
            _sim, _robot, _ik_solver, handles[0], collision_mode=mode,
            save_dir=_images_dirs[mode], asset_id=asset_id,
        )
        rows.append({"asset_id": asset_id, **result})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run graspability check on all assets")
    parser.add_argument("--config-dir",      required=True, type=Path)
    parser.add_argument("--out-dir",         required=True, type=Path)
    parser.add_argument("--collision-mode",  choices=["convex_hull", "vhacd", "both"], default="both")
    parser.add_argument("--scene",           default="data/scene_datasets/habitat-test-scenes/apartment_1.glb")
    parser.add_argument("--save-images",     action="store_true")
    parser.add_argument("--limit",           type=int, default=None)
    parser.add_argument("--workers",         type=int, default=1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = sorted(args.config_dir.glob("*.object_config.json"))
    if not configs:
        raise FileNotFoundError(f"No .object_config.json files in {args.config_dir}")
    if args.limit is not None:
        configs = configs[: args.limit]
        print(f"(--limit {args.limit}: testing first {len(configs)} assets)")

    modes     = ["convex_hull", "vhacd"] if args.collision_mode == "both" else [args.collision_mode]
    asset_ids = [c.stem.replace(".object_config", "") for c in configs]
    csv_path  = args.out_dir / "graspability_results.csv"

    print(f"Running graspability check on {len(configs)} assets "
          f"(modes: {modes}, workers: {args.workers})...")

    init_args = (
        str(args.config_dir), args.scene, args.save_images,
        modes, str(args.out_dir),
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        if args.workers == 1:
            _worker_init(*init_args)
            for asset_id in tqdm(asset_ids):
                for row in _process_asset(asset_id):
                    writer.writerow(row)
                f.flush()
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=args.workers,
                initializer=_worker_init,
                initargs=init_args,
            ) as pool:
                for rows in tqdm(
                    pool.imap_unordered(_process_asset, asset_ids),
                    total=len(asset_ids),
                ):
                    for row in rows:
                        writer.writerow(row)
                    f.flush()

    import pandas as pd
    df = pd.read_csv(csv_path)
    for mode in modes:
        m        = df[df["collision_mode"] == mode]
        success  = (m["grasp_success_rate"] > 0).sum()
        perfect  = (m["grasp_success_rate"] == 1.0).sum()
        mean_rate = m["grasp_success_rate"].mean()
        print(f"\n[{mode}]  any success: {success}/{len(m)}  "
              f"perfect (20/20): {perfect}  mean rate: {mean_rate:.3f}")
        print(f"  errors: {m['error'].notna().sum()}")

    print(f"\nFull results: {csv_path}")

    results_root = args.out_dir.parent
    build_results_json(results_root, results_root / "results.json")


if __name__ == "__main__":
    main()
