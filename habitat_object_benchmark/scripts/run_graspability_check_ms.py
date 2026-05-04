#!/usr/bin/env python3
"""Run snap-based graspability check using SAPIEN 3 / ManiSkill3 (PhysX backend)."""

import argparse
import csv
import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from habitat_object_benchmark.checks import graspability_check_ms
from habitat_object_benchmark.checks.graspability_check_ms import (
    MS_FETCH_URDF_BUNDLED,
    MS_FETCH_URDF_ARM_IK,
)
from habitat_object_benchmark.utils.maniskill_factory import make_scene, setup_robot_drives

FIELDNAMES = [
    "asset_id",
    "collision_mode",
    "grasp_success_rate",
    "grasp_successes",
    "grasp_trials",
    "mean_grasp_width_m",
    "snap_rate",
    "mean_ee_dist_m",
    "error",
]

_scene       = None
_robot       = None
_chain       = None
_ik_solver   = None
_lo          = None
_hi          = None
_modes       = None
_images_dirs = None
_config_dir  = None
_timeout_s   = None


def _worker_init(config_dir, save_images, modes, out_dir, timeout_s):
    global _scene, _robot, _chain, _ik_solver, _lo, _hi
    global _modes, _images_dirs, _config_dir, _timeout_s

    _modes      = modes
    _config_dir = config_dir
    _timeout_s  = timeout_s

    import sapien
    _scene = make_scene(with_renderer=save_images)
    loader = _scene.create_urdf_loader()
    loader.fix_root_link = True
    _robot = loader.load(MS_FETCH_URDF_BUNDLED)
    setup_robot_drives(_robot)

    _chain, _ik_solver, _lo, _hi = graspability_check_ms._make_ik_solver(MS_FETCH_URDF_ARM_IK)

    _images_dirs = {}
    for mode in modes:
        if save_images:
            d = Path(out_dir) / "images" / mode
            d.mkdir(parents=True, exist_ok=True)
            _images_dirs[mode] = str(d)
        else:
            _images_dirs[mode] = None


def _process_asset(asset_id):
    cfg_path = str(Path(_config_dir) / f"{asset_id}.object_config.json")
    if not Path(cfg_path).exists():
        return [{"asset_id": asset_id, "collision_mode": m, "error": "config not found"}
                for m in _modes]

    rows = []
    for mode in _modes:
        result = graspability_check_ms.run_snap(
            _scene, _robot, _chain, _ik_solver,
            cfg_path,
            collision_mode=mode,
            save_dir=_images_dirs[mode],
            asset_id=asset_id,
        )
        rows.append({"asset_id": asset_id, **result})
    return rows


def _error_rows(asset_id, modes, msg):
    return [{"asset_id": asset_id, "collision_mode": m,
             "grasp_success_rate": None, "grasp_successes": None,
             "grasp_trials": None, "mean_grasp_width_m": None,
             "snap_rate": None, "mean_ee_dist_m": None,
             "error": msg} for m in modes]


class _PoolDead(Exception):
    pass


def _run_batch(asset_ids_batch, init_args, workers, csv_file, writer, progress):
    timeout_s = init_args[4]
    modes     = init_args[2]
    ctx  = mp.get_context("spawn")
    pool = ctx.Pool(processes=workers, initializer=_worker_init, initargs=init_args)
    try:
        pending = []
        queue = list(asset_ids_batch)

        def _drain_one(future, asset_id):
            try:
                rows = future.get(timeout=timeout_s)
            except mp.TimeoutError:
                tqdm.write(f"  TIMEOUT ({timeout_s}s): {asset_id}")
                pool.terminate()
                pool.join()
                raise _PoolDead()
            except Exception as e:
                rows = _error_rows(asset_id, modes, str(e))
            for row in rows:
                writer.writerow(row)
            csv_file.flush()
            progress.update(1)

        for aid in queue[:workers]:
            pending.append((pool.apply_async(_process_asset, (aid,)), aid))
        remaining = queue[workers:]

        for aid in remaining:
            future, done_id = pending.pop(0)
            _drain_one(future, done_id)
            pending.append((pool.apply_async(_process_asset, (aid,)), aid))

        for future, aid in pending:
            _drain_one(future, aid)

    except _PoolDead:
        pass
    except Exception as e:
        tqdm.write(f"  _run_batch error: {e}")
    finally:
        try:
            pool.terminate()
        except Exception:
            pass
        try:
            pool.join(timeout=10)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Run snap-based graspability check (ManiSkill/SAPIEN) on assets")
    parser.add_argument("--config-dir",     required=True, type=Path)
    parser.add_argument("--out-dir",        required=True, type=Path)
    parser.add_argument("--collision-mode", choices=["convex_hull", "vhacd", "both"],
                        default="convex_hull")
    parser.add_argument("--save-images",    action="store_true")
    parser.add_argument("--limit",          type=int, default=None)
    parser.add_argument("--workers",        type=int, default=1)
    parser.add_argument("--batch-size",     type=int, default=50)
    parser.add_argument("--timeout",        type=int, default=120)
    parser.add_argument("--resume",         action="store_true")
    args = parser.parse_args()

    grasp_dir = args.out_dir / "graspability"
    grasp_dir.mkdir(parents=True, exist_ok=True)

    configs = sorted(args.config_dir.glob("*.object_config.json"))
    if not configs:
        raise FileNotFoundError(f"No .object_config.json files in {args.config_dir}")

    asset_ids = [c.stem.replace(".object_config", "") for c in configs]
    if args.limit:
        asset_ids = asset_ids[:args.limit]

    modes    = ["convex_hull", "vhacd"] if args.collision_mode == "both" else [args.collision_mode]
    csv_path = grasp_dir / "graspability_results.csv"

    done: set = set()
    write_header = True
    if args.resume and csv_path.exists():
        existing = pd.read_csv(csv_path)
        counts = existing.groupby("asset_id")["collision_mode"].count()
        done = set(counts[counts >= len(modes)].index)
        write_header = False
        print(f"Resuming — {len(done)} done, {len(asset_ids) - len(done)} remaining.")

    asset_ids = [a for a in asset_ids if a not in done]
    if not asset_ids:
        print("Nothing to do.")
        return

    print(f"Running graspability check [ManiSkill/SAPIEN] on {len(asset_ids)} assets "
          f"(modes: {modes}, workers: {args.workers})...")

    init_args = (
        str(args.config_dir), args.save_images,
        modes, str(grasp_dir), args.timeout,
    )

    open_mode = "a" if args.resume else "w"
    batches = [asset_ids[i:i + args.batch_size]
               for i in range(0, len(asset_ids), args.batch_size)]

    with open(csv_path, open_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        overall = tqdm(total=len(asset_ids), desc="assets")
        for batch_idx, batch in enumerate(batches):
            tqdm.write(f"Batch {batch_idx + 1}/{len(batches)} ({len(batch)} assets)...")
            try:
                _run_batch(batch, init_args, args.workers, f, writer, overall)
            except Exception as e:
                tqdm.write(f"  Batch {batch_idx + 1} crashed: {e} — continuing")
                f.flush()
        overall.close()

    df = pd.read_csv(csv_path)
    for mode in modes:
        m = df[df["collision_mode"] == mode]
        n_snap = m["snap_rate"].gt(0).sum()
        print(f"\n[{mode}]  snap_rate>0: {n_snap}/{len(m)}  errors: {m['error'].notna().sum()}")

    print(f"\nFull results: {csv_path}")


if __name__ == "__main__":
    main()
