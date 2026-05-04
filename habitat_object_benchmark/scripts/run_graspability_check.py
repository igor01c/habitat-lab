#!/usr/bin/env python3

import argparse
import csv
import multiprocessing as mp
import os

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["MAGNUM_GPU_VALIDATION"] = "off"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["GLOG_minloglevel"] = "5"

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from habitat_object_benchmark.checks import graspability_check
from habitat_object_benchmark.checks.graspability_check import FetchIKSolver
from habitat_object_benchmark.utils.sim_factory import make_sim, load_fetch_robot, FETCH_URDF

GRASP_MAX_DIM = 0.20   # only test graspability on objects ≤ 20 cm

FIELDNAMES = [
    "asset_id",
    "collision_mode",
    "grasp_success_rate",
    "grasp_successes",
    "grasp_trials",
    "mean_grasp_width_m",
    "error",
]

FIELDNAMES_SNAP = [
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

_sim             = None
_robot           = None
_ik_solver       = None
_modes           = None
_images_dirs     = None
_config_dir      = None
_timeout_s       = None
_save_all_trials = None
_use_snap        = None


def _worker_init(config_dir, scene_path, save_images, modes, out_dir, timeout_s, save_all_trials=False, use_snap=False):
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["MAGNUM_GPU_VALIDATION"] = "off"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    os.environ["GLOG_minloglevel"] = "5"
    import sys
    devnull = open(os.devnull, "w")
    sys.stderr = devnull
    os.dup2(devnull.fileno(), 2)

    global _sim, _robot, _ik_solver, _modes, _images_dirs, _config_dir, _timeout_s, _save_all_trials, _use_snap
    _modes           = modes
    _config_dir      = config_dir
    _timeout_s       = timeout_s
    _save_all_trials = save_all_trials
    _use_snap        = use_snap
    need_renderer = save_images or save_all_trials
    _sim        = make_sim(scene_path=scene_path, with_renderer=need_renderer, simple_floor=True)
    _robot      = load_fetch_robot(_sim)
    _ik_solver  = FetchIKSolver(FETCH_URDF)

    _images_dirs = {}
    for mode in modes:
        if need_renderer:
            d = Path(out_dir) / "images" / mode
            d.mkdir(parents=True, exist_ok=True)
            _images_dirs[mode] = str(d)
        else:
            _images_dirs[mode] = None


def _process_asset(asset_id):
    otm = _sim.get_object_template_manager()
    cfg_path = str(Path(_config_dir) / f"{asset_id}.object_config.json")
    otm.load_configs(cfg_path)
    handles = otm.get_template_handles(asset_id)
    rows = []
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
        if _use_snap:
            result = graspability_check.run_snap(
                _sim, _robot, _ik_solver, handles[0], collision_mode=mode,
                save_dir=_images_dirs[mode], asset_id=asset_id,
            )
        else:
            result = graspability_check.run(
                _sim, _robot, _ik_solver, handles[0], collision_mode=mode,
                save_dir=_images_dirs[mode], asset_id=asset_id,
                save_all_trials=_save_all_trials,
            )
        rows.append({"asset_id": asset_id, **result})
    return rows


def _error_rows(asset_id, modes, msg):
    return [{"asset_id": asset_id, "collision_mode": m,
             "grasp_success_rate": None, "grasp_successes": None,
             "grasp_trials": None, "mean_grasp_width_m": None,
             "error": msg} for m in modes]


def _run_batch(asset_ids_batch, init_args, workers, csv_file, writer, progress):
    timeout_s = init_args[5]
    modes     = init_args[3]
    ctx  = mp.get_context("spawn")
    pool = ctx.Pool(processes=workers, initializer=_worker_init, initargs=init_args)
    try:
        pending = []
        queue   = list(asset_ids_batch)

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


class _PoolDead(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description="Run graspability check on small assets")
    parser.add_argument("--config-dir",     required=True, type=Path)
    parser.add_argument("--out-dir",        required=True, type=Path)
    parser.add_argument("--collision-mode", choices=["convex_hull", "vhacd", "both"], default="both")
    parser.add_argument("--scene",          default="data/scene_datasets/habitat-test-scenes/apartment_1.glb")
    parser.add_argument("--save-images",    action="store_true")
    parser.add_argument("--limit",          type=int, default=None)
    parser.add_argument("--workers",        type=int, default=1)
    parser.add_argument("--batch-size",     type=int, default=50)
    parser.add_argument("--timeout",        type=int, default=60,
                        help="Per-asset timeout in seconds (default: 60)")
    parser.add_argument("--save-all-trials", action="store_true",
                        help="Save GIFs for all trials to trials_debug/<asset_id>/")
    parser.add_argument("--resume",         action="store_true")
    parser.add_argument("--snap",           action="store_true",
                        help="Use snap-based (suction-cup) grasping instead of physics gripper")
    parser.add_argument("--manifest",       type=Path, default=None,
                        help="filtered_manifest.parquet — used to filter by max_dim ≤ GRASP_MAX_DIM")
    parser.add_argument("--max-dim",        type=float, default=GRASP_MAX_DIM,
                        help=f"Maximum object dimension for graspability (default: {GRASP_MAX_DIM} m)")
    args = parser.parse_args()

    grasp_dir = args.out_dir / "graspability"
    grasp_dir.mkdir(parents=True, exist_ok=True)
    configs = sorted(args.config_dir.glob("*.object_config.json"))
    if not configs:
        raise FileNotFoundError(f"No .object_config.json files in {args.config_dir}")

    asset_ids = [c.stem.replace(".object_config", "") for c in configs]

    # Filter by size using manifest if provided
    if args.manifest and args.manifest.exists():
        mf = pd.read_parquet(args.manifest)
        if "max_dim" in mf.columns:
            small = set(mf[mf["max_dim"] <= args.max_dim]["asset_id"])
            before = len(asset_ids)
            asset_ids = [a for a in asset_ids if a in small]
            print(f"Size filter (max_dim ≤ {args.max_dim} m): {before} → {len(asset_ids)} assets")
        else:
            print("Warning: manifest has no max_dim column — skipping size filter")
    else:
        print(f"No manifest provided — running on all {len(asset_ids)} assets (no size filter)")

    if args.limit is not None:
        asset_ids = asset_ids[: args.limit]
        print(f"(--limit {args.limit}: testing first {len(asset_ids)} assets)")

    modes    = ["convex_hull", "vhacd"] if args.collision_mode == "both" else [args.collision_mode]
    csv_path = grasp_dir / "graspability_results.csv"

    # Resume: skip assets already in CSV
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

    method = "snap" if args.snap else "physics"
    print(f"Running graspability check [{method}] on {len(asset_ids)} assets "
          f"(modes: {modes}, workers: {args.workers}, batch_size: {args.batch_size})...")

    init_args = (
        str(args.config_dir), args.scene, args.save_images,
        modes, str(grasp_dir), args.timeout, args.save_all_trials, args.snap,
    )

    fieldnames = FIELDNAMES_SNAP if args.snap else FIELDNAMES
    open_mode  = "a" if args.resume else "w"
    batches    = [asset_ids[i:i + args.batch_size]
                  for i in range(0, len(asset_ids), args.batch_size)]

    with open(csv_path, open_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        overall = tqdm(total=len(asset_ids), desc="assets")
        for batch_idx, batch in enumerate(batches):
            tqdm.write(f"Batch {batch_idx + 1}/{len(batches)} ({len(batch)} assets) — spawning fresh pool...")
            try:
                _run_batch(batch, init_args, args.workers, f, writer, overall)
            except Exception as e:
                tqdm.write(f"  Batch {batch_idx + 1} crashed: {e} — continuing")
                f.flush()
        overall.close()

    df = pd.read_csv(csv_path)
    for mode in modes:
        m         = df[df["collision_mode"] == mode]
        success   = (m["grasp_success_rate"] > 0).sum()
        perfect   = (m["grasp_success_rate"] == 1.0).sum()
        mean_rate = m["grasp_success_rate"].mean()
        print(f"\n[{mode}]  any success: {success}/{len(m)}  "
              f"perfect: {perfect}  mean rate: {mean_rate:.3f}")
        if args.snap and "snap_rate" in m.columns:
            print(f"  snap rate: {m['snap_rate'].mean():.3f}  "
                  f"mean EE dist: {m['mean_ee_dist_m'].mean():.3f} m")
        print(f"  errors: {m['error'].notna().sum()}")

    print(f"\nFull results: {csv_path}")


if __name__ == "__main__":
    main()
