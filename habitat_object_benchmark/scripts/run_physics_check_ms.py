#!/usr/bin/env python3
"""Run physics stability check using SAPIEN 3 / ManiSkill3 (PhysX backend)."""

import argparse
import csv
import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from habitat_object_benchmark.checks import physics_check_ms
from habitat_object_benchmark.utils.maniskill_factory import make_scene

FIELDNAMES = [
    "asset_id",
    "collision_mode",
    "physics_settles",
    "physics_stable",
    "displacement_m",
    "flies_away",
    "penetration_y_m",
    "floor_penetration",
    "settle_time_s",
    "wall_time_s",
    "contact_points_at_rest",
    "error",
]

_scene           = None
_modes           = None
_images_dirs     = None
_config_dir      = None
_timeout_s       = None
_camera_dist     = None


def _worker_init(config_dir, save_images, modes, out_dir, timeout_s, camera_dist):
    global _scene, _modes, _images_dirs, _config_dir, _timeout_s, _camera_dist
    _modes       = modes
    _config_dir  = config_dir
    _timeout_s   = timeout_s
    _camera_dist = camera_dist
    _scene       = make_scene(with_renderer=save_images)
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
        result = physics_check_ms.run(
            _scene, cfg_path,
            collision_mode=mode,
            save_dir=_images_dirs[mode],
            asset_id=asset_id,
            camera_dist=_camera_dist,
        )
        rows.append({"asset_id": asset_id, **result})
    return rows


def _error_rows(asset_id, modes, msg):
    return [{"asset_id": asset_id, "collision_mode": m,
             "physics_settles": None, "physics_stable": None,
             "displacement_m": None, "flies_away": None,
             "penetration_y_m": None, "floor_penetration": None,
             "settle_time_s": None, "wall_time_s": None,
             "contact_points_at_rest": None, "error": msg} for m in modes]


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
    parser = argparse.ArgumentParser(description="Run physics check (ManiSkill/SAPIEN) on assets")
    parser.add_argument("--config-dir",     required=True, type=Path)
    parser.add_argument("--out-dir",        required=True, type=Path)
    parser.add_argument("--collision-mode", choices=["convex_hull", "vhacd", "raw", "both", "all"], default="both")
    parser.add_argument("--save-images",    action="store_true")
    parser.add_argument("--limit",          type=int, default=None)
    parser.add_argument("--workers",        type=int, default=1)
    parser.add_argument("--batch-size",     type=int, default=50)
    parser.add_argument("--timeout",        type=int, default=60)
    parser.add_argument("--resume",         action="store_true")
    args = parser.parse_args()

    physics_dir = args.out_dir / "physics"
    physics_dir.mkdir(parents=True, exist_ok=True)

    configs = sorted(args.config_dir.glob("*.object_config.json"))
    if not configs:
        raise FileNotFoundError(f"No .object_config.json files in {args.config_dir}")

    asset_ids = [c.stem.replace(".object_config", "") for c in configs]
    if args.limit:
        asset_ids = asset_ids[:args.limit]

    if args.collision_mode == "all":
        modes = ["convex_hull", "vhacd", "raw"]
    elif args.collision_mode == "both":
        modes = ["convex_hull", "vhacd"]
    else:
        modes = [args.collision_mode]
    csv_path = physics_dir / "physics_results.csv"

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

    print(f"Running physics check [ManiSkill/SAPIEN] on {len(asset_ids)} assets "
          f"(modes: {modes}, workers: {args.workers})...")

    init_args = (
        str(args.config_dir), args.save_images,
        modes, str(physics_dir), args.timeout, None,
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
        n = len(m)
        settles = m["physics_settles"].eq(True).sum()
        stable  = m["physics_stable"].eq(True).sum()
        flies   = m["flies_away"].eq(True).sum()
        pen     = m["floor_penetration"].eq(True).sum()
        print(f"\n[{mode}]  settles: {settles}/{n}  stable: {stable}/{n}")
        print(f"  flies_away:        {flies}")
        print(f"  floor_penetration: {pen}  (penetration_y < -0.05 m)")
        print(f"  mean displacement_m:  {m['displacement_m'].mean():.4f}")
        print(f"  mean penetration_y_m: {m['penetration_y_m'].mean():.4f}")
        print(f"  mean wall_time_s:     {m['wall_time_s'].mean():.2f}")
        print(f"  errors:               {m['error'].notna().sum()}")

    print(f"\nFull results: {csv_path}")


if __name__ == "__main__":
    main()
