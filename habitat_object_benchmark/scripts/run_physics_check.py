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

from habitat_object_benchmark.checks import physics_check
from habitat_object_benchmark.utils.sim_factory import make_sim

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

# Per-process simulator (initialised once per worker)
_sim = None
_modes = None
_images_dirs = None
_config_dir = None
_timeout_s = None
_debug_html_dir = None


_penetration_method = None


def _worker_init(config_dir, scene_path, save_images, modes, out_dir, timeout_s,
                 debug_html_dir=None, penetration_method="vertex"):
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["MAGNUM_GPU_VALIDATION"] = "off"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    os.environ["GLOG_minloglevel"] = "5"
    # Redirect stderr to suppress C++ warnings (MeshTools, Magnum) from workers
    import sys
    devnull = open(os.devnull, "w")
    sys.stderr = devnull
    os.dup2(devnull.fileno(), 2)

    global _sim, _modes, _images_dirs, _config_dir, _timeout_s, _debug_html_dir, _penetration_method
    _modes = modes
    _config_dir = config_dir
    _timeout_s = timeout_s
    _debug_html_dir = debug_html_dir
    _penetration_method = penetration_method
    _sim = make_sim(scene_path=scene_path, with_renderer=save_images, simple_floor=True)
    # Do NOT bulk-load all configs here — load each template lazily in _process_asset
    # to avoid stalling workers on large GLB datasets (e.g. Objaverse).

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
    cfg_path = str(Path(_config_dir) / f"{asset_id}.object_config.json")
    otm.load_configs(cfg_path)
    handles = otm.get_template_handles(asset_id)
    rows = []
    if not handles:
        for mode in _modes:
            rows.append({"asset_id": asset_id, "collision_mode": mode,
                         "physics_settles": None, "physics_stable": None,
                         "displacement_m": None, "flies_away": None,
                         "penetration_y_m": None, "floor_penetration": None,
                         "settle_time_s": None, "wall_time_s": None,
                         "contact_points_at_rest": None, "error": "handle not found"})
        return rows
    for mode in _modes:
        debug_html = None
        if _debug_html_dir is not None:
            d = Path(_debug_html_dir) / mode
            d.mkdir(parents=True, exist_ok=True)
            debug_html = str(d / f"{asset_id}.html")
        result = physics_check.run(
            _sim, handles[0], collision_mode=mode,
            save_dir=_images_dirs[mode], asset_id=asset_id,
            config_json_path=cfg_path, debug_html=debug_html,
            penetration_method=_penetration_method,
        )
        rows.append({"asset_id": asset_id, **result})
    return rows


def _error_rows(asset_id, modes, msg):
    return [{"asset_id": asset_id, "collision_mode": m,
             "physics_settles": None, "physics_stable": None,
             "displacement_m": None, "flies_away": None,
             "penetration_y_m": None, "floor_penetration": None,
             "settle_time_s": None, "wall_time_s": None,
             "contact_points_at_rest": None,
             "error": msg} for m in modes]


def _run_batch(asset_ids_batch, init_args, workers, csv_file, writer, progress):
    """Process one batch with per-asset timeout via apply_async.

    Submits exactly `workers` tasks at a time so every submitted task is
    actively running when get(timeout=...) is called — avoiding false timeouts
    on queued-but-not-started tasks.
    """
    timeout_s = init_args[5]  # passed through init_args
    modes     = init_args[3]
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=workers, initializer=_worker_init, initargs=init_args)
    try:
        pending = []   # list of (future, asset_id)
        queue   = list(asset_ids_batch)

        def _drain_one(future, asset_id):
            try:
                rows = future.get(timeout=timeout_s)
            except mp.TimeoutError:
                tqdm.write(f"  TIMEOUT ({timeout_s}s): {asset_id}")
                rows = _error_rows(asset_id, modes, f"timeout>{timeout_s}s")
            except Exception as e:
                rows = _error_rows(asset_id, modes, str(e))
            for row in rows:
                writer.writerow(row)
            csv_file.flush()
            progress.update(1)

        # Seed initial tasks
        for aid in queue[:workers]:
            pending.append((pool.apply_async(_process_asset, (aid,)), aid))
        remaining = queue[workers:]

        for aid in remaining:
            future, done_id = pending.pop(0)
            _drain_one(future, done_id)
            pending.append((pool.apply_async(_process_asset, (aid,)), aid))

        for future, aid in pending:
            _drain_one(future, aid)

    except Exception as e:
        tqdm.write(f"  _run_batch unexpected error: {e}")
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
    parser.add_argument("--timeout", type=int, default=20,
                        help="Per-asset timeout in seconds before marking as error (default: 20)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip assets already present in the output CSV")
    parser.add_argument("--debug-html-dir", type=Path, default=None,
                        help="Save a 3D HTML debug visualisation (floor + transformed mesh "
                             "vertices at final pose) for each asset into this directory")
    parser.add_argument("--include", nargs="+", metavar="ASSET_ID",
                        help="Run only these asset IDs (space-separated, or use multiple times)")
    parser.add_argument("--penetration-method",
                        choices=["vertex", "contact", "raycast"], default="vertex",
                        help="Method for floor penetration tracking: "
                             "vertex (trimesh, default), contact (Bullet contact points), "
                             "raycast (upward ray per step)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = sorted(args.config_dir.glob("*.object_config.json"))
    if not configs:
        raise FileNotFoundError(f"No .object_config.json files in {args.config_dir}")
    if args.include is not None:
        include_set = set(args.include)
        configs = [c for c in configs if c.stem.replace(".object_config", "") in include_set]
        if not configs:
            raise FileNotFoundError(f"None of the --include asset IDs found in {args.config_dir}")
        print(f"(--include: running {len(configs)} specified assets)")
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
          f"(modes: {modes}, workers: {args.workers}, batch_size: {args.batch_size}, "
          f"penetration: {args.penetration_method})...")

    init_args = (
        str(args.config_dir), args.scene, args.save_images,
        modes, str(args.out_dir), args.timeout,
        str(args.debug_html_dir) if args.debug_html_dir else None,
        args.penetration_method,
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
            try:
                _run_batch(batch, init_args, args.workers, f, writer, overall)
            except Exception as e:
                tqdm.write(f"  Batch {batch_idx + 1} crashed: {e} — continuing with next batch")
                f.flush()
        overall.close()

    # Summary
    df = pd.read_csv(csv_path)
    for mode in modes:
        m = df[df["collision_mode"] == mode]
        n = len(m)
        settles  = m["physics_settles"].eq(True).sum()
        stable   = m["physics_stable"].eq(True).sum()
        flies    = m["flies_away"].eq(True).sum()
        pen      = m["floor_penetration"].eq(True).sum()
        errors   = m["error"].notna().sum()
        # wall_time_s stats (excludes error/timeout rows which have NaN)
        wt = m["wall_time_s"].dropna()
        within_60s = int((wt <= 60.0).sum())
        total_w = len(wt)
        print(f"\n[{mode}]  settles: {settles}/{n}  stable: {stable}/{n}")
        print(f"  flies_away:        {flies}")
        print(f"  floor_penetration: {pen}  (penetration_y < -0.05 m)")
        print(f"  mean displacement_m:  {m['displacement_m'].mean():.4f}")
        print(f"  mean penetration_y_m: {m['penetration_y_m'].mean():.4f}")
        print(f"  errors/timeouts:      {errors}")
        if total_w > 0:
            print(f"  wall_time: mean={wt.mean():.3f}s  median={wt.median():.3f}s  "
                  f"max={wt.max():.3f}s  within_60s={within_60s}/{total_w} "
                  f"({100*within_60s/total_w:.1f}%)")

    print(f"\nFull results: {csv_path}")


if __name__ == "__main__":
    main()
