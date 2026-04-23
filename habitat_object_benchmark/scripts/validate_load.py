#!/usr/bin/env python3

import argparse
import csv
import os
import time
from pathlib import Path

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["GLOG_minloglevel"] = "5"

from tqdm import tqdm

from habitat_object_benchmark.utils.sim_factory import make_sim


def validate_one(sim, config_path: Path) -> dict:
    asset_id = config_path.stem.replace(".object_config", "")
    result = {"asset_id": asset_id, "load_success": False, "load_error": None, "load_time_s": None}

    try:
        otm = sim.get_object_template_manager()
        rom = sim.get_rigid_object_manager()

        t0 = time.perf_counter()
        otm.load_configs(str(config_path.parent))
        handles = otm.get_template_handles(config_path.stem.split(".")[0])
        if not handles:
            result["load_error"] = "no template handle found"
            return result

        obj = rom.add_object_by_template_handle(handles[0])
        result["load_time_s"] = round(time.perf_counter() - t0, 4)

        if obj is None or not obj.is_alive:
            result["load_error"] = "object not alive after add"
            return result

        rom.remove_object_by_id(obj.object_id)
        result["load_success"] = True

    except Exception as e:
        result["load_error"] = f"{type(e).__name__}: {e}"

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate that all object configs load in habitat-sim")
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--scene", default="data/scene_datasets/habitat-test-scenes/apartment_1.glb")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = sorted(args.config_dir.glob("*.object_config.json"))
    if not configs:
        raise FileNotFoundError(f"No .object_config.json files in {args.config_dir}")

    print(f"Validating {len(configs)} assets...")
    sim = make_sim(scene_path=args.scene, with_renderer=False)

    results = []
    for cfg_path in tqdm(configs):
        results.append(validate_one(sim, cfg_path))

    sim.close()

    csv_path = args.out_dir / "load_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["asset_id", "load_success", "load_error", "load_time_s"])
        writer.writeheader()
        writer.writerows(results)

    failed = [r["asset_id"] for r in results if not r["load_success"]]
    if failed:
        (args.out_dir / "failed_assets.txt").write_text("\n".join(failed))

    n_ok = sum(r["load_success"] for r in results)
    print(f"\nResults: {n_ok}/{len(results)} loaded successfully")
    if failed:
        print(f"Failed ({len(failed)}): see {args.out_dir / 'failed_assets.txt'}")
    print(f"Full results: {csv_path}")


if __name__ == "__main__":
    main()
