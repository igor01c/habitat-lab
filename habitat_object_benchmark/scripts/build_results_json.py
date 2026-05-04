#!/usr/bin/env python3
"""Merge experiment CSVs + image paths into a single results.json.

Single-dataset usage:
    python -m habitat_object_benchmark.scripts.build_results_json \
        --results-dir data/datasets/amara-spatial-10k/results \
        --out         data/datasets/amara-spatial-10k/results/results.json

Multi-dataset usage (produces a combined results.json with a "datasets" key):
    python -m habitat_object_benchmark.scripts.build_results_json \
        --dataset amara=data/datasets/amara-spatial-10k/results \
        --dataset ycb=data/datasets/ycb/results \
        --out inspector/results.json
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Optional


# ── CSV loaders ──────────────────────────────────────────────────────────────

def _parse_bool(val: str) -> Optional[bool]:
    if val.lower() == "true":  return True
    if val.lower() == "false": return False
    return None


def _parse_float(val: str) -> Optional[float]:
    return float(val) if val else None


def _parse_int(val: str) -> Optional[int]:
    return int(val) if val else None


def _gif_path(gif: Path, json_out: Path) -> Optional[str]:
    """Return GIF path relative to the output JSON file's directory, or None."""
    if not gif.exists():
        return None
    return str(os.path.relpath(gif.resolve(), json_out.parent.resolve()))


def load_graspability(csv_path: Path, images_dir: Path, json_out: Path) -> dict:
    """Returns {asset_id: {mode: {metrics + gif_path}}}"""
    data = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            asset_id = r["asset_id"]
            mode     = r["collision_mode"]
            entry = {
                "grasp_success_rate": _parse_float(r["grasp_success_rate"]),
                "grasp_successes":    _parse_int(r["grasp_successes"]),
                "grasp_trials":       _parse_int(r["grasp_trials"]),
                "mean_grasp_width_m": _parse_float(r["mean_grasp_width_m"]),
                "error":              r["error"] or None,
                "gif":                _gif_path(images_dir / mode / f"{asset_id}.gif", json_out),
            }
            data.setdefault(asset_id, {})[mode] = entry
    return data


def load_physics(csv_path: Path, images_dir: Path, json_out: Path) -> dict:
    """Returns {asset_id: {mode: {metrics + gif_path}}}"""
    data = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            asset_id = r["asset_id"]
            mode     = r["collision_mode"]
            error = r["error"] or None
            entry = {
                "physics_settles":        None if error else _parse_bool(r.get("physics_settles")),
                "physics_stable":         None if error else _parse_bool(r.get("physics_stable")),
                "displacement_m":         None if error else _parse_float(r.get("displacement_m")),
                "flies_away":             None if error else _parse_bool(r.get("flies_away")),
                "penetration_y_m":        None if error else _parse_float(r.get("penetration_y_m")),
                "floor_penetration":      None if error else _parse_bool(r.get("floor_penetration")),
                "contact_points_at_rest": None if error else _parse_int(r.get("contact_points_at_rest")),
                "settle_time_s":          None if error else _parse_float(r.get("settle_time_s")),
                "wall_time_s":            _parse_float(r.get("wall_time_s")),
                "error":                  error,
                "gif":                    _gif_path(images_dir / mode / f"{asset_id}.gif", json_out),
            }
            data.setdefault(asset_id, {})[mode] = entry
    return data


# ── Merge ────────────────────────────────────────────────────────────────────

def merge_checks(results_dir: Path, json_out: Path) -> dict:
    assets: dict = {}

    def _update(asset_id, check_name, payload):
        assets.setdefault(asset_id, {})[check_name] = payload

    # Physics — support both flat (legacy) and subdirectory layout
    physics_csv = results_dir / "physics" / "physics_results.csv"
    if not physics_csv.exists():
        physics_csv = results_dir / "physics_results.csv"
    if physics_csv.exists():
        physics_images = physics_csv.parent / "images"
        print(f"Loading physics results from {physics_csv}")
        for asset_id, modes in load_physics(physics_csv, physics_images, json_out).items():
            _update(asset_id, "physics", modes)
    else:
        print(f"  (no physics CSV found at {physics_csv})")

    # Graspability — support both flat (legacy) and subdirectory layout
    grasp_csv = results_dir / "graspability" / "graspability_results.csv"
    if not grasp_csv.exists():
        grasp_csv = results_dir / "graspability_results.csv"
    if grasp_csv.exists():
        grasp_images = grasp_csv.parent / "images"
        print(f"Loading graspability results from {grasp_csv}")
        for asset_id, modes in load_graspability(grasp_csv, grasp_images, json_out).items():
            _update(asset_id, "graspability", modes)
    else:
        print(f"  (no graspability CSV found at {grasp_csv})")

    return assets


# ── Public API ───────────────────────────────────────────────────────────────

def build(results_dir: Path, out: Path) -> None:
    """Single-dataset: merge CSVs under results_dir and write results.json."""
    out.parent.mkdir(parents=True, exist_ok=True)
    assets = merge_checks(results_dir, out)
    payload = {
        "datasets": {
            results_dir.parent.name: {
                "results_dir": str(results_dir.resolve()),
                "assets": assets,
            }
        }
    }
    out.write_text(json.dumps(payload, indent=2))
    n_physics = sum(1 for a in assets.values() if "physics" in a)
    print(f"results.json: {len(assets)} assets ({n_physics} with physics) → {out}")


def build_multi(datasets: dict, out: Path) -> None:
    """Multi-dataset: merge each results_dir and write a combined results.json."""
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"datasets": {}}
    for name, results_dir in datasets.items():
        print(f"\n── {name} ──")
        assets = merge_checks(results_dir, out)
        payload["datasets"][name] = {
            "results_dir": str(results_dir.resolve()),
            "assets": assets,
        }
        n_physics = sum(1 for a in assets.values() if "physics" in a)
        print(f"  {len(assets)} assets ({n_physics} with physics)")

    out.write_text(json.dumps(payload, indent=2))
    print(f"\nresults.json: {len(payload['datasets'])} datasets → {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build results.json from experiment CSVs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results-dir", type=Path,
                       help="Single dataset: root results dir (physics/, graspability/, ...)")
    group.add_argument("--dataset", metavar="NAME=PATH", action="append", dest="datasets",
                       help="Multi-dataset: repeatable NAME=path/to/results pairs")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output path for results.json")
    args = parser.parse_args()

    if args.results_dir:
        build(args.results_dir, args.out)
    else:
        datasets = {}
        for entry in args.datasets:
            if "=" not in entry:
                parser.error(f"--dataset must be NAME=PATH, got: {entry!r}")
            name, path = entry.split("=", 1)
            datasets[name] = Path(path)
        build_multi(datasets, args.out)


if __name__ == "__main__":
    main()
