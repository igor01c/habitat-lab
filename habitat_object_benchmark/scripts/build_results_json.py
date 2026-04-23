#!/usr/bin/env python3
"""Merge experiment CSVs + image paths into a single results.json.

Run this once after experiments complete (or whenever new results are available).
The JSON is consumed by inspector.html via a local HTTP server.

Usage:
    python -m habitat_object_benchmark.scripts.build_results_json \
        --results-dir data/datasets/amara-spatial-10k/results \
        --out         data/datasets/amara-spatial-10k/results/results.json

The script auto-discovers recognised check types under --results-dir:
    <results-dir>/physics/physics_results.csv
    <results-dir>/physics/images/{with,without}/<asset_id>.gif
    (future) <results-dir>/graspability/graspability_results.csv
    (future) <results-dir>/navigation/navigation_results.csv
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


def load_graspability(csv_path: Path, images_dir: Path, results_root: Path) -> dict:
    """Returns {asset_id: {mode: {metrics + gif_path}}}"""
    data = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            asset_id = r["asset_id"]
            mode     = r["collision_mode"]
            gif      = images_dir / mode / f"{asset_id}.gif"
            rel_gif  = str(gif.relative_to(results_root)) if gif.exists() else None
            entry = {
                "grasp_success_rate": _parse_float(r["grasp_success_rate"]),
                "grasp_successes":    _parse_int(r["grasp_successes"]),
                "grasp_trials":       _parse_int(r["grasp_trials"]),
                "mean_grasp_width_m":       _parse_float(r["mean_grasp_width_m"]),
                "error":              r["error"] or None,
                "gif":                rel_gif,
            }
            data.setdefault(asset_id, {})[mode] = entry
    return data


def load_physics(csv_path: Path, images_dir: Path, results_root: Path) -> dict:
    """Returns {asset_id: {mode: {metrics + gif_path}}}"""
    data = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            asset_id = r["asset_id"]
            mode = r["collision_mode"]
            gif = images_dir / mode / f"{asset_id}.gif"
            rel_gif = str(gif.relative_to(results_root)) if gif.exists() else None
            entry = {
                "physics_settles":       _parse_bool(r["physics_settles"]),
                "displacement_m":        _parse_float(r["displacement_m"]),
                "flies_away":            _parse_bool(r["flies_away"]),
                "final_y_offset_m":      _parse_float(r["final_y_offset_m"]),
                "sinks_permanently":     _parse_bool(r["sinks_permanently"]),
                "min_y_offset_m":        _parse_float(r["min_y_offset_m"]),
                "sinks_below_floor":     _parse_bool(r["sinks_below_floor"]),
                "contact_points_at_rest":_parse_int(r["contact_points_at_rest"]),
                "error":                 r["error"] or None,
                "gif":                   rel_gif,
            }
            data.setdefault(asset_id, {})[mode] = entry
    return data


# ── Merge ────────────────────────────────────────────────────────────────────

def merge_checks(results_dir: Path) -> dict:
    assets: dict = {}

    def _update(asset_id, check_name, payload):
        assets.setdefault(asset_id, {})[check_name] = payload

    # Physics
    physics_csv = results_dir / "physics" / "physics_results.csv"
    if physics_csv.exists():
        physics_images = results_dir / "physics" / "images"
        print(f"Loading physics results from {physics_csv}")
        for asset_id, modes in load_physics(physics_csv, physics_images, results_dir).items():
            _update(asset_id, "physics", modes)
    else:
        print(f"  (no physics CSV found at {physics_csv})")

    # Graspability
    grasp_csv = results_dir / "graspability" / "graspability_results.csv"
    if grasp_csv.exists():
        grasp_images = results_dir / "graspability" / "images"
        print(f"Loading graspability results from {grasp_csv}")
        for asset_id, modes in load_graspability(grasp_csv, grasp_images, results_dir).items():
            _update(asset_id, "graspability", modes)
    else:
        print(f"  (no graspability CSV found at {grasp_csv})")

    return assets


# ── Public API ───────────────────────────────────────────────────────────────

def build(results_dir: Path, out: Path) -> None:
    """Merge all experiment CSVs under results_dir and write results.json to out."""
    assets = merge_checks(results_dir)
    payload = {
        "results_dir": str(results_dir.resolve()),
        "assets": assets,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    n_physics = sum(1 for a in assets.values() if "physics" in a)
    print(f"results.json: {len(assets)} assets ({n_physics} with physics) → {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build results.json from experiment CSVs")
    parser.add_argument("--results-dir", required=True, type=Path,
                        help="Root results directory containing physics/, graspability/, etc.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output path for results.json")
    args = parser.parse_args()
    build(args.results_dir, args.out)


if __name__ == "__main__":
    main()
