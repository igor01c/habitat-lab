#!/usr/bin/env python3
"""Smoke test: load a .object_config.json asset into SAPIEN and step physics."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import sapien.physx as physx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to .object_config.json")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--collision-mode", choices=["convex_hull", "vhacd"], default="convex_hull")
    args = parser.parse_args()

    import sapien
    print(f"SAPIEN version: {sapien.__version__}")

    # --- Parse config ---
    cfg = json.loads(args.config.read_text())
    config_parent = args.config.parent  # directory containing the .object_config.json
    render_path = str((config_parent / cfg["render_asset"]).resolve())
    if args.collision_mode == "vhacd":
        vhacd = render_path.replace(".glb", ".vhacd.glb")
        collision_path = vhacd if os.path.exists(vhacd) else render_path
    else:
        collision_path = str((config_parent / cfg.get("collision_asset", cfg["render_asset"])).resolve())

    print(f"Render asset:    {render_path}")
    print(f"Collision asset: {collision_path}")

    # --- Build scene (headless, no renderer) ---
    # SAPIEN uses Z-up, -Z gravity by default
    scene = sapien.Scene()
    scene.set_timestep(1 / 60)
    scene.add_ground(altitude=0.0)

    # --- Load object ---
    builder = scene.create_actor_builder()
    builder.add_visual_from_file(render_path)
    if args.collision_mode == "vhacd":
        builder.add_multiple_convex_collisions_from_file(collision_path)
    else:
        builder.add_multiple_convex_collisions_from_file(collision_path)
    actor = builder.build(name="test_object")
    actor.set_pose(sapien.Pose(p=[0, 0, 1.0]))  # drop from 1m above ground (Z-up)

    print(f"Initial pose: {actor.get_pose()}")

    # --- Step physics ---
    for i in range(args.steps):
        scene.step()

    final_pose = actor.get_pose()
    rb = actor.find_component_by_type(physx.PhysxRigidDynamicComponent)
    vel = rb.get_linear_velocity()
    print(f"Final pose after {args.steps} steps: {final_pose}")
    print(f"Final velocity magnitude: {np.linalg.norm(vel):.4f} m/s")
    print(f"Final Z position: {final_pose.p[2]:.4f} m (Z-up convention)")

    # Sanity check: object should have settled near ground (Z close to 0)
    if final_pose.p[2] < 0.3:
        print("PASS: object fell to ground as expected")
    else:
        print("WARN: object still high — physics may not have run correctly")


if __name__ == "__main__":
    main()
