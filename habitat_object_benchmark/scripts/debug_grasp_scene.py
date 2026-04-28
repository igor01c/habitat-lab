#!/usr/bin/env python3
"""Debug: spawn table + asset + robot, capture frame."""

import argparse
import os

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["GLOG_minloglevel"] = "5"

import sys
from pathlib import Path

import magnum as mn
import numpy as np
import quaternion as qt
from PIL import Image

import habitat_sim

TABLE_CFG = "data/versioned_data/replica_cad_dataset/configs/objects/frl_apartment_table_01.object_config.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-config", required=True, type=Path)
    parser.add_argument("--out", default="frame.png", type=Path)
    args = parser.parse_args()

    # ── Simulator ─────────────────────────────────────────────────────────────
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "NONE"
    backend_cfg.enable_physics = True
    backend_cfg.create_renderer = True

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "color"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [480, 640]
    sensor.position = mn.Vector3(0, 0, 0)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]
    agent_cfg.height = 0.0

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))

    otm = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()

    # ── Robot (load first — reconfigure() resets the sim) ─────────────────────
    print("Loading robot...")
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from habitat_object_benchmark.utils.sim_factory import load_fetch_robot
    robot = load_fetch_robot(sim)

    # Re-fetch managers after reconfigure
    otm = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()

    # ── Floor ──────────────────────────────────────────────────────────────────
    cube = otm.get_template_handles("cubeSolid")[0]
    tmpl = otm.get_template_by_handle(cube)
    tmpl.scale = mn.Vector3(10.0, 0.05, 10.0)
    otm.register_template(tmpl, "__floor__")
    floor_obj = rom.add_object_by_template_handle("__floor__")
    floor_obj.translation = mn.Vector3(0, -0.025, 0)   # top surface at y=0
    floor_obj.motion_type = habitat_sim.physics.MotionType.STATIC

    # ── Table ─────────────────────────────────────────────────────────────────
    print("Loading table...")
    otm.load_configs(TABLE_CFG)
    table_handles = otm.get_template_handles("frl_apartment_table_01")
    if not table_handles:
        raise RuntimeError("Table not found")
    table = rom.add_object_by_template_handle(table_handles[0])
    bb = table.root_scene_node.cumulative_bb
    table.translation = mn.Vector3(0, -float(bb.min[1]), 0)   # place bottom at y=0
    table.motion_type = habitat_sim.physics.MotionType.STATIC  # lock after placing
    table_top_y = -float(bb.min[1]) + float(bb.max[1])
    print(f"  table top Y = {table_top_y:.3f} m")

    # ── Asset on table ────────────────────────────────────────────────────────
    print(f"Loading asset: {args.asset_config.stem}")
    otm.load_configs(str(args.asset_config))
    asset_id = args.asset_config.stem.replace(".object_config", "")
    asset_handles = otm.get_template_handles(asset_id)
    if not asset_handles:
        raise RuntimeError(f"Asset not found: {asset_id}")
    asset = rom.add_object_by_template_handle(asset_handles[0])
    abb = asset.root_scene_node.cumulative_bb
    asset.translation = mn.Vector3(0, table_top_y - float(abb.min[1]), 0)
    asset.motion_type = habitat_sim.physics.MotionType.STATIC
    print(f"  asset at y={float(asset.translation[1]):.3f}, height={abb.size_y():.3f} m")

    # Place robot 1.0 m from the table along +Z, oriented to face the asset
    robot_dist = 1.0
    robot_base = np.array([0.0, 0.0, robot_dist])
    asset_pos  = np.array([0.0, 0.0, 0.0])   # table/asset is at XZ origin
    dx = asset_pos[0] - robot_base[0]
    dz = asset_pos[2] - robot_base[2]
    # arctan2(dx, -dz) gives the yaw to face the asset; +π flips to correct convention
    robot_yaw = float(np.arctan2(dx, -dz)) + np.pi / 2

    robot.base_pos = mn.Vector3(*robot_base)
    robot.base_rot = robot_yaw
    robot.update()
    print(f"  robot at {robot.base_pos}, yaw={np.degrees(robot_yaw):.1f}°")

    # ── Union bounding box (world space) ─────────────────────────────────────
    def rigid_world_aabb(obj):
        """World AABB for a RigidObject: local bb + object translation (no rotation)."""
        bb  = obj.root_scene_node.cumulative_bb
        t   = obj.translation
        return (mn.Vector3(bb.min[0] + t[0], bb.min[1] + t[1], bb.min[2] + t[2]),
                mn.Vector3(bb.max[0] + t[0], bb.max[1] + t[1], bb.max[2] + t[2]))

    def link_world_aabb(link_node):
        """World AABB for one articulated-object link (uses absolute_transformation)."""
        bb  = link_node.cumulative_bb
        T   = link_node.absolute_transformation()
        pts = [T.transform_point(mn.Vector3(x, y, z))
               for x in (bb.min[0], bb.max[0])
               for y in (bb.min[1], bb.max[1])
               for z in (bb.min[2], bb.max[2])]
        return (mn.Vector3(min(p[0] for p in pts), min(p[1] for p in pts), min(p[2] for p in pts)),
                mn.Vector3(max(p[0] for p in pts), max(p[1] for p in pts), max(p[2] for p in pts)))

    asset_bb  = rigid_world_aabb(asset)
    asset_max_y = float(asset_bb[1][1])

    ao = robot.sim_obj
    robot_bbs = []
    for link_id in range(-1, ao.num_links):
        mn_bb, mx_bb = link_world_aabb(ao.get_link_scene_node(link_id))
        if (mx_bb - mn_bb).length() < 10.0:
            robot_bbs.append((mn_bb, mx_bb))
    robot_max_y = max(float(b[1][1]) for b in robot_bbs)

    bbs = [rigid_world_aabb(table), asset_bb] + robot_bbs

    union_min = mn.Vector3(min(b[0][0] for b in bbs),
                           min(b[0][1] for b in bbs),
                           min(b[0][2] for b in bbs))
    union_max = mn.Vector3(max(b[1][0] for b in bbs),
                           max(b[1][1] for b in bbs),
                           max(b[1][2] for b in bbs))
    union_size   = union_max - union_min
    union_centre = (union_min + union_max) * 0.5
    print(f"\nUnion bounding box (robot + asset + table):")
    print(f"  min    = ({union_min[0]:.3f}, {union_min[1]:.3f}, {union_min[2]:.3f})")
    print(f"  max    = ({union_max[0]:.3f}, {union_max[1]:.3f}, {union_max[2]:.3f})")
    print(f"  size   = {union_size[0]:.3f} x {union_size[1]:.3f} x {union_size[2]:.3f} m")
    print(f"  centre = ({union_centre[0]:.3f}, {union_centre[1]:.3f}, {union_centre[2]:.3f})")

    # ── Camera: auto-framed from union bounding box ───────────────────────────
    import math
    hfov_rad = math.radians(90.0)          # sensor default hfov
    aspect   = 640 / 480
    vfov_rad = 2 * math.atan(math.tan(hfov_rad / 2) / aspect)

    half_w = (union_max[0] - union_min[0]) / 2
    half_h = (union_max[1] - union_min[1]) / 2
    # Distance needed so the bbox fits within each FOV axis
    d_horiz = half_w / math.tan(hfov_rad / 2)
    d_vert  = half_h / math.tan(vfov_rad / 2)
    margin  = 1.3   # 30 % breathing room
    cam_dist = max(d_horiz, d_vert) * margin

    cam_height = (asset_max_y + robot_max_y) / 2.0
    print(f"\nCamera height: midpoint(asset_max_y={asset_max_y:.3f}, robot_max_y={robot_max_y:.3f}) = {cam_height:.3f}")

    cam_pos    = np.array([float(union_max[0]) + cam_dist, cam_height, 0.0])
    asset_centre = np.array([(asset_bb[0][0] + asset_bb[1][0]) / 2,
                              (asset_bb[0][1] + asset_bb[1][1]) / 2,
                              (asset_bb[0][2] + asset_bb[1][2]) / 2])
    lookat_pos = asset_centre
    print(f"cam_pos  = {cam_pos.round(3)}")
    print(f"look_at  = {lookat_pos.round(3)}")

    # Compute orientation quaternion from cam_pos → lookat_pos via look_at matrix
    look_mat = mn.Matrix4.look_at(
        mn.Vector3(*cam_pos),
        mn.Vector3(*lookat_pos),
        mn.Vector3(0, 1, 0),
    )
    mq = mn.Quaternion.from_matrix(look_mat.rotation())
    cam_rot = qt.quaternion(float(mq.scalar), float(mq.vector[0]), float(mq.vector[1]), float(mq.vector[2]))

    agent_state = sim.get_agent(0).get_state()
    agent_state.position = mn.Vector3(*cam_pos)
    agent_state.rotation = cam_rot
    sim.get_agent(0).set_state(agent_state)

    # ── Render ────────────────────────────────────────────────────────────────
    obs = sim.get_sensor_observations()
    img = Image.fromarray(obs["color"][:, :, :3])
    img.save(args.out)
    print(f"Saved → {args.out}")
    sim.close()


if __name__ == "__main__":
    main()
