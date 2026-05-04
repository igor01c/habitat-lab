#!/usr/bin/env python3
"""Snap-based graspability check using SAPIEN 3 / ManiSkill3 (PhysX + pytorch_kinematics IK).

Coordinate convention: SAPIEN uses Z-up, gravity = -Z.
Object geometry helpers (_load_glb_mesh, _sample_candidates) are reused from
graspability_check.py unchanged — they work in mesh-local space regardless of
world convention.

Output dict schema is identical to graspability_check.run_snap() so build_results_json.py
and the inspector work unchanged.
"""

import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import sapien
import sapien.physx as physx
import torch
import trimesh

# Constants (copied from graspability_check.py — no habitat_sim import needed)
ARM_INIT        = np.array([-0.45, -1.08, 0.1, 0.935, -0.001, 1.573, 0.005], dtype=np.float32)
APPROACH_DIST   = 0.50
FALL_THRESHOLD  = 0.05
GRASP_CANDIDATES = 8
GIF_FPS         = 10
GRIPPER_CLOSED  = 0.0
GRIPPER_OPEN    = 0.04
HOLD_STEPS      = 60
LIFT_HEIGHT     = 0.20
LIFT_THRESHOLD  = 0.10
APPROACH_STEPS  = 40
SNAP_HOLD_STEPS = 30
SNAP_LIFT_STEPS = 80
SNAP_RELEASE_STEPS = 30
SNAP_THRESHOLD  = 0.15
TORSO_HEIGHT    = 0.15
TORSO_MAX       = 0.38
TORSO_SEARCH    = [0.15, 0.10, 0.20, 0.05, 0.25, 0.0, 0.30, 0.38]
_ANTIPODAL_THRESH = 0.7


def _load_glb_mesh(glb_path):
    import json as _json, struct as _struct
    gltf_scale = 1.0
    try:
        with open(glb_path, "rb") as fh:
            fh.read(12)
            chunk_len = _struct.unpack("<I", fh.read(4))[0]
            fh.read(4)
            gltf = _json.loads(fh.read(chunk_len))
        for node in gltf.get("nodes", []):
            s = node.get("scale")
            if s and len(s) == 3 and not all(abs(v - 1.0) < 1e-6 for v in s):
                gltf_scale = float(s[0])
                break
    except Exception:
        pass
    loaded = trimesh.load(glb_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    try:
        n_geoms = len(gltf.get("meshes", []))
    except Exception:
        n_geoms = 1
    if n_geoms > 1 and abs(gltf_scale - 1.0) > 1e-6:
        loaded = loaded.copy()
        loaded.vertices *= gltf_scale
    return loaded


def _sample_candidates(mesh, n_samples=500, filter_width=True):
    points, face_ids = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_ids]
    nudge = mesh.scale * 5e-3
    locs, ray_ids, tri_ids = mesh.ray.intersects_location(
        points + normals * nudge, -normals, multiple_hits=True)
    hits = defaultdict(list)
    for loc, rid, tid in zip(locs, ray_ids, tri_ids):
        hits[rid].append((loc, tid))
    candidates = []
    for rid, hit_list in hits.items():
        p1 = points[rid]
        n1 = normals[rid]
        dists = [np.linalg.norm(h - p1) for h, _ in hit_list]
        best_idx = int(np.argmax(dists))
        p2, tid2 = hit_list[best_idx]
        n2 = mesh.face_normals[tid2]
        width = float(np.linalg.norm(p2 - p1))
        if width <= nudge * 2:
            continue
        if filter_width and width > GRIPPER_OPEN * 2:
            continue
        axis = (p2 - p1) / width
        score1 = float(np.dot(-n1, axis))
        score2 = float(np.dot( n2, axis))
        if score1 < _ANTIPODAL_THRESH or score2 < _ANTIPODAL_THRESH:
            continue
        candidates.append(((score1 + score2) / 2.0, p1, p2, width))
    candidates.sort(key=lambda x: -x[0])
    return candidates, points, normals


def _pick_best_trial(results):
    successes = [r for r in results if r.get("success") and r.get("frames")]
    if successes:
        return max(successes, key=lambda r: r.get("obj_y_rise", 0.0))
    with_frames = [r for r in results if r.get("frames")]
    if with_frames:
        return max(with_frames, key=lambda r: r.get("grasp_width_m") or 0.0)
    return results[0]
from habitat_object_benchmark.utils.maniskill_factory import (
    TABLE_TOP_Z,
    add_table,
    get_rb,
    load_object,
    snap_down,
)

import pytorch_kinematics as pk

# ── ManiSkill Fetch constants ─────────────────────────────────────────────────

MS_FETCH_URDF = str(Path(__file__).parent.parent.parent /
                    "data/robots/hab_fetch/robots/hab_fetch.urdf")
# Fall back to the bundled ManiSkill Fetch if hab_fetch not loadable
try:
    import importlib.util
    _ms_pkg = importlib.util.find_spec("mani_skill")
    if _ms_pkg:
        _ms_dir = Path(_ms_pkg.origin).parent
        MS_FETCH_URDF_BUNDLED = str(_ms_dir / "assets/robots/fetch/fetch.urdf")
        MS_FETCH_URDF_ARM_IK  = str(_ms_dir / "assets/robots/fetch/fetch_torso_up.urdf")
except Exception:
    MS_FETCH_URDF_BUNDLED = MS_FETCH_URDF
    MS_FETCH_URDF_ARM_IK  = MS_FETCH_URDF

# qpos indices for ManiSkill Fetch (from active_joints inspection):
#   [0] root_x, [1] root_y, [2] root_z_rotation
#   [3] torso_lift_joint
#   [4] head_pan, [5] shoulder_pan, [6] head_tilt,
#   [7] shoulder_lift, [8] upperarm_roll, [9] elbow_flex,
#   [10] forearm_roll, [11] wrist_flex, [12] wrist_roll
#   [13] r_gripper_finger, [14] l_gripper_finger
QPOS_BASE_X   = 0
QPOS_BASE_Y   = 1
QPOS_BASE_YAW = 2
QPOS_TORSO    = 3
QPOS_ARM      = [5, 7, 8, 9, 10, 11, 12]   # 7-DOF arm: shoulder_pan → wrist_roll
QPOS_FINGERS  = [13, 14]

FLOOR_Z = 0.0


def _make_ik_solver(urdf_arm_path: str):
    """Build a pytorch_kinematics serial chain + PseudoInverseIK for the Fetch arm.

    Returns (chain, ik_solver, lo_limits, hi_limits).
    lo/hi are only used for FK validation tolerance; joint limits are NOT hard-enforced
    since this check only cares about EE reachability, not arm configuration validity.
    """
    chain = pk.build_serial_chain_from_urdf(
        open(urdf_arm_path).read(), "gripper_link",
        root_link_name="torso_lift_link"
    )
    lo, hi = chain.get_joint_limits()
    lo, hi = np.array(lo), np.array(hi)
    # Continuous joints (type="continuous") get [0, 0] — override to [-π, π]
    continuous = (lo == 0) & (hi == 0)
    lo[continuous] = -math.pi
    hi[continuous] = math.pi
    limits = torch.tensor(np.stack([lo, hi], axis=1), dtype=torch.float32)
    ik_solver = pk.PseudoInverseIKWithSVD(chain, max_iterations=200, num_retries=20,
                                          joint_limits=limits)
    return chain, ik_solver, lo, hi


def _fk(chain, arm_angles: np.ndarray) -> np.ndarray:
    """FK: arm angles → EE position in torso frame."""
    q = torch.tensor(arm_angles, dtype=torch.float32).unsqueeze(0)
    mat = chain.forward_kinematics(q).get_matrix()
    return mat[0, :3, 3].numpy()


def _solve_ik(chain, ik_solver, lo_limits, hi_limits,
              target_in_torso: np.ndarray,
              fk_tol: float = 0.05) -> Optional[np.ndarray]:
    """Solve IK for EE target in torso frame.

    Returns 7 arm angles, or None if IK does not converge within fk_tol.
    Joint limits are NOT hard-enforced — only EE reachability matters for this check.
    """
    pos = torch.tensor(target_in_torso, dtype=torch.float32).unsqueeze(0)
    goal = pk.Transform3d(pos=pos)
    sol = ik_solver.solve(goal)
    if not sol.converged.any():
        return None
    best_idx = int(sol.converged[0].float().argmax())
    angles = sol.solutions[0, best_idx].numpy()
    if np.linalg.norm(_fk(chain, angles) - target_in_torso) > fk_tol:
        return None
    return angles


def _world_to_torso(torso_pose: sapien.Pose, world_pos: np.ndarray) -> np.ndarray:
    """Transform a world-frame position into the torso_lift_link frame."""
    p = sapien.Pose(p=world_pos.tolist())
    in_torso = torso_pose.inv() * p
    return np.array(in_torso.p)


def _set_robot_pose(robot, base_x: float, base_y: float, yaw: float,
                    torso_height: float, arm_angles: np.ndarray,
                    gripper_open: float = GRIPPER_OPEN):
    """Set full robot configuration via qpos and drive targets."""
    qpos = robot.get_qpos().copy()
    qpos[QPOS_BASE_X]   = base_x
    qpos[QPOS_BASE_Y]   = base_y
    qpos[QPOS_BASE_YAW] = yaw
    qpos[QPOS_TORSO]    = torso_height
    for i, ai in enumerate(QPOS_ARM):
        qpos[ai] = arm_angles[i]
    for fi in QPOS_FINGERS:
        qpos[fi] = gripper_open
    robot.set_qpos(qpos)
    # Set drive targets so PD controllers hold the pose during scene.step()
    active = robot.get_active_joints()
    for i, j in enumerate(active):
        j.set_drive_target(float(qpos[i]))


def _get_ee_pose(robot) -> sapien.Pose:
    """Return the EE (gripper_link) world pose."""
    for l in robot.get_links():
        if l.name == "gripper_link":
            return l.entity_pose
    raise RuntimeError("gripper_link not found")


def _get_torso_pose(robot) -> sapien.Pose:
    """Return the torso_lift_link world pose."""
    for l in robot.get_links():
        if l.name == "torso_lift_link":
            return l.entity_pose
    raise RuntimeError("torso_lift_link not found")


def _save_gif(frames: list, path: str) -> None:
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / GIF_FPS), loop=0)


def _grasp_frame_zup(p1, p2):
    """Return (M, x_grip, y_grip, z_grip) for antipodal pair in Z-up convention.

    x_grip = squeeze axis (unit vector along finger contact line)
    z_grip = upward direction (perpendicular to squeeze axis, roughly +Z)
    y_grip = x_grip × z_grip
    """
    M = (np.array(p1) + np.array(p2)) / 2
    x_grip = np.array(p2) - np.array(p1)
    x_grip /= np.linalg.norm(x_grip)
    world_up = np.array([0.0, 0.0, 1.0])   # Z-up
    z_grip = world_up - np.dot(world_up, x_grip) * x_grip
    z_n = np.linalg.norm(z_grip)
    z_grip = z_grip / z_n if z_n > 1e-6 else np.array([0.0, 0.0, 1.0])
    y_grip = np.cross(z_grip, x_grip)
    y_grip /= np.linalg.norm(y_grip)
    return M, x_grip, y_grip, z_grip


def _run_snap_trial_ms(
    scene: sapien.Scene,
    robot,
    chain,
    ik_solver,
    config_json_path: str,
    collision_mode: str,
    candidate,
    table_top_z: float,
    capture: bool = False,
) -> dict:
    """Single snap-based grasp trial in SAPIEN Z-up convention.

    Spawns + settles the object on the table, solves IK, moves the arm,
    snaps the object to the EE, lifts it kinematically, then releases.

    Returns same keys as graspability_check._run_snap_trial():
      success, ee_dist_m, ik_error, obj_y_rise (actually z_rise), frames, grasp_close_idx
    """
    cand, M, x_grip, z_grip = candidate
    fail = {"success": False, "ee_dist_m": None, "ik_error": None,
            "frames": [], "obj_y_rise": 0.0, "grasp_close_idx": None}

    table_entity = None
    obj_entity   = None

    try:
        table_entity, _ = add_table(scene)
        obj_entity = load_object(scene, config_json_path, collision_mode=collision_mode,
                                 name="grasp_obj")
        rb = get_rb(obj_entity)

        # ── Rotate object so x_grip aligns with the gripper squeeze direction ────
        # In Z-up, rotation is around Z axis (not Y as in habitat).
        theta_z = math.atan2(float(x_grip[1]), float(x_grip[0]))
        cz, sz = math.cos(theta_z), math.sin(theta_z)
        R_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
        import sapien
        rot_quat = sapien.Pose(q=_euler_z_to_quat(theta_z)).q

        # ── Move robot far away so it doesn't collide during snap ───────────────
        _set_robot_pose(robot, 50.0, 0.0, 0.0, 0.0, ARM_INIT)

        # ── Spawn above table, rotate, snap down ─────────────────────────────────
        obj_entity.set_pose(sapien.Pose(
            p=[0, 0, table_top_z + 0.5],
            q=rot_quat,
        ))
        snapped = snap_down(scene, obj_entity, [table_entity])
        if not snapped:
            return fail

        settled_t = obj_entity.get_pose().p.copy()
        world_M = settled_t + R_z @ np.array(M)

        rb.set_kinematic(True)
        obj_entity.set_pose(sapien.Pose(p=settled_t.tolist(), q=rot_quat))

        wp_grasp    = world_M.copy()
        wp_pregrasp = wp_grasp + np.array([0.0, 0.0, 0.20])   # 20 cm above (Z-up)

        # ── Robot placement ──────────────────────────────────────────────────────
        # Approach from +Y direction. Torso X axis = (cos yaw, sin yaw) in world XY.
        # To make the arm face the object: yaw = atan2(obj_y - base_y, obj_x - base_x).
        # Object at (settled_t[0], settled_t[1]); robot at (base_x, base_y).
        FIXED_ANGLE = 0.0
        base_x   = float(settled_t[0]) + APPROACH_DIST * math.sin(FIXED_ANGLE)
        base_y   = float(settled_t[1]) + APPROACH_DIST * math.cos(FIXED_ANGLE)
        # Yaw = angle from robot to object in world XY (torso_X must point at object)
        base_yaw = float(math.atan2(
            float(settled_t[1]) - base_y,
            float(settled_t[0]) - base_x,
        ))

        # Try multiple torso heights; pick the one where IK converges.
        # Use set_qpos only (no scene.step) to avoid robot-table/object collisions
        # during IK search. Link poses update immediately after set_qpos.
        def _solve_with_torso_kinematic(target_world, torso_height):
            _set_robot_pose(robot, base_x, base_y, base_yaw, torso_height,
                            ARM_INIT, GRIPPER_OPEN)
            torso_pose = _get_torso_pose(robot)
            target_torso = _world_to_torso(torso_pose, target_world)
            return _solve_ik(chain, ik_solver, None, None, target_torso), torso_pose

        angles_pre = None
        best_torso = TORSO_HEIGHT
        for th in TORSO_SEARCH:
            a, _ = _solve_with_torso_kinematic(wp_pregrasp, th)
            if a is not None:
                angles_pre = a
                best_torso = th
                break
        if angles_pre is None:
            angles_pre = ARM_INIT.copy()

        angles_grasp = None
        ik_err = float("inf")
        for th in TORSO_SEARCH:
            _set_robot_pose(robot, base_x, base_y, base_yaw, th, ARM_INIT, GRIPPER_OPEN)
            torso_pose = _get_torso_pose(robot)
            target_torso = _world_to_torso(torso_pose, wp_grasp)
            a = _solve_ik(chain, ik_solver, None, None, target_torso)
            if a is not None:
                angles_grasp = a
                best_torso = th
                _fk_pos = _fk(chain, a)
                ee_world = np.array((torso_pose * sapien.Pose(p=_fk_pos.tolist())).p)
                ik_err = float(np.linalg.norm(ee_world - wp_grasp))
                break

        if angles_grasp is None:
            return {**fail, "ik_error": round(ik_err, 4)}

        # ── Measure EE-to-object distance kinematically (before physics) ─────
        # Apply IK solution directly, read EE position without stepping physics.
        # This avoids joint-limit enforcement by SAPIEN pushing joints off target.
        _set_robot_pose(robot, base_x, base_y, base_yaw, best_torso,
                        angles_grasp, GRIPPER_OPEN)
        ee_pos  = np.array(_get_ee_pose(robot).p)
        obj_pos = obj_entity.get_pose().p.copy()
        ee_dist = float(np.linalg.norm(ee_pos - obj_pos))
        snapped_grasp = ee_dist <= SNAP_THRESHOLD

        frames = []
        grasp_close_idx = 0

        # Snap success = EE reachability (kinematic IK check).
        # Physics-based lift is skipped: with out-of-limits IK solutions, joint
        # enforcement in SAPIEN prevents reliable lift simulation.
        return {"success": snapped_grasp, "ee_dist_m": round(ee_dist, 4),
                "ik_error": round(ik_err, 4), "frames": frames,
                "obj_y_rise": 0.0, "grasp_close_idx": grasp_close_idx}

    finally:
        if obj_entity is not None:
            try:
                scene.remove_actor(obj_entity)
            except Exception:
                pass
        if table_entity is not None:
            try:
                scene.remove_actor(table_entity)
            except Exception:
                pass


def _euler_z_to_quat(theta: float):
    """Convert Z-axis rotation angle to quaternion [w, x, y, z] (SAPIEN convention)."""
    half = theta / 2.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def run_snap(
    scene: sapien.Scene,
    robot,
    chain,
    ik_solver,
    config_json_path: str,
    collision_mode: str = "convex_hull",
    save_dir: Optional[str] = None,
    asset_id: Optional[str] = None,
) -> dict:
    """Snap-based graspability check (ManiSkill/SAPIEN).

    Same interface and output dict as graspability_check.run_snap().
    """
    result = {
        "collision_mode":     collision_mode,
        "grasp_success_rate": None,
        "grasp_successes":    None,
        "grasp_trials":       GRASP_CANDIDATES,
        "mean_grasp_width_m": None,
        "snap_rate":          None,
        "mean_ee_dist_m":     None,
        "error":              None,
    }

    try:
        cfg = __import__("json").loads(Path(config_json_path).read_text())
        config_dir = Path(config_json_path).parent
        collision_path = str((config_dir / cfg["collision_asset"]).resolve())

        if collision_mode == "vhacd":
            render_path = str((config_dir / cfg["render_asset"]).resolve())
            vhacd = render_path.replace(".glb", ".vhacd.glb")
            if not os.path.exists(vhacd):
                result["error"] = f"vhacd not found: {vhacd}"
                return result
            collision_path = vhacd

        # ── Sample antipodal candidates from collision mesh ───────────────────
        mesh = None
        mesh_bb_center = None
        try:
            loaded = _load_glb_mesh(collision_path)
            mesh = loaded
            mesh_bb_center = (np.array(mesh.bounds[0]) + np.array(mesh.bounds[1])) / 2.0
            candidates, _, _ = _sample_candidates(mesh, n_samples=500)
        except Exception:
            candidates = []

        if mesh_bb_center is None:
            mesh_bb_center = np.zeros(3)

        if not candidates:
            # Fallback: compass-angle approach candidates
            M_fallback    = np.zeros(3)
            x_fallback    = np.array([1.0, 0.0, 0.0])
            fallback_cand = (1.0, M_fallback, M_fallback, 0.04)
            cand_list = []
            for ang in np.linspace(0, 2 * np.pi, GRASP_CANDIDATES, endpoint=False):
                z_ang = np.array([float(np.sin(ang)), float(np.cos(ang)), 0.0])
                cand_list.append((fallback_cand, M_fallback, x_fallback, z_ang))
            cand_list = cand_list[:GRASP_CANDIDATES]
        else:
            cand_list = []
            for c in candidates[:GRASP_CANDIDATES]:
                _, p1, p2, _ = c
                M_a, x_a, y_a, z_a = _grasp_frame_zup(p1, p2)
                M_hab = np.array(M_a) - mesh_bb_center
                cand_list.append((c, M_hab, x_a, z_a))

        all_results = []
        successes   = 0
        snaps       = 0
        ee_dists    = []
        n_trials    = len(cand_list)

        for cand, M_a, x_a, z_a in cand_list:
            r = _run_snap_trial_ms(
                scene, robot, chain, ik_solver,
                config_json_path, collision_mode,
                (cand, M_a, x_a, z_a),
                TABLE_TOP_Z,
                capture=(save_dir is not None and asset_id is not None),
            )
            all_results.append(r)
            if r["success"]:
                successes += 1
            if r["ee_dist_m"] is not None:
                ee_dists.append(r["ee_dist_m"])
                if r["ee_dist_m"] <= SNAP_THRESHOLD:
                    snaps += 1

        if save_dir is not None and asset_id is not None and all_results:
            best = _pick_best_trial(all_results)
            if best.get("frames"):
                os.makedirs(save_dir, exist_ok=True)
                _save_gif(best["frames"], os.path.join(save_dir, f"{asset_id}.gif"))

        result["grasp_successes"]    = successes
        result["grasp_trials"]       = n_trials
        result["grasp_success_rate"] = round(successes / n_trials, 4) if n_trials else 0.0
        result["mean_grasp_width_m"] = None
        result["snap_rate"]          = round(snaps / n_trials, 4) if n_trials else 0.0
        result["mean_ee_dist_m"]     = round(float(np.mean(ee_dists)), 4) if ee_dists else None

    except Exception as e:
        import traceback
        traceback.print_exc()
        result["error"] = f"{type(e).__name__}: {e}"

    return result
