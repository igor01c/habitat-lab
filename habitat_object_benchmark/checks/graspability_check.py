#!/usr/bin/env python3
"""Graspability check using Fetch robot IK + physics gripper.

For each of N approach angles around the object:
  1. Solve IK (PyBullet) for the arm to reach the object at GRASP_HEIGHT.
     If IK fails → trial failed.
  2. Arm interpolates from rest → grasp pose over APPROACH_STEPS (visible motion).
  3. Object is placed at the EE (kinematic) while the arm settles.
  4. Object switches to dynamic; gripper closes over CLOSE_STEPS.
  5. Gripper holds closed for HOLD_STEPS (~1 s) under gravity.
  6. Drift = distance from EE to object. Success = drift < DRIFT_THRESHOLD.
  7. Gripper opens; object drops (shown in GIF).

What it tests:
  - Reachability: can the arm geometry reach this object from each approach angle?
  - Grasp stability: does the collision mesh shape allow the gripper fingers to
    physically hold the object, or does it slip / get thrown away?

Coordinate systems:
  habitat-sim  →  Y-up   (x, y, z)
  PyBullet     →  Y-up   (hab_fetch.urdf is Y-up in both engines — no swap needed)
"""

import os
import sys

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("GLOG_minloglevel", "5")

import math
from collections import defaultdict

import habitat_sim
import magnum as mn
import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "habitat-lab"))
from habitat.sims.habitat_simulator.sim_utilities import snap_down

# ── Constants ────────────────────────────────────────────────────────────────

APPROACH_DIST   = 0.80    # robot base distance from object (m)
GRASP_CANDIDATES = 8      # number of antipodal candidates to attempt
IK_ERROR_THRESH = 0.02    # m — IK convergence threshold

# Lift-based success criterion
LIFT_HEIGHT     = 0.20    # target EE lift after grasp (m)
LIFT_THRESHOLD  = 0.10    # object must rise at least this much to count as lifted
FALL_THRESHOLD  = 0.05    # object must drop at least this much when gripper opens
TORSO_MAX       = 0.38    # maximum torso_lift_joint value

# Gripper animation steps (at 1/60 s each)
APPROACH_STEPS  = 40      # arm moves from rest to grasp pose
HOLD_STEPS      = 60      # hold after close before lift
LIFT_STEPS      = 80      # arm/torso raises 20 cm
RELEASE_STEPS   = 30      # gripper opens; object falls (captured in GIF)

GRIPPER_OPEN    = 0.04    # each finger position when open (m)
GRIPPER_CLOSED  = 0.0     # each finger position when closed (m)

GIF_FRAMES      = 30
GIF_FPS         = 10

FLOOR_Y         = 0.0
TORSO_HEIGHT    = 0.15    # torso_lift_joint default (matches habitat robot reconfigure)
TORSO_SEARCH    = [0.15, 0.10, 0.20, 0.05, 0.25, 0.0, 0.30, 0.38]

TABLE_CFG       = "data/versioned_data/replica_cad_dataset/configs/objects/frl_apartment_table_01.object_config.json"
ROBOT_DIST      = 1.0     # robot base distance from table/asset (m)

# PyBullet joint indices (from URDF inspection)
PB_EE_LINK          = 17               # gripper_link
PB_ARM_JOINTS       = [10, 11, 12, 13, 14, 15, 16]   # shoulder_pan→wrist_roll
PB_TORSO_JOINT      = 2

ARM_INIT = np.array([-0.45, -1.08, 0.1, 0.935, -0.001, 1.573, 0.005],
                    dtype=np.float32)

_ANTIPODAL_THRESH = 0.7   # min cos(angle) for antipodal score


# ── Antipodal grasp helpers ───────────────────────────────────────────────────

def _load_glb_mesh(glb_path):
    """Load a GLB as a single Trimesh, correctly applying any GLTF world-node scale.

    trimesh's force='mesh' applies the node transform for single-geometry scenes
    but silently skips it for multi-geometry scenes (vhacd convex decompositions).
    This function handles both cases by reading the GLTF JSON scale directly.
    """
    import json as _json, struct as _struct

    # Read world-node scale from the embedded GLTF JSON.
    gltf_scale = 1.0
    try:
        with open(glb_path, "rb") as fh:
            fh.read(12)  # magic + version + total_length
            chunk_len = _struct.unpack("<I", fh.read(4))[0]
            fh.read(4)   # chunk_type
            gltf = _json.loads(fh.read(chunk_len))
        for node in gltf.get("nodes", []):
            s = node.get("scale")
            if s and len(s) == 3 and not all(abs(v - 1.0) < 1e-6 for v in s):
                gltf_scale = float(s[0])  # assume uniform
                break
    except Exception:
        pass

    loaded = trimesh.load(glb_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)

    # force='mesh' applies the world scale correctly for single-geometry scenes,
    # but skips it for multi-geometry scenes (vhacd).  Detect by comparing the
    # loaded extent to the expected scaled extent: if force='mesh' already applied
    # the scale, the bounds will be ~gltf_scale × raw-bounds, otherwise they won't.
    # Simpler heuristic: if the GLB has more than one geometry node it's multi-mesh
    # and force='mesh' didn't apply the world scale → apply it now.
    try:
        n_geoms = len(gltf.get("meshes", []))
    except Exception:
        n_geoms = 1

    if n_geoms > 1 and abs(gltf_scale - 1.0) > 1e-6:
        loaded = loaded.copy()
        loaded.vertices *= gltf_scale

    return loaded


def _sample_candidates(mesh, n_samples=500, filter_width=True):
    """Sample antipodal pairs; return (candidates, points, normals).

    candidates: list of (score, p1, p2, width) sorted by score descending.
    filter_width=False keeps pairs wider than GRIPPER_OPEN*2 (for fallback).
    """
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


def _grasp_frame(p1, p2):
    """Return (M, x_grip, y_grip, z_grip) for an antipodal pair."""
    M = (np.array(p1) + np.array(p2)) / 2
    x_grip = np.array(p2) - np.array(p1)
    x_grip /= np.linalg.norm(x_grip)
    world_up = np.array([0.0, 1.0, 0.0])
    z_grip = world_up - np.dot(world_up, x_grip) * x_grip
    z_n = np.linalg.norm(z_grip)
    z_grip = z_grip / z_n if z_n > 1e-6 else np.array([0.0, 0.0, 1.0])
    y_grip = np.cross(z_grip, x_grip)
    y_grip /= np.linalg.norm(y_grip)
    return M, x_grip, y_grip, z_grip


def _pick_best_trial(results):
    """Return the result dict that best represents the asset's graspability."""
    successes = [r for r in results if r.get("success") and r.get("frames")]
    if successes:
        return max(successes, key=lambda r: r.get("obj_y_rise", 0.0))
    with_frames = [r for r in results if r.get("frames")]
    if with_frames:
        return max(with_frames, key=lambda r: r.get("grasp_width_m") or 0.0)
    return results[0]


def _save_grasp_html(path, mesh,
                     p1_local, p2_local, M_local,
                     x_grip_local, y_grip_local, z_grip_local,
                     snap_pos, R_y, table_top_y,
                     world_M, wp_pregrasp,
                     ee_trajectory, grasp_close_idx,
                     mesh_scale=None, mesh_bb_center=None):
    """Save an interactive Plotly HTML showing the grasp geometry + EE trajectory."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    scale = np.asarray(mesh_scale, dtype=float) if mesh_scale is not None else np.ones(3)
    # mesh_bb_center: AABB centre in (scaled) trimesh local space = CoM in local space.
    # snap_pos is the CoM in world space, so: world = snap_pos + R_y @ (v_scaled - bb_c)
    bb_c = np.asarray(mesh_bb_center, dtype=float) if mesh_bb_center is not None else np.zeros(3)

    def to_world(v):
        return snap_pos + R_y @ (np.asarray(v, dtype=float) * scale - bb_c)

    p1_w = to_world(p1_local)
    p2_w = to_world(p2_local)
    x_w  = R_y @ np.asarray(x_grip_local, dtype=float)
    y_w  = R_y @ np.asarray(y_grip_local, dtype=float)
    z_w  = R_y @ np.asarray(z_grip_local, dtype=float)

    verts_local = np.array(mesh.vertices, dtype=float) * scale
    verts_world = (R_y @ (verts_local - bb_c).T).T + snap_pos
    faces       = np.array(mesh.faces, dtype=int)

    # Derive the table-surface Y from the mesh's actual lowest world vertex.
    # This guarantees the table plane is flush with the object bottom regardless
    # of any snap_pos / table_top_y discrepancy.
    mesh_bottom_y = float(verts_world[:, 1].min())

    traces = []

    # Object mesh
    traces.append(go.Mesh3d(
        x=verts_world[:, 0], y=verts_world[:, 1], z=verts_world[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='lightblue', opacity=0.4, name='Object mesh', showlegend=True,
    ))

    # Table plane flush with the mesh bottom
    hw = 0.3
    tx, tz, ty = float(snap_pos[0]), float(snap_pos[2]), mesh_bottom_y
    tv = np.array([[tx-hw, ty, tz-hw], [tx+hw, ty, tz-hw],
                   [tx+hw, ty, tz+hw], [tx-hw, ty, tz+hw]])
    traces.append(go.Mesh3d(
        x=tv[:, 0], y=tv[:, 1], z=tv[:, 2],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color='tan', opacity=0.3, name='Table top', showlegend=True,
    ))

    # Contact points p1/p2 and squeeze line
    traces.append(go.Scatter3d(x=[p1_w[0]], y=[p1_w[1]], z=[p1_w[2]],
        mode='markers', marker=dict(size=8, color='red'), name='p1', showlegend=True))
    traces.append(go.Scatter3d(x=[p2_w[0]], y=[p2_w[1]], z=[p2_w[2]],
        mode='markers', marker=dict(size=8, color='darkred'), name='p2', showlegend=True))
    traces.append(go.Scatter3d(
        x=[p1_w[0], p2_w[0]], y=[p1_w[1], p2_w[1]], z=[p1_w[2], p2_w[2]],
        mode='lines', line=dict(color='red', width=3), name='squeeze axis', showlegend=True))

    # Grasp midpoint M
    traces.append(go.Scatter3d(x=[world_M[0]], y=[world_M[1]], z=[world_M[2]],
        mode='markers', marker=dict(size=10, color='gold', symbol='circle'),
        name='M (grasp midpoint)', showlegend=True))

    # Gripper axes as short arrows (x=firebrick, y=green, z=blue)
    axis_len = 0.05
    for aname, avec, acolor in [('x_grip', x_w, 'firebrick'), ('y_grip', y_w, 'green'), ('z_grip', z_w, 'royalblue')]:
        tip = world_M + avec * axis_len
        traces.append(go.Scatter3d(
            x=[world_M[0], tip[0]], y=[world_M[1], tip[1]], z=[world_M[2], tip[2]],
            mode='lines+markers',
            line=dict(color=acolor, width=4),
            marker=dict(size=[0, 6], color=acolor),
            name=aname, showlegend=True))

    # Pre-grasp point
    traces.append(go.Scatter3d(x=[wp_pregrasp[0]], y=[wp_pregrasp[1]], z=[wp_pregrasp[2]],
        mode='markers', marker=dict(size=10, color='orange', symbol='circle-open'),
        name='pre-grasp', showlegend=True))

    # EE trajectory
    if ee_trajectory:
        traj = np.array(ee_trajectory, dtype=float)
        traces.append(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
            mode='lines', line=dict(color='purple', width=3),
            name='EE trajectory', showlegend=True))
        if grasp_close_idx is not None and 0 <= grasp_close_idx < len(ee_trajectory):
            cp = ee_trajectory[grasp_close_idx]
            traces.append(go.Scatter3d(x=[cp[0]], y=[cp[1]], z=[cp[2]],
                mode='markers', marker=dict(size=14, color='purple', symbol='diamond'),
                name='gripper close', showlegend=True))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        title=os.path.basename(path),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.write_html(path, include_plotlyjs='cdn')


# ── PyBullet IK solver (one instance per worker) ─────────────────────────────

class FetchIKSolver:
    """Persistent PyBullet instance for fast IK solving."""

    def __init__(self, urdf_path: str):
        import pybullet as p
        self._p      = p
        self._client = p.connect(p.DIRECT)
        self._robot  = p.loadURDF(
            urdf_path, useFixedBase=True,
            physicsClientId=self._client,
        )
        p.resetJointState(self._robot, PB_TORSO_JOINT, TORSO_HEIGHT,
                          physicsClientId=self._client)

        # Build per-DOF limit arrays sized for ALL movable joints (PyBullet requirement).
        # Fixed joints are excluded from IK DOF count automatically.
        n = p.getNumJoints(self._robot, physicsClientId=self._client)
        self._dof_joints = []   # joint indices that are non-fixed
        lower_all, upper_all, ranges_all, rest_all = [], [], [], []
        arm_set = set(PB_ARM_JOINTS)
        for ji in range(n):
            info = p.getJointInfo(self._robot, ji, physicsClientId=self._client)
            if info[2] == p.JOINT_FIXED:
                continue
            self._dof_joints.append(ji)
            if ji in arm_set:
                lo, hi = info[8], info[9]
                if lo >= hi:  # continuous joint (no URDF limits)
                    lo, hi = -3.14159, 3.14159
                rest = ARM_INIT[PB_ARM_JOINTS.index(ji)]
            else:
                lo, hi, rest = -3.14159, 3.14159, 0.0
            lower_all.append(lo)
            upper_all.append(hi)
            ranges_all.append(hi - lo)
            rest_all.append(rest)

        self._lower  = lower_all
        self._upper  = upper_all
        self._ranges = ranges_all
        self._rest   = rest_all
        # Map DOF index → arm joint index within ARM_INIT
        self._dof_to_arm = {
            dof_i: PB_ARM_JOINTS.index(ji)
            for dof_i, ji in enumerate(self._dof_joints)
            if ji in arm_set
        }

        # Compute arm's natural forward direction at default pose (XZ plane, Y-up).
        # PyBullet yaw rotates around Y. The arm's "forward" is not exactly +Z,
        # so we compute the offset to correctly orient the robot toward targets.
        for k, ji in enumerate(PB_ARM_JOINTS):
            p.resetJointState(self._robot, ji, ARM_INIT[k],
                              physicsClientId=self._client)
        ee0 = np.array(p.getLinkState(
            self._robot, PB_EE_LINK, physicsClientId=self._client)[4])
        self._arm_forward_angle = float(np.arctan2(ee0[0], ee0[2]))

    @staticmethod
    def _ee_target_orn(approach_angle: float):
        """
        Desired gripper_link orientation for a horizontal side grasp.

        gripper_link axes in world frame:
          X = reach direction (from robot toward object) = (-sin α, 0, -cos α)
          Y = finger squeeze axis — we want this HORIZONTAL = (-cos α, 0,  sin α)
          Z = X × Y = (0, 1, 0)  [up]

        Keeping Y horizontal means fingers squeeze left/right, not up/down,
        so contact normals are horizontal and friction can counteract gravity.

        Returns quaternion as (x, y, z, w) for PyBullet.
        """
        a = approach_angle
        R = np.array([
            [-np.sin(a), -np.cos(a), 0.0],
            [0.0,         0.0,       1.0],
            [-np.cos(a),  np.sin(a), 0.0],
        ], dtype=float)
        t = R[0, 0] + R[1, 1] + R[2, 2]
        if t > 0:
            s = 0.5 / np.sqrt(t + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return (x, y, z, w)

    def solve(self, base_xz: np.ndarray, base_yaw: float,
              target_hab: np.ndarray, approach_angle: float = None,
              torso_height: float = None):
        """
        Solve IK for EE at target_hab (habitat Y-up coords).
        hab_fetch.urdf is Y-up in PyBullet — no coordinate swap needed.
        base_xz = (x, z) horizontal position; floor is Y=0.
        approach_angle: if given, constrains EE orientation so fingers
                        squeeze horizontally (perpendicular to approach).
        Returns (arm_angles_7, ik_error_m).
        arm_angles_7 is None if IK did not converge.
        """
        p, c, r = self._p, self._client, self._robot

        pb_base = (float(base_xz[0]), 0.0, float(base_xz[1]))
        # Subtract arm_forward_angle so the arm's natural forward aligns with base_yaw
        pb_orn  = p.getQuaternionFromEuler([0, base_yaw - self._arm_forward_angle, 0])
        p.resetBasePositionAndOrientation(r, pb_base, pb_orn,
                                          physicsClientId=c)
        # Reset torso each call (base reset clears states)
        t_height = torso_height if torso_height is not None else TORSO_HEIGHT
        p.resetJointState(r, PB_TORSO_JOINT, t_height,
                          physicsClientId=c)

        pb_target = (float(target_hab[0]), float(target_hab[1]), float(target_hab[2]))
        target_orn = (self._ee_target_orn(approach_angle)
                      if approach_angle is not None else None)
        ik = p.calculateInverseKinematics(
            r, PB_EE_LINK, pb_target,
            **({"targetOrientation": target_orn} if target_orn is not None else {}),
            lowerLimits=self._lower,
            upperLimits=self._upper,
            jointRanges=self._ranges,
            restPoses=self._rest,
            residualThreshold=1e-5,
            maxNumIterations=1000,
            physicsClientId=c,
        )

        # Extract arm angles from full-DOF IK solution
        arm_angles = np.zeros(len(PB_ARM_JOINTS), dtype=np.float32)
        for dof_i, arm_i in self._dof_to_arm.items():
            arm_angles[arm_i] = ik[dof_i]

        # Verify via FK
        for k, ji in enumerate(PB_ARM_JOINTS):
            p.resetJointState(r, ji, arm_angles[k], physicsClientId=c)
        ee_state = p.getLinkState(r, PB_EE_LINK, physicsClientId=c)
        ee_pos   = np.array(ee_state[4], dtype=float)
        error    = float(np.linalg.norm(ee_pos - target_hab))

        return (arm_angles if error < IK_ERROR_THRESH else None), error

    def close(self):
        self._p.disconnect(self._client)


# ── Camera helpers ────────────────────────────────────────────────────────────

def _look_at_rotation(eye: np.ndarray, target: np.ndarray):
    import quaternion as qt
    look_mat = mn.Matrix4.look_at(
        mn.Vector3(*eye.tolist()),
        mn.Vector3(*target.tolist()),
        mn.Vector3(0, 1, 0),
    )
    mq = mn.Quaternion.from_matrix(look_mat.rotation())
    return qt.quaternion(float(mq.scalar), float(mq.vector[0]),
                         float(mq.vector[1]), float(mq.vector[2]))


def _setup_camera(sim: habitat_sim.Simulator, asset_pos: np.ndarray,
                  robot_base: np.ndarray, cam_height: float,
                  lookat_y: float = None, view_angle_deg: float = 90.0):
    """
    Camera placed at view_angle_deg from the robot→asset axis in XZ.
    90° = pure side view; 45° = diagonal view.
    cam_height  — camera position Y
    lookat_y    — look-at point Y (asset bbox centre Y); defaults to cam_height
    """
    if lookat_y is None:
        lookat_y = cam_height

    dx = float(robot_base[0]) - float(asset_pos[0])
    dz = float(robot_base[2]) - float(asset_pos[2])
    length = float(np.sqrt(dx * dx + dz * dz)) or 1.0
    nx, nz = dx / length, dz / length   # unit vector robot→asset direction

    import math
    a = math.radians(view_angle_deg)
    # Rotate robot→asset unit vector by view_angle_deg around Y
    side_x = nx * math.cos(a) - nz * math.sin(a)
    side_z = nx * math.sin(a) + nz * math.cos(a)

    dist = 2.4
    eye    = np.array([float(asset_pos[0]) + side_x * dist,
                       cam_height,
                       float(asset_pos[2]) + side_z * dist])
    target = np.array([float(asset_pos[0]), lookat_y, float(asset_pos[2])])

    rot = _look_at_rotation(eye, target)
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = mn.Vector3(*eye.tolist())
    state.rotation = rot
    agent.set_state(state)


def _capture_frame(sim: habitat_sim.Simulator) -> Image.Image:
    return Image.fromarray(sim.get_sensor_observations()["color"][:, :, :3])


def _save_gif(frames: list, path: str) -> None:
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / GIF_FPS), loop=0)


def _capture_static(asset_config_path: str, asset_id: str, save_path: str,
                    grasp_point: np.ndarray = None, return_geometry: bool = False):
    """
    Build a minimal sim (scene_id=NONE, floor only, no walls) that exactly
    mirrors debug_grasp_scene.py: table + asset on table + robot at rest.
    Captures one side-view frame and saves as a single-frame GIF.
    """
    from habitat_object_benchmark.utils.sim_factory import load_fetch_robot, FETCH_URDF

    # ── Minimal sim ───────────────────────────────────────────────────────────
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "NONE"
    backend_cfg.enable_physics = True
    backend_cfg.create_renderer = True

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "color"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [720, 1280]
    sensor.position = mn.Vector3(0, 0, 0)
    sensor.hfov = mn.Deg(60.0)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]
    agent_cfg.height = 0.0

    s = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))

    # ── Robot first (reconfigure resets the sim) ──────────────────────────────
    robot = load_fetch_robot(s, FETCH_URDF)

    otm = s.get_object_template_manager()
    rom = s.get_rigid_object_manager()

    # ── Floor only (no walls) ─────────────────────────────────────────────────
    cube = otm.get_template_handles("cubeSolid")[0]
    tmpl = otm.get_template_by_handle(cube)
    tmpl.scale = mn.Vector3(10.0, 0.05, 10.0)
    otm.register_template(tmpl, "__floor__")
    floor_obj = rom.add_object_by_template_handle("__floor__")
    floor_obj.translation = mn.Vector3(0, -0.025, 0)
    floor_obj.motion_type = habitat_sim.physics.MotionType.STATIC
    # Remove flat shading so the floor receives scene lighting
    tmpl.shader_type = "pbr"
    tmpl.force_flat_shading = False
    otm.register_template(tmpl, "__floor__")

    # ── Table ─────────────────────────────────────────────────────────────────
    otm.load_configs(TABLE_CFG)
    table_handles = otm.get_template_handles("frl_apartment_table_01")
    if not table_handles:
        s.close()
        return
    table = rom.add_object_by_template_handle(table_handles[0])
    tbb = table.root_scene_node.cumulative_bb
    table.translation = mn.Vector3(0, -float(tbb.min[1]), 0)
    table.motion_type = habitat_sim.physics.MotionType.STATIC
    table_top_y = -float(tbb.min[1]) + float(tbb.max[1])

    # ── Asset on table ────────────────────────────────────────────────────────
    otm.load_configs(asset_config_path)
    handles = otm.get_template_handles(asset_id)
    if not handles:
        s.close()
        return
    obj = rom.add_object_by_template_handle(handles[0])
    abb = obj.root_scene_node.cumulative_bb
    obj.translation = mn.Vector3(0, table_top_y - float(abb.min[1]), 0)
    obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
    # ── Robot placement ───────────────────────────────────────────────────────
    static_dist = 0.5
    approach_angle = 0.0
    base_x = 0.0 + static_dist * float(np.sin(approach_angle))
    base_z = 0.0 + static_dist * float(np.cos(approach_angle))
    base_xz = np.array([base_x, base_z])

    robot_yaw = float(np.arctan2(0.0 - base_x, -(0.0 - base_z))) + np.pi / 2

    ik_solver = FetchIKSolver(FETCH_URDF)
    base_yaw = robot_yaw + ik_solver._arm_forward_angle
    hab_yaw  = robot_yaw

    robot_base = np.array([base_x, FLOOR_Y, base_z])
    robot.base_pos = mn.Vector3(*robot_base.tolist())
    robot.base_rot = robot_yaw
    robot.arm_joint_pos = ARM_INIT.copy()
    robot.gripper_joint_pos = np.array([GRIPPER_OPEN, GRIPPER_OPEN])
    robot.update()

    # ── Robot max Y for camera height ─────────────────────────────────────────
    ao = robot.sim_obj
    robot_max_y = 0.0
    for link_id in range(-1, ao.num_links):
        node = ao.get_link_scene_node(link_id)
        bb   = node.cumulative_bb
        T    = node.absolute_transformation()
        pts  = [T.transform_point(mn.Vector3(x, y, z))
                for x in (bb.min[0], bb.max[0])
                for y in (bb.min[1], bb.max[1])
                for z in (bb.min[2], bb.max[2])]
        mn_pt = mn.Vector3(min(p[0] for p in pts), min(p[1] for p in pts), min(p[2] for p in pts))
        mx_pt = mn.Vector3(max(p[0] for p in pts), max(p[1] for p in pts), max(p[2] for p in pts))
        if (mx_pt - mn_pt).length() < 10.0:
            robot_max_y = max(robot_max_y, float(mx_pt[1]))

    # ── Let object settle under physics ──────────────────────────────────────
    SETTLE_PHYSICS = 60   # ~1 s at 60 Hz
    for _ in range(SETTLE_PHYSICS):
        s.step_physics(1.0 / 60.0)

    # ── Compute grasp target from actual settled world position ───────────────
    obj_t = np.array(obj.translation)
    abb_centre_local = np.array([(float(abb.min[i]) + float(abb.max[i])) / 2.0
                                  for i in range(3)])
    obj_centre_world = obj_t + abb_centre_local
    asset_centre_y = float(obj_centre_world[1])

    wp_grasp = grasp_point if grasp_point is not None else obj_centre_world.copy()
    wp_lift  = np.array([wp_grasp[0], wp_grasp[1] + 0.15, wp_grasp[2]])

    # ── Camera: diagonal side view looking at settled object ──────────────────
    _setup_camera(s,
                  asset_pos=obj_centre_world,
                  robot_base=robot_base,
                  cam_height=robot_max_y,
                  lookat_y=asset_centre_y,
                  view_angle_deg=45.0)
    print(f"[capture_static] table_top={table_top_y:.3f}  obj_centre={obj_centre_world.round(3)}  obj_t={obj_t.round(3)}", flush=True)

    # ── Solve IK against actual settled position ──────────────────────────────
    # Offset between the IK EE link (palm plate, link 17) and the actual
    # finger contact point in the approach direction.
    EE_CONTACT_OFFSET = np.array([0.0, 0.0, 0.0])

    def _solve_relaxed(target, torso_override=None):
        """Solve IK searching over torso heights; returns best-converging solution.
        target is the semantic contact point; EE_CONTACT_OFFSET is added so the
        palm (EE link) lands above it by the finger-to-palm delta."""
        ik_target = target + EE_CONTACT_OFFSET
        search    = [torso_override] if torso_override is not None else TORSO_SEARCH
        best_angles, best_err, best_torso = None, float("inf"), TORSO_HEIGHT
        for th in search:
            angles, err = ik_solver.solve(base_xz, base_yaw, ik_target, None,
                                          torso_height=float(th))
            if angles is not None:
                return angles, float(th)
            if err < best_err:
                best_angles, best_err, best_torso = angles, err, float(th)
        print(f"[capture_static] relaxed IK contact={target.round(3)} ik_target={ik_target.round(3)} best_err={best_err:.3f} torso={best_torso:.2f}", flush=True)
        import pybullet as _p
        c, r = ik_solver._client, ik_solver._robot
        pb_base = (float(base_xz[0]), 0.0, float(base_xz[1]))
        pb_orn  = _p.getQuaternionFromEuler([0, base_yaw - ik_solver._arm_forward_angle, 0])
        _p.resetBasePositionAndOrientation(r, pb_base, pb_orn, physicsClientId=c)
        _p.resetJointState(r, PB_TORSO_JOINT, best_torso, physicsClientId=c)
        ik = _p.calculateInverseKinematics(
            r, PB_EE_LINK, ik_target.tolist(),
            lowerLimits=ik_solver._lower, upperLimits=ik_solver._upper,
            jointRanges=ik_solver._ranges, restPoses=ik_solver._rest,
            residualThreshold=1e-5, maxNumIterations=1000, physicsClientId=c,
        )
        raw = np.zeros(len(PB_ARM_JOINTS), dtype=np.float32)
        for dof_i, arm_i in ik_solver._dof_to_arm.items():
            raw[arm_i] = ik[dof_i]
        return raw, best_torso

    wp_pregrasp          = np.array([wp_grasp[0], wp_grasp[1] + 0.20, wp_grasp[2]])
    angles_pre,  _       = _solve_relaxed(wp_pregrasp)
    angles_grasp, best_torso = _solve_relaxed(wp_grasp)

    # Pre-compute lift angles while solver is still open.
    # Strategy: raise torso as much as possible; use arm IK for remainder.
    torso_raise  = min(LIFT_HEIGHT, TORSO_MAX - best_torso)
    arm_lift_rem = LIFT_HEIGHT - torso_raise
    new_torso    = best_torso + torso_raise
    if arm_lift_rem > 0.01:
        angles_lifted_pre, _ = _solve_relaxed(
            wp_grasp + np.array([0.0, arm_lift_rem, 0.0]),
            torso_override=new_torso)
    else:
        angles_lifted_pre = angles_grasp.copy()

    ik_solver.close()
    print(f"[capture_static] wp_grasp={wp_grasp.round(3)}  torso={best_torso:.3f}", flush=True)

    # Monkey-patch robot.update() so our torso height overrides the hardcoded
    # fix_back_val=0.15 that FetchRobot.update() always resets to.
    import types
    _orig_update = robot.__class__.update
    _desired_torso = best_torso

    def _update_with_torso(self):
        _orig_update(self)
        self._set_joint_pos(self.back_joint_id, _desired_torso)
        self._set_motor_pos(self.back_joint_id, _desired_torso)

    robot.update = types.MethodType(_update_with_torso, robot)
    robot.update()
    actual_torso = ao.joint_positions[robot.joint_pos_indices[robot.back_joint_id]]
    print(f"[capture_static] torso set to {best_torso:.4f}, actual={actual_torso:.4f}", flush=True)

    SIM_STEPS     = 80
    CAPTURE_EVERY = 4
    frames        = []

    def _capture():
        frames.append(_capture_frame(s))

    def _interp_arm(a, b, n=SIM_STEPS):
        for i in range(1, n + 1):
            t = i / n
            robot.arm_joint_pos = a + t * (b - a)
            robot.update()
            s.step_physics(1.0 / 60.0)
            if i % CAPTURE_EVERY == 0:
                _capture()

    def _interp_gripper(start, end, n=SIM_STEPS):
        for i in range(1, n + 1):
            t = i / n
            g = start + t * (end - start)
            robot.gripper_joint_pos = np.array([g, g])
            robot.update()
            s.step_physics(1.0 / 60.0)
            if i % CAPTURE_EVERY == 0:
                _capture()

    def _hold(n):
        for i in range(n):
            robot.update()
            s.step_physics(1.0 / 60.0)
            if i % CAPTURE_EVERY == 0:
                _capture()

    s.step_physics(1.0 / 60.0)
    _capture()

    torso_dof = robot.joint_pos_indices[robot.back_joint_id]
    print(f"[capture_static] torso before phase1: {ao.joint_positions[torso_dof]:.4f}", flush=True)
    # Keep object STATIC during arm approach so it isn't knocked away
    obj.motion_type = habitat_sim.physics.MotionType.STATIC
    # Phase 1: approach pre-grasp (object still STATIC, no collision risk)
    _interp_arm(ARM_INIT, angles_pre)
    # Phase 2: descend to grasp pose (object still STATIC)
    _interp_arm(angles_pre, angles_grasp)
    ee_pos_pre  = np.array(robot.ee_transform(0).translation)
    obj_pos_pre = np.array(obj.translation)
    dist_pre    = float(np.linalg.norm(ee_pos_pre - obj_pos_pre))
    print(f"[capture_static] pre-close: EE={ee_pos_pre.round(3)}  obj={obj_pos_pre.round(3)}  dist={dist_pre:.4f}", flush=True)

    # Let object settle on table before the arm arrives
    _hold(10)
    # Switch to DYNAMIC so physics contact drives the grasp.
    # Step briefly to bleed off any accumulated penetration before close.
    obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
    _hold(10)

    # Phase 4: force-limited gripper close.
    from habitat_sim.physics import JointMotorSettings

    GRIPPER_MAX_IMPULSE = 0.3

    for jidx in robot.params.gripper_joints:
        motor_id, _ = robot.joint_motors[jidx]
        jms = JointMotorSettings()
        jms.position_target = GRIPPER_CLOSED
        jms.position_gain   = robot.params.arm_mtr_pos_gain
        jms.velocity_gain   = robot.params.arm_mtr_vel_gain
        jms.max_impulse     = GRIPPER_MAX_IMPULSE
        ao.update_joint_motor(motor_id, jms)

    for _ci in range(60):
        s.step_physics(1.0 / 60.0)
        if _ci % CAPTURE_EVERY == 0:
            _capture()

    # Sync internal gripper state so robot.update() keeps fingers closed.
    robot.gripper_joint_pos = np.array([GRIPPER_CLOSED, GRIPPER_CLOSED])
    # Phase 5: hold briefly
    _hold(HOLD_STEPS)

    # Phase 6: motor-driven lift — torso + arm move together to carry the object.
    # Boost gripper to max clamping force.
    for jidx in robot.params.gripper_joints:
        motor_id, _ = robot.joint_motors[jidx]
        jms = JointMotorSettings()
        jms.position_target = GRIPPER_CLOSED
        jms.position_gain   = robot.params.arm_mtr_pos_gain
        jms.velocity_gain   = robot.params.arm_mtr_vel_gain
        jms.max_impulse     = 500.0
        ao.update_joint_motor(motor_id, jms)

    obj_y_before_lift = float(obj.translation[1])
    print(f"[capture_static] lift: best_torso={best_torso:.3f} torso_raise={torso_raise:.3f} new_torso={new_torso:.3f} arm_lift_rem={arm_lift_rem:.3f}", flush=True)

    # Drive arm joints to lifted target via motors (creates real physics forces).
    for i, jidx in enumerate(robot.params.arm_joints):
        motor_id, _ = robot.joint_motors[jidx]
        jms = JointMotorSettings()
        jms.position_target = float(angles_lifted_pre[i])
        jms.position_gain   = robot.params.arm_mtr_pos_gain
        jms.velocity_gain   = robot.params.arm_mtr_vel_gain
        jms.max_impulse     = 500.0
        ao.update_joint_motor(motor_id, jms)

    # Drive torso to new_torso via motor.
    torso_motor_id, _ = robot.joint_motors[robot.back_joint_id]
    jms_t = JointMotorSettings()
    jms_t.position_target = new_torso
    jms_t.position_gain   = robot.params.arm_mtr_pos_gain
    jms_t.velocity_gain   = robot.params.arm_mtr_vel_gain
    jms_t.max_impulse     = 500.0
    ao.update_joint_motor(torso_motor_id, jms_t)

    # Step physics WITHOUT robot.update() so motors drive joints with real forces.
    for i in range(LIFT_STEPS + 20):
        s.step_physics(1.0 / 60.0)
        if i % CAPTURE_EVERY == 0:
            _capture()

    obj_y_after_lift = float(obj.translation[1])
    lifted_enough    = (obj_y_after_lift - obj_y_before_lift) > LIFT_THRESHOLD
    print(f"[capture_static] obj_y before={obj_y_before_lift:.3f} after_lift={obj_y_after_lift:.3f}  "
          f"rise={obj_y_after_lift - obj_y_before_lift:.3f}  {'OK' if lifted_enough else 'NOT LIFTED'}", flush=True)

    # Phase 7: open gripper and let object fall
    for jidx in robot.params.gripper_joints:
        motor_id, _ = robot.joint_motors[jidx]
        jms = JointMotorSettings()
        jms.position_target = GRIPPER_OPEN
        jms.position_gain   = robot.params.arm_mtr_pos_gain
        jms.velocity_gain   = robot.params.arm_mtr_vel_gain
        jms.max_impulse     = 1000.0
        ao.update_joint_motor(motor_id, jms)
    _hold(RELEASE_STEPS)
    obj_y_after_open = float(obj.translation[1])
    fell = (obj_y_after_lift - obj_y_after_open) > FALL_THRESHOLD
    success = lifted_enough and fell
    print(f"[capture_static] obj_y after_open={obj_y_after_open:.3f}  "
          f"drop={obj_y_after_lift - obj_y_after_open:.3f}  "
          f"{'GRASP SUCCESS' if success else 'GRASP FAILED'}", flush=True)

    # Restore original update before closing
    del robot.update
    s.close()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    _save_gif(frames, save_path)

    if return_geometry:
        bb_world_min = obj_t + np.array([float(abb.min[0]), float(abb.min[1]), float(abb.min[2])])
        bb_world_max = obj_t + np.array([float(abb.max[0]), float(abb.max[1]), float(abb.max[2])])
        return {
            "bb_min": bb_world_min, "bb_max": bb_world_max,
            "wp_grasp": wp_grasp, "wp_pregrasp": wp_pregrasp, "wp_lift": wp_lift,
            "robot_base": robot_base, "table_top_y": table_top_y,
        }


# ── Table-top snap (robot moved aside) ───────────────────────────────────────

def _get_snap_pos(sim, robot, use_handle, table_top_y: float,
                  support_obj_ids: list = None):
    """Snap object onto the table surface at table_top_y."""
    rom = sim.get_rigid_object_manager()
    obj = rom.add_object_by_template_handle(use_handle)
    if obj is None or not obj.is_alive:
        return None, None
    bb        = obj.root_scene_node.cumulative_bb
    aabb_size = np.array([bb.size_x(), bb.size_y(), bb.size_z()])
    # Start 0.3 m above table
    obj.translation = mn.Vector3(0.0,
                                 table_top_y + 0.3 - float(bb.min[1]),
                                 0.0)
    obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC

    saved = mn.Vector3(robot.base_pos)
    robot.base_pos = mn.Vector3(50.0, FLOOR_Y, 0.0)
    robot.update()

    snapped  = snap_down(sim, obj,
                         support_obj_ids=support_obj_ids,
                         max_collision_depth=0.5)
    snap_pos = np.array(obj.translation) if snapped else None
    rom.remove_object_by_id(obj.object_id)

    robot.base_pos = saved
    robot.update()
    return snap_pos, aabb_size


# ── Single trial ─────────────────────────────────────────────────────────────

GRIPPER_MAX_IMPULSE = 0.3   # N·s — force-limited close; shared with _capture_static


def _run_trial(sim, robot, ik_solver, use_handle,
               candidate, snap_pos, capture: bool = False):
    """
    Full gripper grasp trial using antipodal candidate + physics-based success.

    candidate = (cand, M, x_grip, z_grip) where:
      cand   = (score, p1, p2, width) from _sample_candidates
      M      = grasp midpoint in mesh-local coords
      x_grip = squeeze axis (unit vector)
      z_grip = approach axis (unit vector, roughly upward)

    1. Spawn object at snap_pos, let it settle (DYNAMIC).
    2. Compute world_M = obj_t + M; derive approach angle from z_grip.
    3. Try approach from both sides of squeeze axis; pick best IK.
    4. Arm moves REST → pre-grasp → grasp pose.
    5. Force-limited gripper close.
    6. Hold under gravity; motor-driven lift.
    7. Success = lifted_enough AND fell after release.
    """
    import types
    from habitat_sim.physics import JointMotorSettings

    cand, M, x_grip, z_grip = candidate

    rom = sim.get_rigid_object_manager()
    ao  = robot.sim_obj
    obj = None
    torso_patched = False

    try:
        # ── Rotate object around Y so x_grip faces the robot ─────────────────
        # Robot is fixed at approach_angle=0 (same as _capture_static).
        # We rotate the object so its squeeze axis aligns with the gripper's
        # squeeze direction at approach_angle=0, which is (-1, 0, 0).
        # theta_y = atan2(x_grip[2], x_grip[0]) rotates x_grip → (1,0,0).
        theta_y = math.atan2(float(x_grip[2]), float(x_grip[0]))
        rot_y   = mn.Quaternion.rotation(mn.Rad(theta_y), mn.Vector3(0, 1, 0))

        # Rotation matrix for M offset computation
        cy, sy = math.cos(theta_y), math.sin(theta_y)
        R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)

        # ── Spawn object: settle with rot_y applied, then compute world_M ───────
        obj = rom.add_object_by_template_handle(use_handle)
        if obj is None or not obj.is_alive:
            return {"success": False, "grasp_width_m": None, "ik_error": None,
                    "frames": [], "obj_y_rise": 0.0, "ee_trajectory": [], "grasp_close_idx": None}
        obj.translation = mn.Vector3(float(snap_pos[0]),
                                     float(snap_pos[1]),
                                     float(snap_pos[2]))
        obj.rotation    = rot_y
        obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        # Run ~60 settle steps (1 s) so the object reaches its final resting position
        # before we compute IK targets from it.
        for _ in range(60):
            sim.step_physics(1.0 / 60.0)

        settled_t = np.array(obj.translation)
        # M is in mesh-local coords; apply the same rotation to get world offset
        world_M = settled_t + R_y @ np.array(M)

        # Freeze object while we set up the robot and solve IK
        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        obj.translation = mn.Vector3(float(settled_t[0]), float(settled_t[1]), float(settled_t[2]))
        obj.rotation    = rot_y

        wp_grasp    = world_M.copy()
        wp_pregrasp = wp_grasp + np.array([0.0, 0.20, 0.0])

        # Fixed robot position — same as _capture_static (approach_angle = 0.0).
        # orn_angle = 0.0 → _ee_target_orn squeezes along (-1,0,0) = x_grip after rotation.
        FIXED_ANGLE = 0.0
        base_x  = float(settled_t[0]) + APPROACH_DIST * math.sin(FIXED_ANGLE)
        base_z  = float(settled_t[2]) + APPROACH_DIST * math.cos(FIXED_ANGLE)
        base_xz = np.array([base_x, base_z])
        base_yaw = float(FIXED_ANGLE + math.pi)
        orn_angle = FIXED_ANGLE   # _ee_target_orn(0) → squeeze = (-1,0,0)

        EE_CONTACT_OFFSET = np.array([0.0, 0.0, 0.0])

        def _solve_relaxed(target, torso_override=None):
            ik_target = target + EE_CONTACT_OFFSET
            search    = [torso_override] if torso_override is not None else TORSO_SEARCH
            best_angles, best_err, best_torso = None, float("inf"), TORSO_HEIGHT
            for th in search:
                angles, err = ik_solver.solve(base_xz, base_yaw, ik_target,
                                              orn_angle, torso_height=float(th))
                if angles is not None:
                    return angles, float(th), err
                if err < best_err:
                    best_angles, best_err, best_torso = angles, err, float(th)
            import pybullet as _p
            c, r = ik_solver._client, ik_solver._robot
            _p.resetBasePositionAndOrientation(
                r, (float(base_xz[0]), 0.0, float(base_xz[1])),
                _p.getQuaternionFromEuler([0, base_yaw - ik_solver._arm_forward_angle, 0]),
                physicsClientId=c)
            _p.resetJointState(r, PB_TORSO_JOINT, best_torso, physicsClientId=c)
            ik = _p.calculateInverseKinematics(
                r, PB_EE_LINK, ik_target.tolist(),
                lowerLimits=ik_solver._lower, upperLimits=ik_solver._upper,
                jointRanges=ik_solver._ranges, restPoses=ik_solver._rest,
                residualThreshold=1e-5, maxNumIterations=1000, physicsClientId=c)
            raw = np.zeros(len(PB_ARM_JOINTS), dtype=np.float32)
            for dof_i, arm_i in ik_solver._dof_to_arm.items():
                raw[arm_i] = ik[dof_i]
            return raw, best_torso, best_err

        angles_pre,   best_torso, _      = _solve_relaxed(wp_pregrasp)
        angles_grasp, best_torso, ik_err = _solve_relaxed(wp_grasp)

        if angles_grasp is None:
            return {"success": False, "grasp_width_m": None,
                    "ik_error": round(float(ik_err), 4), "frames": [], "obj_y_rise": 0.0,
                    "ee_trajectory": [], "grasp_close_idx": None}

        # ── Place robot ───────────────────────────────────────────────────────
        hab_yaw = base_yaw - ik_solver._arm_forward_angle
        robot.base_pos = mn.Vector3(base_x, FLOOR_Y, base_z)
        robot.base_rot = hab_yaw
        robot.arm_joint_pos = ARM_INIT.copy()
        robot.gripper_joint_pos = np.array([GRIPPER_OPEN, GRIPPER_OPEN])
        robot.update()

        _orig_update   = robot.__class__.update
        _desired_torso = best_torso

        def _update_with_torso(self):
            _orig_update(self)
            self._set_joint_pos(self.back_joint_id, _desired_torso)
            self._set_motor_pos(self.back_joint_id, _desired_torso)

        robot.update  = types.MethodType(_update_with_torso, robot)
        torso_patched = True
        robot.update()

        frames        = []
        ee_trajectory = []

        if capture:
            _setup_camera(sim,
                          asset_pos=np.array([float(settled_t[0]), float(world_M[1]), float(settled_t[2])]),
                          robot_base=np.array([base_x, FLOOR_Y, base_z]),
                          cam_height=float(world_M[1]) + 1.2,
                          lookat_y=float(world_M[1]),
                          view_angle_deg=45.0)

        def _cap():
            if capture:
                frames.append(_capture_frame(sim))

        def _interp_arm(a, b, n=APPROACH_STEPS):
            for i in range(1, n + 1):
                robot.arm_joint_pos = a + (i / n) * (b - a)
                robot.update()
                sim.step_physics(1.0 / 60.0)
                ee_trajectory.append(list(robot.ee_transform(0).translation))
                if i % 4 == 0:
                    _cap()

        def _cartesian_descent(start_pos, end_pos, n=APPROACH_STEPS):
            """Straight-line Cartesian interpolation from start_pos to end_pos.
            Calls PyBullet IK directly at each waypoint (position-only, no
            orientation constraint) and always takes the best solution regardless
            of residual — the threshold-gate in ik_solver.solve() is what was
            causing the arm to freeze.  Explicitly seeds PyBullet joint state
            from the previous solution for reliable warm-starting."""
            import pybullet as _p
            c, r   = ik_solver._client, ik_solver._robot
            pb_base = (float(base_xz[0]), 0.0, float(base_xz[1]))
            pb_orn  = _p.getQuaternionFromEuler(
                [0, base_yaw - ik_solver._arm_forward_angle, 0])

            current = angles_pre.copy()
            # Seed PyBullet arm joints with angles_pre so first step warm-starts correctly.
            _p.resetBasePositionAndOrientation(r, pb_base, pb_orn, physicsClientId=c)
            _p.resetJointState(r, PB_TORSO_JOINT, best_torso, physicsClientId=c)
            for k, ji in enumerate(PB_ARM_JOINTS):
                _p.resetJointState(r, ji, float(current[k]), physicsClientId=c)

            for i in range(1, n + 1):
                t      = i / n
                target = np.asarray(start_pos) + t * (np.asarray(end_pos) - np.asarray(start_pos))

                _p.resetBasePositionAndOrientation(r, pb_base, pb_orn, physicsClientId=c)
                _p.resetJointState(r, PB_TORSO_JOINT, best_torso, physicsClientId=c)
                ik = _p.calculateInverseKinematics(
                    r, PB_EE_LINK, target.tolist(),
                    lowerLimits=ik_solver._lower, upperLimits=ik_solver._upper,
                    jointRanges=ik_solver._ranges, restPoses=ik_solver._rest,
                    residualThreshold=1e-5, maxNumIterations=200, physicsClientId=c)

                new_angles = np.zeros(len(PB_ARM_JOINTS), dtype=np.float32)
                for dof_i, arm_i in ik_solver._dof_to_arm.items():
                    new_angles[arm_i] = ik[dof_i]

                # Seed next call from this solution for warm-starting.
                for k, ji in enumerate(PB_ARM_JOINTS):
                    _p.resetJointState(r, ji, float(new_angles[k]), physicsClientId=c)

                current = new_angles
                robot.arm_joint_pos = current
                robot.update()
                sim.step_physics(1.0 / 60.0)
                ee_trajectory.append(list(robot.ee_transform(0).translation))
                if i % 4 == 0:
                    _cap()
            return current

        # Keep object KINEMATIC during arm approach so it doesn't swing from
        # incidental contact; switch to DYNAMIC only just before gripper close.
        # Phase 1 — joint-space approach to pre-grasp (gets the arm in the right region)
        _interp_arm(ARM_INIT, angles_pre)
        # Phase 2 — Cartesian descent: EE follows a straight line from wp_pregrasp to wp_grasp
        final_grasp_angles = _cartesian_descent(wp_pregrasp, wp_grasp)
        grasp_close_idx = len(ee_trajectory) - 1

        # Switch to DYNAMIC now so the gripper can interact with the object
        obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC

        # Slow, gentle close: low position_gain limits finger velocity per physics
        # step so collision detection can stop the fingers before they tunnel through.
        for jidx in robot.params.gripper_joints:
            motor_id, _ = robot.joint_motors[jidx]
            jms = JointMotorSettings()
            jms.position_target = GRIPPER_CLOSED
            jms.position_gain   = robot.params.arm_mtr_pos_gain * 0.1
            jms.velocity_gain   = robot.params.arm_mtr_vel_gain
            jms.max_impulse     = GRIPPER_MAX_IMPULSE
            ao.update_joint_motor(motor_id, jms)

        for i in range(180):
            sim.step_physics(1.0 / 240.0)  # sub-step for finer collision detection
            if i % 12 == 0:
                _cap()

        grasp_width = float(ao.joint_positions[robot.joint_pos_indices[robot.params.gripper_joints[0]]]) * 2.0

        # Sync internal gripper state so robot.update() keeps fingers closed.
        robot.gripper_joint_pos = np.array([GRIPPER_CLOSED, GRIPPER_CLOSED])
        # Brief hold to let grip stabilise
        for i in range(HOLD_STEPS):
            robot.arm_joint_pos = final_grasp_angles
            robot.update()
            sim.step_physics(1.0 / 60.0)
            if i % 4 == 0:
                _cap()

        # ── Motor-driven lift — torso + arm motors carry the object upward ───
        # Boost gripper to max clamping force.
        for jidx in robot.params.gripper_joints:
            motor_id, _ = robot.joint_motors[jidx]
            jms = JointMotorSettings()
            jms.position_target = GRIPPER_CLOSED
            jms.position_gain   = robot.params.arm_mtr_pos_gain
            jms.velocity_gain   = robot.params.arm_mtr_vel_gain
            jms.max_impulse     = 500.0
            ao.update_joint_motor(motor_id, jms)

        torso_raise  = min(LIFT_HEIGHT, TORSO_MAX - best_torso)
        arm_lift_rem = LIFT_HEIGHT - torso_raise
        new_torso    = best_torso + torso_raise
        if arm_lift_rem > 0.01:
            angles_lifted, _, _ = _solve_relaxed(
                wp_grasp + np.array([0.0, arm_lift_rem, 0.0]),
                torso_override=new_torso)
        else:
            angles_lifted = final_grasp_angles.copy()

        obj_y_before_lift = float(obj.translation[1])

        # Drive arm joints via motors (real physics forces, not kinematic).
        for i, jidx in enumerate(robot.params.arm_joints):
            motor_id, _ = robot.joint_motors[jidx]
            jms = JointMotorSettings()
            jms.position_target = float(angles_lifted[i])
            jms.position_gain   = robot.params.arm_mtr_pos_gain
            jms.velocity_gain   = robot.params.arm_mtr_vel_gain
            jms.max_impulse     = 500.0
            ao.update_joint_motor(motor_id, jms)

        torso_motor_id, _ = robot.joint_motors[robot.back_joint_id]
        jms_t = JointMotorSettings()
        jms_t.position_target = new_torso
        jms_t.position_gain   = robot.params.arm_mtr_pos_gain
        jms_t.velocity_gain   = robot.params.arm_mtr_vel_gain
        jms_t.max_impulse     = 500.0
        ao.update_joint_motor(torso_motor_id, jms_t)

        # Step physics WITHOUT robot.update() so motors drive with real forces.
        for i in range(LIFT_STEPS + 20):
            sim.step_physics(1.0 / 60.0)
            if i % 4 == 0:
                _cap()

        obj_y_after_lift = float(obj.translation[1])
        lifted_enough    = (obj_y_after_lift - obj_y_before_lift) > LIFT_THRESHOLD

        # ── Open gripper and check that the object falls ─────────────────────
        for jidx in robot.params.gripper_joints:
            motor_id, _ = robot.joint_motors[jidx]
            jms = JointMotorSettings()
            jms.position_target = GRIPPER_OPEN
            jms.position_gain   = robot.params.arm_mtr_pos_gain
            jms.velocity_gain   = robot.params.arm_mtr_vel_gain
            jms.max_impulse     = 1000.0
            ao.update_joint_motor(motor_id, jms)
        for i in range(RELEASE_STEPS):
            sim.step_physics(1.0 / 60.0)
            if i % 4 == 0:
                _cap()

        obj_y_after_open = float(obj.translation[1])
        fell    = (obj_y_after_lift - obj_y_after_open) > FALL_THRESHOLD
        success = lifted_enough and fell

        obj_y_rise = max(0.0, obj_y_after_lift - float(snap_pos[1]))
        return {"success": success, "grasp_width_m": round(grasp_width, 4),
                "ik_error": round(float(ik_err), 4), "frames": frames,
                "obj_y_rise": round(obj_y_rise, 4),
                "ee_trajectory": ee_trajectory, "grasp_close_idx": grasp_close_idx}

    finally:
        if torso_patched:
            try:
                del robot.update
            except AttributeError:
                pass
        if obj is not None and obj.is_alive:
            rom.remove_object_by_id(obj.object_id)


# ── Public API ────────────────────────────────────────────────────────────────

def run(sim, robot, ik_solver, asset_handle: str,
        collision_mode: str = "convex_hull",
        save_dir: str = None, asset_id: str = None,
        save_all_trials: bool = False) -> dict:
    result = {
        "collision_mode":        collision_mode,
        "grasp_success_rate":    None,
        "grasp_successes":       None,
        "grasp_trials":          GRASP_CANDIDATES,
        "mean_grasp_width_m":    None,
        "error":                 None,
    }

    otm = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()
    table_obj_id = None
    try:
        # ── Add table once per asset (static, removed afterwards) ────────────
        otm.load_configs(TABLE_CFG)
        table_handles = otm.get_template_handles("frl_apartment_table_01")
        if table_handles:
            table = rom.add_object_by_template_handle(table_handles[0])
            tbb = table.root_scene_node.cumulative_bb
            table.translation = mn.Vector3(0, -float(tbb.min[1]), 0)
            table.motion_type = habitat_sim.physics.MotionType.STATIC
            table_obj_id = table.object_id
            table_top_y = -float(tbb.min[1]) + float(tbb.max[1])
        else:
            table_top_y = FLOOR_Y

        template = otm.get_template_by_handle(asset_handle)

        config_dir    = os.path.dirname(asset_handle)
        config_parent = os.path.dirname(config_dir)

        if collision_mode == "convex_hull":
            otm.register_template(template, asset_handle + "__g_convex_hull")
            use_handle = asset_handle + "__g_convex_hull"
            # habitat-sim strips the leading ../ from JSON paths when storing the handle,
            # so resolve relative to config_parent rather than config_dir.
            collision_rel  = template.collision_asset_handle
            if collision_rel and not os.path.isabs(collision_rel):
                collision_mesh_path = os.path.normpath(os.path.join(config_parent, collision_rel))
            else:
                collision_mesh_path = collision_rel or ""
        elif collision_mode == "vhacd":
            render_rel = template.render_asset_handle
            if render_rel and not os.path.isabs(render_rel):
                render_abs = os.path.normpath(os.path.join(config_parent, render_rel))
            else:
                render_abs = render_rel or ""
            vhacd = render_abs.replace(".glb", ".vhacd.glb")
            if not os.path.exists(vhacd):
                result["error"] = f"vhacd not found: {vhacd}"
                return result
            template.collision_asset_handle = vhacd
            template.join_collision_meshes  = False
            otm.register_template(template, asset_handle + "__g_vhacd")
            use_handle = asset_handle + "__g_vhacd"
            collision_mesh_path = vhacd
        else:
            raise ValueError(f"Unknown collision_mode: {collision_mode}")

        snap_pos, _ = _get_snap_pos(sim, robot, use_handle, table_top_y,
                                    support_obj_ids=[table_obj_id] if table_obj_id is not None else None)
        if snap_pos is None:
            result["error"] = "snap_down failed"
            return result

        # ── Sample antipodal candidates from the collision mesh ───────────────
        mesh = None
        mesh_bb_center = None
        candidates = []
        if collision_mesh_path and os.path.exists(collision_mesh_path):
            try:
                loaded = _load_glb_mesh(collision_mesh_path)
                # Apply habitat-sim object scale (template.scale) if non-unity
                tmpl_scale = template.scale
                scale_xyz  = np.array([tmpl_scale[0], tmpl_scale[1], tmpl_scale[2]], dtype=float)
                if not np.allclose(scale_xyz, 1.0):
                    loaded = loaded.copy()
                    loaded.vertices *= scale_xyz
                mesh = loaded
                mesh_bb_center = (np.array(mesh.bounds[0]) + np.array(mesh.bounds[1])) / 2.0
                candidates, _, _ = _sample_candidates(mesh, n_samples=500)
            except Exception:
                pass

        if mesh_bb_center is None:
            mesh_bb_center = np.zeros(3)

        # Fallback: no antipodal candidates → compass-angle approach
        if not candidates:
            # Get AABB centre in object-local coords
            _tmp = rom.add_object_by_template_handle(use_handle)
            if _tmp is not None and _tmp.is_alive:
                _tmp.translation = mn.Vector3(float(snap_pos[0]), float(snap_pos[1]), float(snap_pos[2]))
                abb = _tmp.root_scene_node.cumulative_bb
                abb_c_local = np.array([(float(abb.min[i]) + float(abb.max[i])) / 2.0 for i in range(3)])
                rom.remove_object_by_id(_tmp.object_id)
            else:
                abb_c_local = np.zeros(3)
            M_fallback    = abb_c_local  # mesh-local AABB centre
            x_fallback    = np.array([1.0, 0.0, 0.0])
            z_fallback    = np.array([0.0, 1.0, 0.0])
            y_fallback    = np.array([0.0, 0.0, 1.0])
            fallback_cand = (1.0, M_fallback, M_fallback, 0.04)
            cand_list = [(fallback_cand, M_fallback, x_fallback, y_fallback, z_fallback)]
            # Generate K compass-angle variants
            for ang in np.linspace(0, 2 * np.pi, GRASP_CANDIDATES, endpoint=False):
                z_ang = np.array([float(np.sin(ang)), 0.0, float(np.cos(ang))])
                cand_list.append((fallback_cand, M_fallback, x_fallback, y_fallback, z_ang))
            cand_list = cand_list[:GRASP_CANDIDATES]
        else:
            cand_list = []
            for c in candidates[:GRASP_CANDIDATES]:
                _, p1, p2, _ = c
                M_a, x_a, y_a, z_a = _grasp_frame(p1, p2)
                # Shift from trimesh origin (mesh bottom) to habitat-sim origin (CoM)
                M_hab = np.array(M_a) - mesh_bb_center
                cand_list.append((c, M_hab, x_a, y_a, z_a))

        capturing = save_dir is not None and asset_id is not None
        all_results = []
        successes, widths = 0, []
        n_trials = len(cand_list)

        for i, (cand, M_a, x_a, y_a, z_a) in enumerate(cand_list):
            r = _run_trial(sim, robot, ik_solver, use_handle,
                           (cand, M_a, x_a, z_a), snap_pos,
                           capture=capturing)
            all_results.append(r)
            if r["success"]:
                successes += 1
            if r.get("grasp_width_m") is not None:
                widths.append(r["grasp_width_m"])

            # Per-trial HTML debug visualization
            if save_all_trials and mesh is not None and save_dir is not None and asset_id is not None:
                theta_y = math.atan2(float(x_a[2]), float(x_a[0]))
                cy, sy  = math.cos(theta_y), math.sin(theta_y)
                R_y     = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
                # M_a is already shifted by -mesh_bb_center (hab-sim CoM origin)
                world_M_i = snap_pos + R_y @ np.asarray(M_a)
                _, p1_i, p2_i, _ = cand
                wp_pre_i  = world_M_i + np.array([0.0, 0.20, 0.0])
                html_dir  = os.path.join(save_dir, "trials_debug", asset_id)
                os.makedirs(html_dir, exist_ok=True)
                _save_grasp_html(
                    path=os.path.join(html_dir, f"trial_{i:02d}.html"),
                    mesh=mesh,
                    p1_local=p1_i, p2_local=p2_i, M_local=M_a,
                    x_grip_local=x_a, y_grip_local=y_a, z_grip_local=z_a,
                    snap_pos=snap_pos, R_y=R_y, table_top_y=table_top_y,
                    world_M=world_M_i, wp_pregrasp=wp_pre_i,
                    ee_trajectory=r.get("ee_trajectory", []),
                    grasp_close_idx=r.get("grasp_close_idx"),
                    mesh_bb_center=mesh_bb_center,
                )

        if capturing and all_results:
            best = _pick_best_trial(all_results)
            if best.get("frames"):
                os.makedirs(save_dir, exist_ok=True)
                _save_gif(best["frames"], os.path.join(save_dir, f"{asset_id}.gif"))
            if save_all_trials:
                debug_dir = os.path.join(save_dir, "trials_debug", asset_id)
                os.makedirs(debug_dir, exist_ok=True)
                for i, r in enumerate(all_results):
                    if r.get("frames"):
                        _save_gif(r["frames"],
                                  os.path.join(debug_dir, f"trial_{i:02d}.gif"))

        result["grasp_successes"]    = successes
        result["grasp_trials"]       = n_trials
        result["grasp_success_rate"] = round(successes / n_trials, 4) if n_trials else 0.0
        result["mean_grasp_width_m"] = round(float(np.mean(widths)), 4) if widths else None

    except Exception as e:
        import traceback
        traceback.print_exc()
        result["error"] = f"{type(e).__name__}: {e}"

    finally:
        if table_obj_id is not None:
            try:
                rom.remove_object_by_id(table_obj_id)
            except Exception:
                pass

    return result
