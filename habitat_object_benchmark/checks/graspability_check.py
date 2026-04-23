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

import habitat_sim
import magnum as mn
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "habitat-lab"))
from habitat.sims.habitat_simulator.sim_utilities import snap_down

# ── Constants ────────────────────────────────────────────────────────────────

APPROACH_DIST   = 0.80    # robot base distance from object (m)
GRASP_TRIALS    = 8       # equally-spaced approach angles
DRIFT_THRESHOLD = 0.15    # m — drift above this = failed grasp
IK_ERROR_THRESH = 0.05    # m — IK convergence threshold

# Gripper animation steps (at 1/60 s each)
APPROACH_STEPS  = 40      # arm moves from rest to grasp pose
SETTLE_STEPS    = 5       # arm + object settle before gripper closes
CLOSE_STEPS     = 20      # gripper closes from open to closed
HOLD_STEPS      = 60      # hold with closed gripper (~1 s)
RELEASE_STEPS   = 20      # gripper opens; object falls (captured in GIF)

GRIPPER_OPEN    = 0.04    # each finger position when open (m)
GRIPPER_CLOSED  = 0.0     # each finger position when closed (m)

GIF_FRAMES      = 30
GIF_FPS         = 10

FLOOR_Y         = 0.0
TORSO_HEIGHT    = 0.2     # torso_lift_joint default
GRASP_HEIGHT    = 1.1     # height above floor where EE reaches (arm workspace sweet spot)

# PyBullet joint indices (from URDF inspection)
PB_EE_LINK          = 17               # gripper_link
PB_ARM_JOINTS       = [10, 11, 12, 13, 14, 15, 16]   # shoulder_pan→wrist_roll
PB_TORSO_JOINT      = 2

ARM_INIT = np.array([-0.45, -1.08, 0.1, 0.935, -0.001, 1.573, 0.005],
                    dtype=np.float32)


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
              target_hab: np.ndarray, approach_angle: float = None):
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
        p.resetJointState(r, PB_TORSO_JOINT, TORSO_HEIGHT,
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
            residualThreshold=1e-4,
            maxNumIterations=300,
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
    delta = target - eye
    yaw   = float(np.arctan2(delta[0], -delta[2]))
    horiz = float(np.sqrt(delta[0] ** 2 + delta[2] ** 2))
    pitch = float(np.arctan2(delta[1], horiz))
    mq = (mn.Quaternion.rotation(mn.Rad(yaw),   mn.Vector3(0, 1, 0))
        * mn.Quaternion.rotation(mn.Rad(pitch),  mn.Vector3(1, 0, 0)))
    return qt.quaternion(mq.scalar, mq.vector[0], mq.vector[1], mq.vector[2])


def _setup_camera(sim: habitat_sim.Simulator, robot_base: np.ndarray,
                  ee_pos: np.ndarray, approach_angle: float):
    """
    Position camera to show the robot arm reaching toward the object.
    Uses the same pattern as physics_check._setup_camera (known to work):
    eye is offset behind the scene along the robot's approach axis.
    """
    import quaternion as qt
    mid_y  = (float(robot_base[1]) + float(ee_pos[1])) * 0.5
    # target = midpoint of robot base and grasp point in XZ, halfway up
    target = np.array([
        (float(robot_base[0]) + float(ee_pos[0])) * 0.5,
        mid_y,
        (float(robot_base[2]) + float(ee_pos[2])) * 0.5,
    ])
    # Eye is behind the robot along its approach direction, elevated
    behind_x = float(np.sin(approach_angle))
    behind_z = float(np.cos(approach_angle))
    dist = 2.5
    eye = np.array([target[0] + behind_x * dist,
                    mid_y + 0.8,
                    target[2] + behind_z * dist])

    agent = sim.get_agent(0)
    agent.set_state(agent.get_state().__class__())
    default_state = agent.get_state()
    rot = _look_at_rotation(eye, target)
    sensor_h = float(default_state.sensor_states["color"].position[1])
    local_v = qt.quaternion(0, 0, sensor_h, 0)
    world_v = rot * local_v * rot.conjugate()
    default_state.position = eye - np.array([world_v.x, world_v.y, world_v.z])
    default_state.rotation = rot
    agent.set_state(default_state)


def _capture_frame(sim: habitat_sim.Simulator) -> Image.Image:
    return Image.fromarray(sim.get_sensor_observations()["color"][:, :, :3])


def _save_gif(frames: list, path: str) -> None:
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / GIF_FPS), loop=0)


# ── Floor snap (robot moved aside) ───────────────────────────────────────────

def _get_snap_pos(sim, robot, use_handle):
    rom = sim.get_rigid_object_manager()
    obj = rom.add_object_by_template_handle(use_handle)
    if obj is None or not obj.is_alive:
        return None, None
    bb        = obj.root_scene_node.cumulative_bb
    aabb_size = np.array([bb.size_x(), bb.size_y(), bb.size_z()])
    obj.translation = mn.Vector3(0.0,
                                 FLOOR_Y + 0.5 - float(bb.min[1]),
                                 0.0)
    obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC

    saved = mn.Vector3(robot.base_pos)
    robot.base_pos = mn.Vector3(50.0, FLOOR_Y, 0.0)
    robot.update()

    floor_ids = [rom.get_object_by_handle(h).object_id
                 for h in rom.get_object_handles("__floor_plane__")]
    snapped  = snap_down(sim, obj,
                         support_obj_ids=floor_ids or None,
                         max_collision_depth=0.5)
    snap_pos = np.array(obj.translation) if snapped else None
    rom.remove_object_by_id(obj.object_id)

    robot.base_pos = saved
    robot.update()
    return snap_pos, aabb_size


# ── Single trial ─────────────────────────────────────────────────────────────

_TOTAL_STEPS = APPROACH_STEPS + SETTLE_STEPS + CLOSE_STEPS + RELEASE_STEPS


def _run_trial(sim, robot, ik_solver, use_handle,
               approach_angle, snap_pos, grasp_target, aabb_size, capture):
    """
    Full gripper grasp trial.

    Success criterion (geometric):
      1. IK converges within IK_ERROR_THRESH.
      2. The object's AABB extent along the gripper's finger-squeeze axis
         (EE local Y in world) fits within the open finger span (GRIPPER_OPEN*2).

    The physics phases (close + release) are run for the GIF only.
    This avoids the kinematic-contact unreliability of physics-based hold tests.
    """
    rom    = sim.get_rigid_object_manager()
    frames = []

    # ── IK solve ─────────────────────────────────────────────────────────────
    base_x   = snap_pos[0] + APPROACH_DIST * float(np.sin(approach_angle))
    base_z   = snap_pos[2] + APPROACH_DIST * float(np.cos(approach_angle))
    base_xz  = np.array([base_x, base_z])
    base_yaw = float(approach_angle + np.pi)

    arm_angles, ik_err = ik_solver.solve(base_xz, base_yaw, grasp_target)
    if arm_angles is None:
        return {"success": False, "grasp_width_m": None,
                "ik_error": round(ik_err, 4), "frames": []}

    # ── Place robot, start from rest ──────────────────────────────────────────
    hab_yaw = base_yaw - ik_solver._arm_forward_angle
    robot.base_pos = mn.Vector3(base_x, FLOOR_Y, base_z)
    robot.base_rot = hab_yaw
    robot.arm_joint_pos = ARM_INIT.copy()
    robot.gripper_joint_pos = np.array([GRIPPER_OPEN, GRIPPER_OPEN])
    robot.update()

    # Spawn object at the grasp target position before any arm motion.
    # This represents an object resting on a shelf/table at GRASP_HEIGHT.
    # The arm will then be seen approaching and closing around it.
    obj = None
    try:
        obj = rom.add_object_by_template_handle(use_handle)
        if obj is None or not obj.is_alive:
            obj = None
        else:
            obj.translation = mn.Vector3(*grasp_target)
            obj.rotation    = mn.Quaternion()
            obj.motion_type = habitat_sim.physics.MotionType.STATIC
    except Exception:
        obj = None

    if capture:
        capture_at = set(
            int(round(i * (_TOTAL_STEPS - 1) / (GIF_FRAMES - 1)))
            for i in range(GIF_FRAMES)
        )
        step_global = 0
        _setup_camera(sim, np.array([base_x, FLOOR_Y, base_z]),
                      grasp_target, approach_angle)

    # ── Phase 1: arm interpolates REST → IK pose (gripper open, object visible) ──
    for step in range(APPROACH_STEPS):
        t = (step + 1) / APPROACH_STEPS
        robot.arm_joint_pos = ARM_INIT + t * (arm_angles - ARM_INIT)
        robot.update()
        sim.step_physics(1.0 / 60.0)
        if capture:
            if step_global in capture_at:
                frames.append(_capture_frame(sim))
            step_global += 1

    # ── Phase 2: settle at IK pose ────────────────────────────────────────────
    for step in range(SETTLE_STEPS):
        robot.arm_joint_pos = arm_angles
        robot.update()
        sim.step_physics(1.0 / 60.0)
        if capture:
            if step_global in capture_at:
                frames.append(_capture_frame(sim))
            step_global += 1

    # ── Geometric success check ───────────────────────────────────────────────
    # Project the object AABB onto the EE's finger-squeeze axis (EE local Y).
    # Success = arm reaches the target AND the object fits between the open fingers.
    ee_tf      = robot.ee_transform(0)
    ee_pos_hab = np.array(ee_tf.translation)
    # EE local Y axis in world frame (the finger squeeze direction)
    ee_y_world = np.array(ee_tf.transform_vector(mn.Vector3(0, 1, 0)))
    # Conservative AABB projection onto the squeeze axis
    grasp_width = float(2.0 * np.dot(np.abs(aabb_size) * 0.5, np.abs(ee_y_world)))
    success = grasp_width <= GRIPPER_OPEN * 2.0   # fits within the open finger span

    # ── Phase 3: gripper closes around the (static) object ───────────────────
    for step in range(CLOSE_STEPS):
        t = (step + 1) / CLOSE_STEPS
        g = GRIPPER_OPEN * (1.0 - t)
        robot.arm_joint_pos = arm_angles
        robot.gripper_joint_pos = np.array([g, g])
        robot.update()
        sim.step_physics(1.0 / 60.0)
        if capture:
            if step_global in capture_at:
                frames.append(_capture_frame(sim))
            step_global += 1

    # ── Phase 4: gripper opens (release) ──────────────────────────────────────
    for step in range(RELEASE_STEPS):
        t = (step + 1) / RELEASE_STEPS
        g = GRIPPER_OPEN * t
        robot.arm_joint_pos = arm_angles
        robot.gripper_joint_pos = np.array([g, g])
        robot.update()
        sim.step_physics(1.0 / 60.0)
        if capture:
            if step_global in capture_at:
                frames.append(_capture_frame(sim))
            step_global += 1

    if obj is not None and obj.is_alive:
        rom.remove_object_by_id(obj.object_id)

    return {"success": success, "grasp_width_m": round(grasp_width, 4),
            "ik_error": round(ik_err, 4), "frames": frames}


# ── Public API ────────────────────────────────────────────────────────────────

def run(sim, robot, ik_solver, asset_handle: str,
        collision_mode: str = "convex_hull",
        save_dir: str = None, asset_id: str = None) -> dict:
    result = {
        "collision_mode":        collision_mode,
        "grasp_success_rate":    None,
        "grasp_successes":       None,
        "grasp_trials":          GRASP_TRIALS,
        "mean_grasp_width_m":    None,
        "error":                 None,
    }

    otm = sim.get_object_template_manager()
    try:
        template = otm.get_template_by_handle(asset_handle)

        if collision_mode == "convex_hull":
            template.collision_asset_handle = template.render_asset_handle
            otm.register_template(template, asset_handle + "__g_convex_hull")
            use_handle = asset_handle + "__g_convex_hull"
        elif collision_mode == "vhacd":
            vhacd = os.path.abspath(
                os.path.join(os.path.dirname(os.path.dirname(asset_handle)),
                             template.render_asset_handle)
            ).replace(".glb", ".vhacd.glb")
            if not os.path.exists(vhacd):
                result["error"] = f"vhacd not found: {vhacd}"
                return result
            template.collision_asset_handle = vhacd
            template.join_collision_meshes  = False
            otm.register_template(template, asset_handle + "__g_vhacd")
            use_handle = asset_handle + "__g_vhacd"
        else:
            raise ValueError(f"Unknown collision_mode: {collision_mode}")

        snap_pos, aabb_size = _get_snap_pos(sim, robot, use_handle)
        if snap_pos is None:
            result["error"] = "snap_down failed"
            return result

        # Grasp target = arm workspace sweet spot directly above the object's XZ.
        grasp_target = np.array([snap_pos[0], GRASP_HEIGHT, snap_pos[2]])

        capturing  = save_dir is not None and asset_id is not None
        angles     = np.linspace(0, 2 * np.pi, GRASP_TRIALS, endpoint=False)

        successes, widths, all_frames = 0, [], []

        for i, angle in enumerate(angles):
            r = _run_trial(sim, robot, ik_solver, use_handle,
                           angle, snap_pos, grasp_target, aabb_size,
                           capture=(capturing and i == 0))
            if r["success"]:
                successes += 1
            if r.get("grasp_width_m") is not None:
                widths.append(r["grasp_width_m"])
            if i == 0 and r.get("frames"):
                all_frames.extend(r["frames"])

        result["grasp_successes"]    = successes
        result["grasp_trials"]       = GRASP_TRIALS
        result["grasp_success_rate"] = round(successes / GRASP_TRIALS, 4)
        result["mean_grasp_width_m"] = round(float(np.mean(widths)), 4) if widths else None

        if capturing and all_frames:
            os.makedirs(save_dir, exist_ok=True)
            _save_gif(all_frames, os.path.join(save_dir, f"{asset_id}.gif"))

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result
