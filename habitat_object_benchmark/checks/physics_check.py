#!/usr/bin/env python3

import sys
import os

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("GLOG_minloglevel", "5")

import habitat_sim
import magnum as mn
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "habitat-lab"))
from habitat.sims.habitat_simulator.sim_utilities import snap_down

FLOOR_Y = 0.0                 # top surface of the simple floor plane (matches sim_factory)
SPAWN_HEIGHT = 0.2            # gap between object bottom (AABB min Y) and floor at spawn
PHYSICS_STEPS = 300      # 5 seconds at 1/60 s per step
GIF_FRAMES = 20          # number of frames captured across the full simulation
GIF_FPS = 10             # playback speed of the output GIF
FLY_THRESHOLD        = 1.5    # XZ displacement → explosion/teleport (relaxed: rolling/tipping OK)
FINAL_SINK_THRESHOLD = -0.10  # final_y_offset below this → permanent floor penetration (strict)
MIN_SINK_THRESHOLD   = -0.15  # min_y_offset below this → severe trajectory penetration (strict)
SETTLE_VEL_THRESHOLD = 0.01   # linear velocity (m/s) below which object is considered settled
def _look_at_rotation(eye: np.ndarray, target: np.ndarray):
    """Returns a numpy quaternion that orients the agent from eye toward target."""
    import quaternion as qt
    delta = target - eye
    yaw = float(np.arctan2(delta[0], -delta[2]))
    horiz = float(np.sqrt(delta[0] ** 2 + delta[2] ** 2))
    pitch = float(np.arctan2(delta[1], horiz))
    q_yaw = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3(0, 1, 0))
    q_pitch = mn.Quaternion.rotation(mn.Rad(pitch), mn.Vector3(1, 0, 0))
    mq = q_yaw * q_pitch
    return qt.quaternion(mq.scalar, mq.vector[0], mq.vector[1], mq.vector[2])


def _setup_camera(sim: habitat_sim.Simulator, snap_pos: np.ndarray, aabb_size: np.ndarray,
                  spawn_pos: np.ndarray = None):
    # Frame the full drop: target at midpoint between floor and spawn,
    # eye at same height offset back far enough to see the whole trajectory.
    top_y = float(spawn_pos[1]) if spawn_pos is not None else snap_pos[1] + aabb_size[1] * 2.0
    mid_y = (snap_pos[1] + top_y) / 2.0
    target = np.array([snap_pos[0], mid_y, snap_pos[2]])

    drop_height = top_y - snap_pos[1]
    dist = max(drop_height * 1.2, aabb_size[2] * 3.0, 0.8)
    eye = np.array([snap_pos[0], mid_y, snap_pos[2] + dist])

    agent = sim.get_agent(0)
    # Reset to default so sensor_states give the local offset, not a stale world position.
    agent.set_state(agent.get_state().__class__())
    default_state = agent.get_state()
    rot = _look_at_rotation(eye, target)
    # Compute agent body position so rotated sensor offset lands at eye.
    # Rotate local sensor offset (0, sensor_h, 0) by agent quaternion to get world offset.
    import quaternion as qt
    sensor_h = float(default_state.sensor_states["color"].position[1])
    local_v = qt.quaternion(0, 0, sensor_h, 0)
    world_v = rot * local_v * rot.conjugate()
    world_offset = np.array([world_v.x, world_v.y, world_v.z])
    default_state.position = eye - world_offset
    default_state.rotation = rot
    agent.set_state(default_state)
    return eye, target


def _capture_frame(sim: habitat_sim.Simulator) -> Image.Image:
    obs = sim.get_sensor_observations()
    return Image.fromarray(obs["color"][:, :, :3])


def _render_frame(sim: habitat_sim.Simulator, save_path: str) -> None:
    _capture_frame(sim).save(save_path)


def _save_gif(frames: list, save_path: str) -> None:
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / GIF_FPS),
        loop=0,
    )


def run(sim: habitat_sim.Simulator, asset_handle: str, collision_mode: str = "convex_hull",
        save_dir: str = None, asset_id: str = None) -> dict:
    """
    asset_handle:   template handle returned by otm.get_template_handles()
    collision_mode: "convex_hull" → render mesh as collision (1 convex hull)
                    "vhacd"       → <name>.vhacd.glb as collision (N-part compound)
    """
    result = {
        "collision_mode": collision_mode,
        "physics_settles": False,
        "displacement_m": None,
        "flies_away": None,
        "final_y_offset_m": None,
        "sinks_permanently": None,
        "min_y_offset_m": None,
        "sinks_below_floor": None,
        "settle_time_s": None,
        "contact_points_at_rest": None,
        "error": None,
    }

    rom = sim.get_rigid_object_manager()
    otm = sim.get_object_template_manager()
    obj = None
    use_handle = None

    try:
        template = otm.get_template_by_handle(asset_handle)

        if collision_mode == "convex_hull":
            # Use the dedicated collision mesh from the object config (single convex hull).
            otm.register_template(template, asset_handle + "__convex_hull")
            use_handle = asset_handle + "__convex_hull"

        elif collision_mode == "vhacd":
            # Use the pre-generated VHACD GLB (N-part compound collision shape)
            render_path = os.path.abspath(
                os.path.join(os.path.dirname(os.path.dirname(asset_handle)),
                             template.render_asset_handle))
            vhacd_path = render_path.replace(".glb", ".vhacd.glb")
            if not os.path.exists(vhacd_path):
                result["error"] = f"vhacd file not found: {vhacd_path}"
                return result
            template.collision_asset_handle = vhacd_path
            template.join_collision_meshes = False
            otm.register_template(template, asset_handle + "__vhacd")
            use_handle = asset_handle + "__vhacd"

        else:
            raise ValueError(f"Unknown collision_mode: {collision_mode}")

        obj = rom.add_object_by_template_handle(use_handle)
        if obj is None or not obj.is_alive:
            result["error"] = "failed to instantiate object"
            return result

        # Use snap_down to find the resting position on the floor, then raise the
        # object by SPAWN_HEIGHT so the GIF shows it actually dropping.
        aabb_min_y = float(obj.root_scene_node.cumulative_bb.min[1])
        obj.translation = mn.Vector3(0.0, FLOOR_Y + 0.5 - aabb_min_y, 0.0)
        obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC

        # Resolve the floor object id so snap_down uses it as the support surface.
        rom_tmp = sim.get_rigid_object_manager()
        floor_ids = [
            rom_tmp.get_object_by_handle(h).object_id
            for h in rom_tmp.get_object_handles("__floor_plane__")
        ]
        support_ids = floor_ids if floor_ids else None

        # max_collision_depth=0.5: Bullet convex-hull contact depths are often
        # inflated; the actual settling quality is measured by our displacement metrics.
        snapped = snap_down(sim, obj, support_obj_ids=support_ids, max_collision_depth=0.5)
        if not snapped:
            result["error"] = "snap_down failed — no valid surface below spawn point"
            return result

        # snap_pos is the floor-level rest position — used for displacement metrics.
        snap_pos = np.array(obj.translation)

        # Raise the object SPAWN_HEIGHT above snap so the GIF shows it dropping.
        obj.translation = mn.Vector3(snap_pos[0], snap_pos[1] + SPAWN_HEIGHT, snap_pos[2])

        capturing = save_dir is not None and asset_id is not None
        if capturing:
            os.makedirs(save_dir, exist_ok=True)
            bb = obj.root_scene_node.cumulative_bb
            aabb_size = np.array([bb.size_x(), bb.size_y(), bb.size_z()])
            spawn_pos = np.array(obj.translation)
            _setup_camera(sim, snap_pos, aabb_size, spawn_pos)
            gif_frames: list = []
            # Evenly-spaced step indices at which to capture a frame
            capture_at = set(
                int(round(i * (PHYSICS_STEPS - 1) / (GIF_FRAMES - 1)))
                for i in range(GIF_FRAMES)
            )

        # Run simulation, tracking minimum Y and settle time.
        min_y = float(obj.translation[1])
        settle_step = None  # first step where velocity stays below threshold

        for step in range(PHYSICS_STEPS):
            sim.step_physics(1.0 / 60.0)
            y = float(obj.translation[1])
            if y < min_y:
                min_y = y
            if settle_step is None:
                vel = obj.linear_velocity
                speed = float(mn.Vector3(vel).length())
                if speed < SETTLE_VEL_THRESHOLD:
                    settle_step = step
            elif obj.linear_velocity is not None:
                vel = obj.linear_velocity
                speed = float(mn.Vector3(vel).length())
                if speed >= SETTLE_VEL_THRESHOLD:
                    settle_step = None  # object moved again, reset
            if capturing and step in capture_at:
                gif_frames.append(_capture_frame(sim))

        final_pos = np.array(obj.translation)

        if capturing:
            _save_gif(gif_frames, os.path.join(save_dir, f"{asset_id}.gif"))

        displacement     = float(np.linalg.norm((final_pos - snap_pos)[[0, 2]]))
        final_y_offset   = round(float(final_pos[1]) - float(snap_pos[1]), 4)
        min_y_offset     = round(min_y - float(snap_pos[1]), 4)
        obj_contacts     = [c for c in sim.get_physics_contact_points()
                            if obj.object_id in (c.object_id_a, c.object_id_b)]

        result["displacement_m"]         = round(displacement, 4)
        result["flies_away"]             = displacement > FLY_THRESHOLD
        result["final_y_offset_m"]       = final_y_offset
        result["sinks_permanently"]      = final_y_offset < FINAL_SINK_THRESHOLD
        result["min_y_offset_m"]         = min_y_offset
        result["sinks_below_floor"]      = min_y_offset < MIN_SINK_THRESHOLD
        result["settle_time_s"]          = round(settle_step / 60.0, 3) if settle_step is not None else None
        result["contact_points_at_rest"] = len(obj_contacts)
        # Pass/fail: only floor penetration and explosion are hard failures.
        # Rolling, tipping, and slow settling are allowed.
        result["physics_settles"] = (
            not (displacement > FLY_THRESHOLD)
            and final_y_offset >= FINAL_SINK_THRESHOLD
            and min_y_offset   >= MIN_SINK_THRESHOLD
        )

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        if obj is not None and obj.is_alive:
            rom.remove_object_by_id(obj.object_id)
        if use_handle is not None and otm.get_library_has_handle(use_handle):
            otm.remove_template_by_handle(use_handle)

    return result
