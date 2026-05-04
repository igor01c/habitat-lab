#!/usr/bin/env python3

import sys
import os
import time

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("GLOG_minloglevel", "5")

import habitat_sim
import magnum as mn
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


SPAWN_CLEARANCE      = 0.10   # gap between collision mesh AABB bottom and floor Y=0 at spawn
PHYSICS_STEPS = 600      # 10 seconds at 1/60 s per step
GIF_FRAMES = 20          # number of frames captured across the full simulation
GIF_FPS = 10             # playback speed of the output GIF
FLY_THRESHOLD        = 1.5    # XZ displacement (m) → explosion/teleport
FLOOR_PENETRATION_THRESHOLD = -0.05   # penetration_y_m below this → floor penetration failure
SETTLE_LIN_THRESHOLD = 0.01   # linear velocity (m/s) below which object is considered settled
SETTLE_ANG_THRESHOLD = 0.1    # angular velocity (rad/s) below which object is considered settled

def _load_collision_vertices(config_json_path: str) -> np.ndarray:
    """Load collision mesh vertices in object-local Y-up frame (Nx3).

    Transform chain applied here (offline, once):
      raw GLB vertices → node scale+rotation (no translation) → template 'up' rotation

    At runtime only obj.rotation (physics rotation) + obj.translation are needed
    to reach world space.  Node translation is excluded because habitat-sim
    re-centres each collision sub-mesh at the shape's local origin and does not
    carry the GLB node offset into the simulated position.
    """
    import json, trimesh
    cfg = json.loads(open(config_json_path).read())
    collision_rel = cfg.get("collision_asset", cfg.get("render_asset"))
    collision_path = os.path.abspath(
        os.path.join(os.path.dirname(config_json_path), collision_rel))
    loaded = trimesh.load(collision_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        parts = []
        for node in loaded.graph.nodes_geometry:
            T, geom_name = loaded.graph[node]
            # Apply full node transform (including translation).
            # Habitat-sim uses the complete GLB node transform to position each
            # convex hull shape within the actor's local frame — stripping the
            # translation was incorrect for V-HACD assets (e.g. Objaverse) where
            # each hull sits at a different offset within the object.
            verts = trimesh.transformations.transform_points(
                loaded.geometry[geom_name].vertices, T)
            parts.append(np.array(verts))
        verts = np.vstack(parts)
    else:
        verts = np.array(loaded.vertices)

    # Rotate vertices so that the template 'up' axis aligns with world Y.
    up = np.array(cfg.get("up", [0.0, 1.0, 0.0]), dtype=float)
    up /= np.linalg.norm(up)
    world_up = np.array([0.0, 1.0, 0.0])
    if not np.allclose(up, world_up, atol=1e-6):
        R_up, _ = Rotation.align_vectors([world_up], [up])
        verts = (R_up.as_matrix() @ verts.T).T

    # Subtract raw mesh AABB centre — caller will replace this with Habitat's
    # hull AABB centre (from obj.aabb) so both frames agree.
    aabb_center = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
    verts -= aabb_center
    return verts, aabb_center


def _world_min_y(verts_local: np.ndarray, obj) -> float:
    """Return lowest world Y of mesh vertices given the object's current pose."""
    t = np.array(obj.translation)
    mq = obj.rotation  # mn.Quaternion — physics rotation only (up already in verts)
    R = Rotation.from_quat([mq.vector.x, mq.vector.y, mq.vector.z, mq.scalar]).as_matrix()
    world_y = (R @ verts_local.T)[1] + t[1]
    return float(world_y.min())


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


def _save_debug_html(verts_local: np.ndarray, obj, save_path: str,
                     config_json_path: str = None) -> None:
    """Save interactive 3D HTML showing floor plane + collision mesh at final pose."""
    import json
    import plotly.graph_objects as go
    import trimesh

    t = np.array(obj.translation)
    mq = obj.rotation
    R = Rotation.from_quat([mq.vector.x, mq.vector.y, mq.vector.z, mq.scalar]).as_matrix()

    traces = []

    # Collision mesh with actual faces
    if config_json_path is not None:
        try:
            cfg = json.loads(open(config_json_path).read())
            collision_rel = cfg.get("collision_asset", cfg.get("render_asset"))
            collision_path = os.path.abspath(
                os.path.join(os.path.dirname(config_json_path), collision_rel))
            loaded = trimesh.load(collision_path, process=False)

            up = cfg.get("up", [0.0, 1.0, 0.0])
            world_up = np.array([0.0, 1.0, 0.0])
            asset_up = np.array(up, dtype=float)
            asset_up /= np.linalg.norm(asset_up)
            if not np.allclose(asset_up, world_up):
                rot_up, _ = Rotation.align_vectors([world_up], [asset_up])
            else:
                rot_up = Rotation.identity()

            def _local_verts(verts, node_T):
                """Apply full node transform then up-rotation."""
                v = trimesh.transformations.transform_points(verts, node_T)
                return (rot_up.as_matrix() @ v.T).T

            # Collect all sub-meshes in local space, compute shared AABB center
            parts_local_v, parts_f = [], []
            if isinstance(loaded, trimesh.Scene):
                offset = 0
                for node in loaded.graph.nodes_geometry:
                    node_T, geom_name = loaded.graph[node]
                    sub = loaded.geometry[geom_name]
                    v = _local_verts(np.array(sub.vertices), node_T)
                    parts_local_v.append(v)
                    parts_f.append(np.array(sub.faces) + offset)
                    offset += len(v)
            else:
                v = _local_verts(np.array(loaded.vertices), np.eye(4))
                parts_local_v.append(v)
                parts_f.append(np.array(loaded.faces))

            all_local_v = np.vstack(parts_local_v)
            # Mirror _load_collision_vertices: subtract AABB center so mesh is
            # centred at the body origin, matching how habitat-sim positions shapes.
            aabb_center = (all_local_v.max(axis=0) + all_local_v.min(axis=0)) / 2.0
            all_local_v -= aabb_center

            # Apply physics pose (rotation + translation)
            all_world_v = (R @ all_local_v.T).T + t
            all_f = np.vstack(parts_f)

            traces.append(go.Mesh3d(
                x=all_world_v[:, 0], y=all_world_v[:, 1], z=all_world_v[:, 2],
                i=all_f[:, 0], j=all_f[:, 1], k=all_f[:, 2],
                color="steelblue", opacity=0.6, name="collision mesh",
                lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
            ))
        except Exception:
            pass

    # Fallback: vertex scatter if mesh loading failed
    if not traces:
        world_verts = (R @ verts_local.T).T + t
        traces.append(go.Scatter3d(
            x=world_verts[:, 0], y=world_verts[:, 1], z=world_verts[:, 2],
            mode="markers", marker=dict(size=2, color="steelblue"),
            name="mesh vertices",
        ))

    world_verts_approx = (R @ verts_local.T).T + t
    min_y = float(world_verts_approx[:, 1].min())
    extent = float(max(world_verts_approx[:, 0].ptp(), world_verts_approx[:, 2].ptp(), 0.2)) * 0.6 + 0.1
    cx, cz = float(t[0]), float(t[2])
    grid = np.linspace(-extent + cx, extent + cx, 4)
    gx, gz = np.meshgrid(grid, grid)
    gy = np.zeros_like(gx)
    traces.insert(0, go.Surface(
        x=gx, y=gy, z=gz, opacity=0.25,
        colorscale=[[0, "cyan"], [1, "cyan"]], showscale=False, name="floor Y=0",
    ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"min world Y = {min_y:.4f} m  |  obj.translation Y = {t[1]:.4f} m",
        scene=dict(xaxis_title="X", yaxis_title="Y (up)", zaxis_title="Z",
                   aspectmode="data"),
    )
    fig.write_html(save_path)


def _save_gif(frames: list, save_path: str) -> None:
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / GIF_FPS),
        loop=0,
    )


PENETRATION_METHODS = ("vertex", "contact", "raycast")


def _penetration_contact(sim, obj, physics_steps, capture_fn=None, capture_at=None):
    """Track minimum floor-contact Y over the simulation (contact-point method).

    The static stage (floor/walls) has object_id = 0 in Bullet-backed habitat-sim.
    habitat_sim.physics.STAGE_ID was added in a later version and may not exist.
    """
    stage_id = getattr(habitat_sim.physics, "STAGE_ID", 0)
    min_contact_y = 0.0
    settle_step = None
    for step in range(physics_steps):
        sim.step_physics(1.0 / 60.0)
        for c in sim.get_physics_contact_points():
            if obj.object_id in (c.object_id_a, c.object_id_b):
                # position_on_b_in_ws is world-space contact on body B;
                # use whichever position is on the floor (stage) side
                if c.object_id_b == stage_id or c.object_id_a == stage_id:
                    pt_y = float(c.position_on_b_in_ws[1])
                    if pt_y < min_contact_y:
                        min_contact_y = pt_y
        lin = float(mn.Vector3(obj.linear_velocity).length())
        ang = float(mn.Vector3(obj.angular_velocity).length())
        if settle_step is None:
            if lin < SETTLE_LIN_THRESHOLD and ang < SETTLE_ANG_THRESHOLD:
                settle_step = step
        else:
            if lin >= SETTLE_LIN_THRESHOLD or ang >= SETTLE_ANG_THRESHOLD:
                settle_step = None
        if capture_fn and capture_at and step in capture_at:
            capture_fn()
    return min_contact_y, settle_step


def _penetration_raycast(sim, obj, physics_steps, capture_fn=None, capture_at=None):
    """Track minimum bottom Y by casting a ray upward from below after each step."""
    min_bottom_y = SPAWN_CLEARANCE
    settle_step = None
    for step in range(physics_steps):
        sim.step_physics(1.0 / 60.0)
        # Cast ray upward from below the object's current XZ at Y = -0.5 m
        t = obj.translation
        ray = habitat_sim.geo.Ray(
            mn.Vector3(t.x, -0.5, t.z),
            mn.Vector3(0.0, 1.0, 0.0),
        )
        result = sim.cast_ray(ray)
        if result.has_hits():
            for hit in result.hits:
                if hit.object_id == obj.object_id:
                    bottom_y = float(hit.point.y)
                    if bottom_y < min_bottom_y:
                        min_bottom_y = bottom_y
                    break  # first hit on our object is the bottom
        lin = float(mn.Vector3(obj.linear_velocity).length())
        ang = float(mn.Vector3(obj.angular_velocity).length())
        if settle_step is None:
            if lin < SETTLE_LIN_THRESHOLD and ang < SETTLE_ANG_THRESHOLD:
                settle_step = step
        else:
            if lin >= SETTLE_LIN_THRESHOLD or ang >= SETTLE_ANG_THRESHOLD:
                settle_step = None
        if capture_fn and capture_at and step in capture_at:
            capture_fn()
    return min(min_bottom_y, 0.0), settle_step


def run(sim: habitat_sim.Simulator, asset_handle: str, collision_mode: str = "convex_hull",
        save_dir: str = None, asset_id: str = None,
        config_json_path: str = None, debug_html: str = None,
        penetration_method: str = "vertex") -> dict:
    """
    asset_handle:     template handle returned by otm.get_template_handles()
    config_json_path: path to .object_config.json — used to load collision mesh vertices
                      for accurate world-space penetration tracking under rotation.
    collision_mode:   "convex_hull" | "vhacd"
    debug_html:       if set, save a 3D HTML visualisation of the floor + transformed
                      mesh vertices at the final resting pose to this path.
    """
    result = {
        "collision_mode": collision_mode,
        "physics_settles": None,
        "physics_stable": None,
        "displacement_m": None,
        "flies_away": None,
        "penetration_y_m": None,
        "floor_penetration": None,
        "settle_time_s": None,
        "wall_time_s": None,
        "contact_points_at_rest": None,
        "error": None,
    }
    _t0 = time.perf_counter()

    rom = sim.get_rigid_object_manager()
    otm = sim.get_object_template_manager()
    obj = None
    use_handle = None

    try:
        template = otm.get_template_by_handle(asset_handle)

        if collision_mode == "convex_hull":
            otm.register_template(template, asset_handle + "__convex_hull")
            use_handle = asset_handle + "__convex_hull"

        elif collision_mode == "vhacd":
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

        # Spawn: use obj.aabb (collision shape AABB) rather than cumulative_bb (visual
        # mesh AABB). For Objaverse where collision_asset == render_asset, Habitat
        # builds an internal convex hull; obj.aabb reflects that hull, not the raw mesh.
        # Read AABB at identity pose (translation=0) so it gives the local-frame hull AABB.
        local_hull_aabb = obj.aabb
        mesh_aabb_min_y = float(local_hull_aabb.min[1])
        spawn_y = -mesh_aabb_min_y + SPAWN_CLEARANCE
        spawn_pos = np.array([0.0, spawn_y, 0.0])
        obj.translation = mn.Vector3(*spawn_pos)
        obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC

        # vertex method requires external mesh loading; contact/raycast use Habitat only
        collision_verts = None
        if penetration_method == "vertex":
            if config_json_path is None:
                raise ValueError("config_json_path is required for vertex penetration method")
            collision_verts, _ = _load_collision_vertices(config_json_path)
            # Normalize vertices so their AABB exactly matches Habitat's collision hull AABB
            # (from local_hull_aabb, read at identity pose above). This corrects for any
            # discrepancy between the raw mesh AABB and the hull Habitat builds internally —
            # e.g. Objaverse GLBs from Sketchfab have a Z-up→Y-up root rotation that
            # Habitat's Magnum loader applies differently from trimesh, producing a ~7%
            # scale difference in Y. Without this, spawn-alignment errors of 3–46 cm appear.
            hab_min = np.array([local_hull_aabb.min[i] for i in range(3)])
            hab_max = np.array([local_hull_aabb.max[i] for i in range(3)])
            raw_min = collision_verts.min(axis=0)
            raw_max = collision_verts.max(axis=0)
            raw_half = (raw_max - raw_min) / 2.0
            hab_half = (hab_max - hab_min) / 2.0
            hab_center = (hab_max + hab_min) / 2.0
            scale = np.where(raw_half > 1e-8, hab_half / raw_half, 1.0)
            collision_verts = collision_verts * scale + hab_center

        capturing = save_dir is not None and asset_id is not None
        if capturing:
            os.makedirs(save_dir, exist_ok=True)
            bb = obj.root_scene_node.cumulative_bb
            aabb_size = np.array([bb.size_x(), bb.size_y(), bb.size_z()])
            _setup_camera(sim, np.zeros(3), aabb_size, spawn_pos)
            gif_frames: list = []
            capture_at = set(
                int(round(i * (PHYSICS_STEPS - 1) / (GIF_FRAMES - 1)))
                for i in range(GIF_FRAMES)
            )

        capture_fn = (lambda: gif_frames.append(_capture_frame(sim))) if capturing else None

        if penetration_method == "contact":
            min_mesh_bottom_y, settle_step = _penetration_contact(
                sim, obj, PHYSICS_STEPS, capture_fn, capture_at if capturing else None)
            final_bottom_y = min_mesh_bottom_y  # contacts don't give a separate final value
        elif penetration_method == "raycast":
            min_mesh_bottom_y, settle_step = _penetration_raycast(
                sim, obj, PHYSICS_STEPS, capture_fn, capture_at if capturing else None)
            final_bottom_y = min_mesh_bottom_y
        else:  # vertex
            min_mesh_bottom_y = SPAWN_CLEARANCE
            settle_step = None
            for step in range(PHYSICS_STEPS):
                sim.step_physics(1.0 / 60.0)
                mesh_bottom_y = _world_min_y(collision_verts, obj)
                if mesh_bottom_y < min_mesh_bottom_y:
                    min_mesh_bottom_y = mesh_bottom_y
                lin = float(mn.Vector3(obj.linear_velocity).length())
                ang = float(mn.Vector3(obj.angular_velocity).length())
                if settle_step is None:
                    if lin < SETTLE_LIN_THRESHOLD and ang < SETTLE_ANG_THRESHOLD:
                        settle_step = step
                else:
                    if lin >= SETTLE_LIN_THRESHOLD or ang >= SETTLE_ANG_THRESHOLD:
                        settle_step = None
                if capturing and step in capture_at:
                    gif_frames.append(_capture_frame(sim))
            final_bottom_y = _world_min_y(collision_verts, obj)

        final_pos = np.array(obj.translation)

        # --- Sanity checks on mesh transform alignment ---
        # The object was dropped from above the floor; it is physically impossible
        # for the mesh bottom to end above Y=0 unless our vertex transform is wrong.
        # Bullet's per-convex-hull collision margin holds the mesh vertices above
        # Y=0 even at floor contact. This is a known engine artifact, not a
        # transform error — confirmed by the spawn alignment check passing (transform
        # is exact at identity rotation; positive values only appear post-simulation
        # after rotation + contact). Observed ceiling: ~1.5 cm for thin VHACD assets
        # (shoji door). 2 cm gives safe clearance; real transform bugs show at dm scale.
        _FLOAT_TOL = 0.02
        if min_mesh_bottom_y > _FLOAT_TOL:
            msg = (
                f"min_mesh_bottom_y={min_mesh_bottom_y:.4f} m > {_FLOAT_TOL} m: "
                "collision vertex transform is misaligned with Habitat-sim physics"
            )
            print(f"\n[ASSERT] {asset_id} ({collision_mode}): {msg}", flush=True)
            raise AssertionError(msg)
        if final_bottom_y > _FLOAT_TOL:
            msg = (
                f"final_bottom_y={final_bottom_y:.4f} m > {_FLOAT_TOL} m: "
                "collision vertex transform is misaligned with Habitat-sim physics"
            )
            print(f"\n[ASSERT] {asset_id} ({collision_mode}): {msg}", flush=True)
            raise AssertionError(msg)
        # Excessively deep penetration suggests the same transform bug in the other
        # direction — keep the asset but emit a prominent warning.
        _DEEP_WARN = FLOOR_PENETRATION_THRESHOLD * 10   # −0.50 m
        if min_mesh_bottom_y < _DEEP_WARN:
            print(
                f"\n[WARN] {asset_id} ({collision_mode}): "
                f"min_mesh_bottom_y={min_mesh_bottom_y:.4f} m — "
                f"far below floor ({_DEEP_WARN:.2f} m threshold). "
                "Possible collision vertex transform error.",
                flush=True,
            )
        if settle_step is not None and final_bottom_y < _DEEP_WARN:
            print(
                f"\n[WARN] {asset_id} ({collision_mode}): "
                f"final_bottom_y={final_bottom_y:.4f} m after settling — "
                f"far below floor ({_DEEP_WARN:.2f} m threshold). "
                "Possible collision vertex transform error.",
                flush=True,
            )
        # -------------------------------------------------

        if capturing:
            _save_gif(gif_frames, os.path.join(save_dir, f"{asset_id}.gif"))

        if debug_html is not None and collision_verts is not None:
            _save_debug_html(collision_verts, obj, debug_html,
                             config_json_path=config_json_path)

        displacement   = float(np.linalg.norm((final_pos - spawn_pos)[[0, 2]]))
        penetration_y  = round(min(float(min_mesh_bottom_y), 0.0), 4)
        obj_contacts   = [c for c in sim.get_physics_contact_points()
                          if obj.object_id in (c.object_id_a, c.object_id_b)]

        flies_away       = displacement > FLY_THRESHOLD
        floor_penetration = penetration_y < FLOOR_PENETRATION_THRESHOLD
        physics_settles  = settle_step is not None

        result["displacement_m"]         = round(displacement, 4)
        result["flies_away"]             = flies_away
        result["penetration_y_m"]        = penetration_y
        result["floor_penetration"]      = floor_penetration
        result["settle_time_s"]          = round(settle_step / 60.0, 3) if settle_step is not None else None
        result["contact_points_at_rest"] = len(obj_contacts)
        result["physics_settles"]        = physics_settles
        result["physics_stable"]         = physics_settles and not flies_away and not floor_penetration

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        result["wall_time_s"] = round(time.perf_counter() - _t0, 3)
        if obj is not None and obj.is_alive:
            rom.remove_object_by_id(obj.object_id)
        if use_handle is not None and otm.get_library_has_handle(use_handle):
            otm.remove_template_by_handle(use_handle)

    return result
