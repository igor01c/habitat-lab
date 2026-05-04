#!/usr/bin/env python3
"""Physics stability check using SAPIEN 3 / ManiSkill3 (PhysX backend).

Coordinate convention: SAPIEN uses Z-up, gravity = -Z.
Output dict schema is identical to physics_check.py (Bullet version).
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sapien
import sapien.physx as physx
from scipy.spatial.transform import Rotation

from habitat_object_benchmark.utils.maniskill_factory import (
    get_rb,
    load_object,
    make_scene,
)

SPAWN_CLEARANCE             = 0.10   # m — AABB bottom above floor at spawn
PHYSICS_STEPS               = 600    # 10 s at 60 Hz
GIF_FRAMES                  = 20
GIF_FPS                     = 10
FLY_THRESHOLD               = 1.5    # m — XY displacement → flies away
FLOOR_PENETRATION_THRESHOLD = -0.05  # m — penetration below this → failure
SETTLE_VEL_THRESHOLD        = 0.01   # m/s linear
SETTLE_ANG_THRESHOLD        = 0.1    # rad/s angular


def _save_gif(frames: list, save_path: str) -> None:
    import imageio
    arrays = [np.array(f.convert("RGB")) for f in frames]
    imageio.mimsave(save_path, arrays, fps=GIF_FPS, loop=0)


def _make_lookat_matrix(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera pose matrix (4x4) for a camera at eye looking at target (OpenGL/pyrender convention)."""
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    right_n = np.linalg.norm(right)
    if right_n < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= right_n
    up = np.cross(right, fwd)
    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = up
    mat[:3, 2] = -fwd
    mat[:3, 3] = eye
    return mat


class _PyRenderScene:
    """Lightweight pyrender offscreen scene for physics GIF capture."""

    def __init__(self, render_path: str, spawn_z: float,
                 aabb_extent: np.ndarray = None,
                 orient_q_wxyz: np.ndarray = None,
                 camera_dist: float = None,
                 width: int = 640, height: int = 480):
        import os
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import pyrender
        import trimesh
        import trimesh.transformations as tf

        # ── Load mesh first so we can compute its true bounding sphere ────────
        orig_path = render_path + ".orig"
        load_path = orig_path if os.path.exists(orig_path) else render_path
        file_type = "glb" if load_path.endswith(".orig") else None
        try:
            kwargs = {"process": False}
            if file_type:
                kwargs["file_type"] = file_type
            loaded = trimesh.load(load_path, **kwargs)
            if isinstance(loaded, trimesh.Scene):
                meshes = []
                for node in loaded.graph.nodes_geometry:
                    node_T, geom = loaded.graph[node]
                    m = loaded.geometry[geom].copy()
                    m.apply_transform(node_T)
                    meshes.append(m)
                if not meshes:
                    meshes = list(loaded.geometry.values())
            else:
                meshes = [loaded]
            obj_mesh = pyrender.Mesh.from_trimesh(meshes, smooth=True)
            all_verts = np.vstack([m.vertices for m in meshes])
            mesh_extent = all_verts.max(axis=0) - all_verts.min(axis=0)
            bounding_r  = float(np.linalg.norm(mesh_extent)) / 2.0
        except Exception:
            bounding_r = 0.2
            obj_mesh = pyrender.Mesh.from_trimesh(
                trimesh.creation.icosphere(radius=bounding_r), smooth=False
            )

        # ── Camera ────────────────────────────────────────────────────────────
        safe_spawn_z = float(np.clip(spawn_z, 0.1, 5.0))
        mid_z = safe_spawn_z / 2.0
        dist  = camera_dist if camera_dist is not None else float(np.clip(
            max(safe_spawn_z * 1.5, bounding_r * 3.0), bounding_r * 2.0 + 0.1, 10.0
        ))
        eye      = np.array([0.0, -dist, mid_z])
        target   = np.array([0.0, 0.0,  mid_z])
        cam_pose = _make_lookat_matrix(eye, target)

        self._scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[0.15, 0.15, 0.2])

        floor_half = float(np.clip(bounding_r * 3.0, 1.0, 10.0))
        floor_mesh = trimesh.creation.box([floor_half * 2, floor_half * 2, 0.005])
        floor_mesh.visual.vertex_colors = [120, 120, 120, 255]
        self._scene.add(pyrender.Mesh.from_trimesh(floor_mesh, smooth=False),
                        pose=tf.translation_matrix([0, 0, 0.0025]))
        self._obj_node = self._scene.add(obj_mesh, pose=np.eye(4))

        # Store orient rotation so capture() can compose: entity_rotation * orient_rotation
        if orient_q_wxyz is not None:
            q = orient_q_wxyz
            self._R_orient = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        else:
            self._R_orient = np.eye(3)

        key = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
        self._scene.add(key, pose=_make_lookat_matrix(
            np.array([1.5, -2.0, 3.0]), np.array([0.0, 0.0, 0.8])
        ))
        fill = pyrender.DirectionalLight(color=[0.7, 0.8, 1.0], intensity=2.0)
        self._scene.add(fill, pose=_make_lookat_matrix(
            np.array([-2.0, -1.0, 1.5]), np.array([0.0, 0.0, 0.8])
        ))

        cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(60), znear=0.01, zfar=100.0)
        self._scene.add(cam, pose=cam_pose)
        self._renderer = pyrender.OffscreenRenderer(width, height)

    def capture(self, obj_pose_p: np.ndarray, obj_pose_q: np.ndarray) -> "Image":
        """Render current frame. obj_pose_q is [w, x, y, z]."""
        from PIL import Image
        q_xyzw = [obj_pose_q[1], obj_pose_q[2], obj_pose_q[3], obj_pose_q[0]]
        R_entity = Rotation.from_quat(q_xyzw).as_matrix()
        mat = np.eye(4)
        mat[:3, :3] = R_entity @ self._R_orient
        mat[:3, 3] = obj_pose_p
        self._scene.set_pose(self._obj_node, mat)
        color, _ = self._renderer.render(self._scene)
        return Image.fromarray(color)

    def close(self):
        try:
            self._renderer.delete()
        except Exception:
            pass


def run(
    scene: sapien.Scene,
    config_json_path: str,
    collision_mode: str = "convex_hull",
    save_dir: Optional[str] = None,
    asset_id: Optional[str] = None,
    camera_dist: Optional[float] = None,
) -> dict:
    """Run physics stability check.

    scene: a freshly-created SAPIEN scene (make_scene()) with no persistent objects.
    config_json_path: path to .object_config.json.
    Returns same dict schema as physics_check.run().
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

    obj_entity = None
    pr_scene = None
    _t0 = None

    try:
        cfg = json.loads(Path(config_json_path).read_text())
        config_dir = Path(config_json_path).parent

        # Capture SAPIEN's C-level stderr to detect silent collision loading failures.
        _pipe_r, _pipe_w = os.pipe()
        _old_stderr = os.dup(2)
        os.dup2(_pipe_w, 2)
        os.close(_pipe_w)
        try:
            obj_entity = load_object(scene, config_json_path,
                                     collision_mode=collision_mode,
                                     name=asset_id or "obj")
        finally:
            os.dup2(_old_stderr, 2)
            os.close(_old_stderr)
            _sapien_log = os.read(_pipe_r, 65536).decode(errors="replace")
            os.close(_pipe_r)

        if "failed to load a component" in _sapien_log:
            raise RuntimeError(f"SAPIEN could not load collision mesh: {_sapien_log[:300].strip()}")

        rb = get_rb(obj_entity)

        _t0 = time.perf_counter()

        # ── Spawn height from SAPIEN's own physics AABB ───────────────────────
        # Query the world AABB at identity pose (kinematic so gravity doesn't move it).
        rb.set_kinematic(True)
        obj_entity.set_pose(sapien.Pose())
        scene.step()
        _aabb_full  = rb.compute_global_aabb_tight()
        aabb_min_z  = float(_aabb_full[0][2])
        aabb_extent = np.array(_aabb_full[1]) - np.array(_aabb_full[0])
        spawn_z = -aabb_min_z + SPAWN_CLEARANCE
        obj_entity.set_pose(sapien.Pose(p=[0.0, 0.0, spawn_z]))
        rb.set_kinematic(False)
        rb.wake_up()
        spawn_pos = np.array(obj_entity.get_pose().p)

        # ── GIF setup ─────────────────────────────────────────────────────────
        capturing = save_dir is not None and asset_id is not None
        pr_scene = None
        if capturing:
            render_path = str((config_dir / cfg["render_asset"]).resolve())
            os.makedirs(save_dir, exist_ok=True)
            from habitat_object_benchmark.utils.maniskill_factory import _up_to_sapien_pose
            orient_pose = _up_to_sapien_pose(cfg.get("up", [0.0, 0.0, 1.0]))
            pr_scene = _PyRenderScene(
                render_path, spawn_z,
                aabb_extent=aabb_extent,
                orient_q_wxyz=np.array(orient_pose.q),
                camera_dist=camera_dist,
            )
            gif_frames = []
            # Bias frames toward the early drop phase (first 10% of steps)
            # so the fall is visible even when PhysX settles quickly.
            early_end  = PHYSICS_STEPS // 10          # step 60
            early_frames = GIF_FRAMES // 2            # 10 frames in first 10%
            late_frames  = GIF_FRAMES - early_frames  # 10 frames in remaining 90%
            capture_at = set(
                int(round(i * (early_end - 1) / max(early_frames - 1, 1)))
                for i in range(early_frames)
            ) | set(
                early_end + int(round(i * (PHYSICS_STEPS - 1 - early_end) / max(late_frames - 1, 1)))
                for i in range(late_frames)
            )

        # ── Simulation loop ───────────────────────────────────────────────────
        min_mesh_bottom_z = SPAWN_CLEARANCE   # object bottom starts at 0.10 m
        settle_step = None
        exploded = False

        for step in range(PHYSICS_STEPS):
            scene.step()

            bottom_z = float(rb.compute_global_aabb_tight()[0][2])
            if bottom_z < min_mesh_bottom_z:
                min_mesh_bottom_z = bottom_z

            pose = obj_entity.get_pose()
            actor_pos = np.array(pose.p)
            actor_q   = np.array(pose.q)

            xy_disp = float(np.linalg.norm((actor_pos - spawn_pos)[:2]))
            if xy_disp > FLY_THRESHOLD:
                exploded = True
                break

            lin = float(np.linalg.norm(rb.get_linear_velocity()))
            ang = float(np.linalg.norm(rb.get_angular_velocity()))
            if settle_step is None:
                if lin < SETTLE_VEL_THRESHOLD and ang < SETTLE_ANG_THRESHOLD:
                    settle_step = step
            else:
                if lin >= SETTLE_VEL_THRESHOLD or ang >= SETTLE_ANG_THRESHOLD:
                    settle_step = None

            if capturing and step in capture_at:
                gif_frames.append(pr_scene.capture(actor_pos, actor_q))

        final_pos = actor_pos if exploded else np.array(obj_entity.get_pose().p)

        if capturing:
            _save_gif(gif_frames, os.path.join(save_dir, f"{asset_id}.gif"))
            pr_scene.close()
            pr_scene = None

        # ── Contacts ──────────────────────────────────────────────────────────
        if not exploded:
            contacts = scene.get_contacts()
            obj_contacts = [
                c for c in contacts
                if any(b.entity is obj_entity
                       for b in (c.bodies if hasattr(c, "bodies") else []))
            ]
        else:
            obj_contacts = []

        # ── Final metrics ─────────────────────────────────────────────────────
        displacement      = float(np.linalg.norm((final_pos - spawn_pos)[:2]))
        penetration_z     = round(min(float(min_mesh_bottom_z), 0.0), 4)
        flies_away        = displacement > FLY_THRESHOLD
        floor_penetration = penetration_z < FLOOR_PENETRATION_THRESHOLD
        physics_settles   = settle_step is not None and not exploded
        physics_stable    = physics_settles and not flies_away and not floor_penetration

        result["displacement_m"]         = round(displacement, 4)
        result["flies_away"]             = flies_away
        result["penetration_y_m"]        = penetration_z  # Z stored as "y" for CSV compat
        result["floor_penetration"]      = floor_penetration
        result["settle_time_s"]          = round(settle_step / 60.0, 3) if settle_step is not None else None
        result["contact_points_at_rest"] = len(obj_contacts)
        result["physics_settles"]        = physics_settles
        result["physics_stable"]         = physics_stable

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        if _t0 is not None:
            result["wall_time_s"] = round(time.perf_counter() - _t0, 3)
        if pr_scene is not None:
            try:
                pr_scene.close()
            except Exception:
                pass
        if obj_entity is not None:
            try:
                scene.remove_actor(obj_entity)
            except Exception:
                pass

    return result
