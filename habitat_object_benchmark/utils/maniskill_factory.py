#!/usr/bin/env python3
"""SAPIEN 3 / ManiSkill3 scene factory for the object benchmark.

Coordinate convention: SAPIEN uses Z-up, -Z gravity.
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
from scipy.spatial.transform import Rotation

FETCH_URDF = "data/robots/hab_fetch/robots/hab_fetch.urdf"

FLOOR_Z = 0.0
TABLE_HALF_SIZE = [0.6, 0.5, 0.4]   # X, Y, Z half-extents → top at Z = 0.4
TABLE_TOP_Z = TABLE_HALF_SIZE[2] * 2  # 0.8 m


def make_scene(with_renderer: bool = False, timestep: float = 1 / 60) -> sapien.Scene:
    scene = sapien.Scene()
    scene.set_timestep(timestep)
    if with_renderer:
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([1, 1, -1], [1, 1, 1])
    scene.add_ground(altitude=FLOOR_Z)
    return scene


def get_rb(entity: sapien.Entity) -> physx.PhysxRigidDynamicComponent:
    """Return the dynamic rigid-body component of an entity."""
    return entity.find_component_by_type(physx.PhysxRigidDynamicComponent)


def _up_to_sapien_pose(up: list) -> sapien.Pose:
    """Return a Pose whose rotation aligns the mesh's semantic 'up' vector with SAPIEN's world +Z.

    SAPIEN world is Z-up.  Asset configs declare the mesh-local 'up' direction.
    If that vector is already [0,0,1] no rotation is applied.
    """
    world_up = np.array([0.0, 0.0, 1.0])
    asset_up = np.array(up, dtype=float)
    asset_up /= np.linalg.norm(asset_up)

    if np.allclose(asset_up, world_up):
        return sapien.Pose()

    # align_vectors finds the rotation mapping asset_up → world_up
    rot, _ = Rotation.align_vectors([world_up], [asset_up])
    q_xyzw = rot.as_quat()  # [x, y, z, w]
    return sapien.Pose(q=[q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def _glb_node_scale(glb_path: str) -> float:
    """Return the uniform node scale embedded in a GLB file (1.0 if none)."""
    try:
        import trimesh
        scene = trimesh.load(glb_path, process=False)
        if isinstance(scene, trimesh.Scene):
            for node in scene.graph.nodes:
                T, geom = scene.graph[node]
                if geom:
                    s = float(np.linalg.det(T[:3, :3]) ** (1.0 / 3.0))
                    if s > 0:
                        return s
    except Exception:
        pass
    return 1.0


def load_object(
    scene: sapien.Scene,
    config_json_path: str,
    collision_mode: str = "convex_hull",
    scale: float = 1.0,
    name: Optional[str] = None,
) -> sapien.Entity:
    """Load a rigid dynamic object from a .object_config.json into a SAPIEN scene.

    Applies the config's 'up' orientation and 'friction_coefficient' so behaviour
    matches the Habitat/Bullet baseline.
    """
    cfg = json.loads(Path(config_json_path).read_text())
    config_dir = Path(config_json_path).parent
    render_path = str((config_dir / cfg["render_asset"]).resolve())
    collision_path = str((config_dir / cfg["collision_asset"]).resolve())

    if collision_mode == "raw":
        collision_path = render_path
    elif collision_mode == "vhacd":
        vhacd = render_path.replace(".glb", ".vhacd.glb")
        if not os.path.exists(vhacd):
            raise FileNotFoundError(f"VHACD not found: {vhacd}")
        collision_path = vhacd

    # Orientation: rotate mesh so its semantic 'up' aligns with SAPIEN world +Z
    up = cfg.get("up", [0.0, 0.0, 1.0])
    orient_pose = _up_to_sapien_pose(up)

    # Friction from config (static = dynamic = friction_coefficient)
    friction = float(cfg.get("friction_coefficient", 0.5))
    material = physx.PhysxMaterial(
        static_friction=friction,
        dynamic_friction=friction,
        restitution=0.0,
    )

    # SAPIEN applies the GLB-embedded node scale to both visual and collision loaders.
    # Some assets have mismatched node scales (render=1.0, collision=0.4001), causing
    # the visual and collision to appear at different sizes in the scene.
    # Compensate by passing the collision node scale as the visual scale explicitly.
    collision_node_scale = _glb_node_scale(collision_path)
    render_node_scale = _glb_node_scale(render_path)
    if abs(collision_node_scale - render_node_scale) > 1e-4:
        visual_scale = scale * (collision_node_scale / render_node_scale)
    else:
        visual_scale = scale

    builder = scene.create_actor_builder()
    builder.add_visual_from_file(render_path, pose=orient_pose, scale=[visual_scale] * 3)
    builder.add_multiple_convex_collisions_from_file(
        collision_path, pose=orient_pose, scale=[scale] * 3, material=material
    )
    entity_name = name or os.path.basename(config_json_path)
    return builder.build(name=entity_name)


def add_table(scene: sapien.Scene) -> Tuple[sapien.Entity, float]:
    """Add a static box table. Returns (entity, table_top_z)."""
    hx, hy, hz = TABLE_HALF_SIZE
    builder = scene.create_actor_builder()
    mat = sapien.render.RenderMaterial(base_color=[0.6, 0.4, 0.2, 1.0])
    builder.add_box_collision(half_size=[hx, hy, hz])
    builder.add_box_visual(half_size=[hx, hy, hz], material=mat)
    table = builder.build_static(name="__table__")
    table.set_pose(sapien.Pose(p=[0, 0, hz]))  # bottom at Z=0, top at Z=2*hz
    return table, TABLE_TOP_Z


def load_fetch_robot(
    scene: sapien.Scene,
    urdf_path: str = FETCH_URDF,
    setup_drives: bool = False,
) -> sapien.physx.PhysxArticulation:
    """Load Fetch URDF into SAPIEN scene. Returns the Articulation."""
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(urdf_path)
    if setup_drives:
        setup_robot_drives(robot)
    return robot


def setup_robot_drives(
    robot: sapien.physx.PhysxArticulation,
    stiffness: float = 1e3,
    damping: float = 1e2,
) -> None:
    """Set PD drive properties on all active joints so set_qpos targets are held."""
    for j in robot.get_active_joints():
        j.set_drive_properties(stiffness=stiffness, damping=damping)


def snap_down(
    scene: sapien.Scene,
    entity: sapien.Entity,
    support_entities: list,
    max_steps: int = 300,
    vel_threshold: float = 0.005,
) -> bool:
    """Drop entity under gravity until it settles on a support surface or times out."""
    rb = get_rb(entity)
    rb.set_kinematic(False)
    for _ in range(max_steps):
        scene.step()
        v = np.linalg.norm(rb.get_linear_velocity())
        if v < vel_threshold:
            return True
    return False
