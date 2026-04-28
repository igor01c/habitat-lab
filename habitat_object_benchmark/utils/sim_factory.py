#!/usr/bin/env python3

import sys
import os
from pathlib import Path
from typing import Optional, Tuple

import habitat_sim
import magnum as mn

FETCH_URDF = "data/robots/hab_fetch/robots/hab_fetch.urdf"

DEFAULT_SCENE = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"
FLOOR_Y = 0.0       # top surface of the floor plane
FLOOR_THICKNESS = 0.1
FLOOR_HALF_EXTENT = 10.0


def make_sim(
    scene_path: str = DEFAULT_SCENE,
    config_dir: Optional[str] = None,
    with_renderer: bool = False,
    gpu_device_id: int = 0,
    simple_floor: bool = False,
) -> habitat_sim.Simulator:
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "NONE" if simple_floor else scene_path
    backend_cfg.enable_physics = True
    backend_cfg.gpu_device_id = gpu_device_id
    backend_cfg.create_renderer = with_renderer

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = 0.0

    if with_renderer:
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = "color"
        sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        sensor_spec.resolution = [720, 1280]
        sensor_spec.hfov = mn.Deg(60.0)
        sensor_spec.position = mn.Vector3(0, 0, 0)
        agent_cfg.sensor_specifications = [sensor_spec]
    else:
        agent_cfg.sensor_specifications = []

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))

    if simple_floor:
        _add_floor(sim)

    if config_dir is not None:
        sim.get_object_template_manager().load_configs(str(config_dir))

    return sim


def load_fetch_robot(sim: habitat_sim.Simulator, urdf_path: str = FETCH_URDF):
    """Load a FetchRobot into an existing simulator and return it."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "habitat-lab"))
    from habitat.articulated_agents.robots.fetch_robot import FetchRobot
    from omegaconf import OmegaConf

    agent_cfg = OmegaConf.create({"articulated_agent_urdf": urdf_path})
    robot = FetchRobot(agent_cfg, sim, fixed_base=True)
    robot.reconfigure()
    robot.update()
    return robot


def _add_floor(sim: habitat_sim.Simulator) -> None:
    otm = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()
    cube_handle = otm.get_template_handles("cubeSolid")[0]

    def _add_static_box(name, scale, translation):
        tmpl = otm.get_template_by_handle(cube_handle)
        tmpl.scale = scale
        tmpl.shader_type = "pbr"
        tmpl.force_flat_shading = False
        otm.register_template(tmpl, name)
        obj = rom.add_object_by_template_handle(name, light_setup_key=habitat_sim.gfx.NO_LIGHT_KEY)
        obj.translation = translation
        obj.motion_type = habitat_sim.physics.MotionType.STATIC
        return obj

    E = FLOOR_HALF_EXTENT
    T = FLOOR_THICKNESS / 2
    H = 6.0  # room height

    # Floor (physics-enabled, tagged for snap_down)
    _add_static_box("__floor_plane__",
        mn.Vector3(E, T, E), mn.Vector3(0, FLOOR_Y - T, 0))
    # Ceiling
    _add_static_box("__ceiling__",
        mn.Vector3(E, T, E), mn.Vector3(0, FLOOR_Y + H + T, 0))
    # Back wall (-Z)
    _add_static_box("__wall_back__",
        mn.Vector3(E, H / 2, T), mn.Vector3(0, FLOOR_Y + H / 2, -E))
    # Front wall (+Z)
    _add_static_box("__wall_front__",
        mn.Vector3(E, H / 2, T), mn.Vector3(0, FLOOR_Y + H / 2, E))
    # Left wall (-X)
    _add_static_box("__wall_left__",
        mn.Vector3(T, H / 2, E), mn.Vector3(-E, FLOOR_Y + H / 2, 0))
    # Right wall (+X)
    _add_static_box("__wall_right__",
        mn.Vector3(T, H / 2, E), mn.Vector3(E, FLOOR_Y + H / 2, 0))
