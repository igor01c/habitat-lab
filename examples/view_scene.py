#!/usr/bin/env python3

import habitat_sim
import numpy as np
from PIL import Image

SCENE = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"
OUTPUT = "frame.png"

sensor_spec = habitat_sim.CameraSensorSpec()
sensor_spec.uuid = "color"
sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
sensor_spec.resolution = [480, 640]

agent_cfg = habitat_sim.AgentConfiguration()
agent_cfg.sensor_specifications = [sensor_spec]

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = SCENE

sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

obs = sim.reset()
rgb = obs["color"][:, :, :3]  # drop alpha channel
Image.fromarray(rgb).save(OUTPUT)
print(f"Saved {OUTPUT}")

sim.close()
