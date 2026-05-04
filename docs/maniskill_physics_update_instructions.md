# ManiSkill Physics Check — Update Instructions

## Context

The Bullet physics check (`physics_check.py`) was redesigned after the ManiSkill port was first written. The ManiSkill version (`physics_check_ms.py`) still uses the **old metric schema** (snap-based, relative offsets). This document instructs you to bring `physics_check_ms.py` and `run_physics_check_ms.py` in line with the current Bullet design.

---

## What Changed in the Bullet Version (the authoritative reference)

### Experiment structure

**Old (ManiSkill current):**
- Phase 1: snap the object onto the table via a raycast/gravity settle → record `snap_pos`
- Phase 2: raise object 0.2 m above `snap_pos`, drop again
- All offset metrics (`final_y_offset_m`, `min_y_offset_m`) are relative to `snap_pos`

**New (Bullet current, target for ManiSkill):**
- No snap phase. Single drop from a consistent height: AABB bottom 10 cm above the floor (`SPAWN_CLEARANCE = 0.10`)
- Floor at Z=0 is the only reference
- Vertical metrics are absolute penetration below the floor, measured via collision mesh vertices

### Metric schema — OLD vs NEW

| Old field (remove) | New field (add) | Notes |
|---|---|---|
| `final_y_offset_m` | — | removed |
| `sinks_permanently` | — | removed |
| `min_y_offset_m` | — | removed |
| `sinks_below_floor` | — | removed |
| — | `penetration_y_m` | `min(min_collision_vertex_Z, 0.0)` — negative = below floor |
| — | `floor_penetration` | `penetration_y_m < -0.05` |
| `physics_settles` | `physics_settles` | unchanged: velocity stayed below threshold |
| — | `physics_stable` | `physics_settles and not flies_away and not floor_penetration` |
| `displacement_m` | `displacement_m` | XY distance from spawn XY to final XY (unchanged) |
| `flies_away` | `flies_away` | `displacement_m > 1.5` (unchanged) |
| `settle_time_s` | `settle_time_s` | unchanged |
| — | `wall_time_s` | wall-clock seconds from instantiation to end of simulation (see below) |
| `contact_points_at_rest` | `contact_points_at_rest` | unchanged |
| `error` | `error` | unchanged |

---

## Changes Required

### 1. `checks/physics_check_ms.py`

#### Remove snap-phase entirely

Delete everything related to the two-phase drop:
- The `SPAWN_GAP`, `FINAL_SINK_THRESHOLD`, `MIN_SINK_THRESHOLD` constants
- The first `snap_down()` call and `snap_pos` recording
- The second `obj_entity.set_pose(...)` raise-above-snap-pos
- The `min_z` tracking relative to `spawn_z`
- The `final_z_offset` and `min_z_offset` computations

#### New spawn logic

##### Critical: AABB normalization (do not skip)

A raw trimesh load of the collision GLB may give a different AABB than what SAPIEN/PhysX uses internally. This was observed on Objaverse assets exported from Sketchfab: those GLBs contain a root node with a 90° X rotation (Z-up → Y-up coordinate fix) that trimesh and SAPIEN's loader apply differently, producing a ~7% scale difference in the vertical axis and translation offsets of 3–46 cm. Without correction, the spawned object is mis-placed and penetration tracking is wrong.

The fix is to **normalize the trimesh-loaded vertices so that their AABB exactly matches the simulator's reported AABB** (SAPIEN equivalent: `actor.get_collision_shapes()` AABB or the actor's axis-aligned bounding box at identity pose). Load the vertices, compute their raw AABB, then rescale each axis to match what the simulator reports. See how this is done in the Bullet version in `_load_collision_vertices` + the normalization block in `physics_check.py::run()`.

In SAPIEN, after instantiating the actor at identity pose, you can get its AABB via:

```python
# At identity pose (before calling set_pose with spawn_z):
aabb = actor.get_global_axis_aligned_bounding_box()  # sapien.physx AABB, or equivalent
hab_min = np.array([aabb.lower[0], aabb.lower[1], aabb.lower[2]])
hab_max = np.array([aabb.upper[0], aabb.upper[1], aabb.upper[2]])
```

Then after loading `local_verts` from trimesh (centered at their own AABB center), apply:

```python
raw_min = local_verts.min(axis=0)
raw_max = local_verts.max(axis=0)
raw_half = (raw_max - raw_min) / 2.0
hab_half = (hab_max - hab_min) / 2.0
hab_center = (hab_max + hab_min) / 2.0
scale = np.where(raw_half > 1e-8, hab_half / raw_half, 1.0)
local_verts = local_verts * scale + hab_center
```

This replaces the simple `aabb_center` subtraction. After this, `local_verts.min(axis=0)` exactly equals `hab_min` (the simulator's bottom), eliminating spawn misalignment for all datasets.

---

```python
SPAWN_CLEARANCE = 0.10   # 10 cm above floor

# After loading object, before simulation:
# Compute AABB of collision shape in actor-local frame
import trimesh
cfg = json.loads(Path(config_json_path).read_text())
config_dir = Path(config_json_path).parent
collision_path = str((config_dir / cfg["collision_asset"]).resolve())
col_scene = trimesh.load(collision_path, process=False)
# Collect vertices applying FULL node transform (including translation).
# For VHACD assets each hull sits at a different offset — translation must be included.
parts = []
if isinstance(col_scene, trimesh.Scene):
    for node in col_scene.graph.nodes_geometry:
        T, geom = col_scene.graph[node]
        v = trimesh.transformations.transform_points(col_scene.geometry[geom].vertices, T)
        parts.append(v)
else:
    parts.append(np.array(col_scene.vertices))
local_verts = np.vstack(parts)
# Apply orient_pose rotation (same as applied to collision shape)
from scipy.spatial.transform import Rotation as R_
up = cfg.get("up", [0.0, 0.0, 1.0])
orient_pose = _up_to_sapien_pose(up)   # already imported from maniskill_factory
q = orient_pose.q  # [w,x,y,z]
rot = R_.from_quat([q[1], q[2], q[3], q[0]])
local_verts = (rot.as_matrix() @ local_verts.T).T
# Subtract raw AABB center (initial centering)
raw_aabb_center = (local_verts.max(axis=0) + local_verts.min(axis=0)) / 2.0
local_verts -= raw_aabb_center

# Normalize to simulator AABB (see "Critical: AABB normalization" note above).
# Query the actor's AABB at identity pose here and apply the scale+recenter transform.
# ... (see normalization block above) ...

aabb_min_z = float(local_verts[:, 2].min())   # Z is up in SAPIEN

spawn_z = -aabb_min_z + SPAWN_CLEARANCE
obj_entity.set_pose(sapien.Pose(p=[0.0, 0.0, spawn_z]))
rb.set_kinematic(False)
spawn_pos = obj_entity.get_pose().p.copy()
```

#### Per-step tracking

Replace the `min_z` tracking (relative to snap) with absolute floor penetration via mesh vertices:

```python
min_mesh_bottom_z = float(spawn_z)   # starts high, we track the minimum

for step in range(PHYSICS_STEPS):
    scene.step()
    pose = obj_entity.get_pose()
    actor_pos = np.array(pose.p)
    actor_q   = np.array(pose.q)  # [w,x,y,z]

    # Transform local verts to world to find lowest Z
    R_world = R_.from_quat([actor_q[1], actor_q[2], actor_q[3], actor_q[0]]).as_matrix()
    world_verts_z = (R_world @ local_verts.T)[2] + actor_pos[2]
    bottom_z = float(world_verts_z.min())
    if bottom_z < min_mesh_bottom_z:
        min_mesh_bottom_z = bottom_z

    speed = float(np.linalg.norm(rb.get_linear_velocity()))
    # settle tracking unchanged ...
```

#### Updated result dict

```python
result = {
    "collision_mode": collision_mode,
    "physics_settles": False,
    "physics_stable": False,
    "displacement_m": None,
    "flies_away": None,
    "penetration_y_m": None,
    "floor_penetration": None,
    "settle_time_s": None,
    "contact_points_at_rest": None,
    "error": None,
}
```

#### Updated metrics computation (end of run)

```python
final_pos = obj_entity.get_pose().p.copy()
displacement = float(np.linalg.norm((final_pos - spawn_pos)[:2]))   # XY
penetration_z = round(min(min_mesh_bottom_z, 0.0), 4)

flies_away        = displacement > FLY_THRESHOLD          # 1.5 m
floor_penetration = penetration_z < -0.05
physics_settles   = settle_step is not None
physics_stable    = physics_settles and not flies_away and not floor_penetration

result["displacement_m"]         = round(displacement, 4)
result["flies_away"]             = flies_away
result["penetration_y_m"]        = penetration_z          # named y for CSV compat with Bullet
result["floor_penetration"]      = floor_penetration
result["settle_time_s"]          = round(settle_step / 60.0, 3) if settle_step is not None else None
result["contact_points_at_rest"] = len(obj_contacts)
result["physics_settles"]        = physics_settles
result["physics_stable"]         = physics_stable
```

#### Timing metric — `wall_time_s`

Add a `wall_time_s` field that records the wall-clock seconds from object instantiation to the end of the simulation loop. Start the timer immediately after the actor is created (before mesh loading and AABB normalization) and stop it at the end of the simulation, in a `finally` block so it is always written even on exception.

The timer covers: collision mesh loading (trimesh), AABB normalization, PhysX hull building, and the full 600-step simulation. It does **not** include GIF/image rendering — do not start the timer before and stop it after `_PyRenderScene` calls. All paper runs should be done without image saving so that `wall_time_s` reflects pure simulation cost and is comparable across datasets.

```python
import time
_t0 = time.perf_counter()
# ... mesh loading, spawn, simulation loop ...
# In finally block:
result["wall_time_s"] = round(time.perf_counter() - _t0, 3)
```

Add `"wall_time_s": None` to the initial result dict, and add `"wall_time_s"` to `FIELDNAMES` in `run_physics_check_ms.py` between `settle_time_s` and `contact_points_at_rest`.

#### GIF / pyrender

The `_PyRenderScene` should now frame the single drop from `spawn_z` to the floor (Z=0), not a snap-based drop. The table is no longer used — replace it with a floor plane at Z=0:

```python
# Remove add_table() call entirely.
# In _PyRenderScene: replace the brown box table with a flat floor surface at Z=0.
# Camera should look at the drop midpoint between spawn_z and Z=0.
```

---

### 2. `scripts/run_physics_check_ms.py`

Update `FIELDNAMES`:

```python
FIELDNAMES = [
    "asset_id",
    "collision_mode",
    "physics_settles",
    "physics_stable",
    "displacement_m",
    "flies_away",
    "penetration_y_m",
    "floor_penetration",
    "contact_points_at_rest",
    "settle_time_s",
    "error",
]
```

---

## Constants to Keep / Change

```python
# Keep
PHYSICS_STEPS = 600          # 10 s at 60 Hz
FLY_THRESHOLD = 1.5          # m
SETTLE_VEL_THRESHOLD = 0.01  # m/s (linear)

# Add
SPAWN_CLEARANCE = 0.10       # m — AABB bottom above floor at spawn
FLOOR_PENETRATION_THRESHOLD = -0.05  # m

# Remove
SPAWN_GAP = 0.2
FINAL_SINK_THRESHOLD = -0.10
MIN_SINK_THRESHOLD = -0.15
```

---

## What Does NOT Change

- `maniskill_factory.py` — no changes needed
- `load_object()`, `_up_to_sapien_pose()`, `_glb_node_scale()` — unchanged
- `snap_down()` — no longer called in physics check (still used in graspability)
- `graspability_check_ms.py` — unchanged
- `run_graspability_check_ms.py` — unchanged
- `build_results_json.py` — already reads with `.get()`, handles new schema
- `inspector.html` — already handles `penetration_y_m`/`floor_penetration` via `?.` fallbacks
