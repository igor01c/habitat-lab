#!/usr/bin/env python3
"""Step-by-step antipodal grasp sampler debug script.

Steps (run one at a time with --step N):
  1  Load mesh + render PNG
  2  Sample surface points + normals
  3  Ray casting → antipodal candidate pairs
  4  Width filter + antipodal score
  5  Compute 6-DOF grasp pose for best candidate
  6  Finger depth check (knuckle clearance)
  7  Approach clearance (collision-free straight-line path)
  8  IK solve for best surviving candidate (habitat-sim)
  9  Full execution GIF (approach → grasp → lift → release)
  10 Loop top-K candidates, report grasp_success_rate / ik_reachable_rate
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
import trimesh.creation

# ── Fetch gripper constants ───────────────────────────────────────────────────
GRIPPER_OPEN        = 0.04    # each finger max opening (m)
FETCH_FINGER_DEPTH  = 0.08    # finger length from knuckle to tip (m)
FETCH_FINGER_WIDTH  = 0.012   # finger thickness (m)
APPROACH_DIST       = 0.80    # robot base distance from object (m)
PREGRASP_OFFSET     = 0.25    # pre-grasp standoff along approach axis (m)
TOP_K               = 10      # candidates to attempt IK on

# ── Shared helpers ────────────────────────────────────────────────────────────

def _load_mesh(asset_config: Path) -> trimesh.Trimesh:
    import json
    cfg = json.loads(asset_config.read_text())
    base = asset_config.parent
    for key in ("collision_asset", "render_asset"):
        rel = cfg.get(key)
        if rel:
            p = (base / rel).resolve()
            if p.exists():
                loaded = trimesh.load(str(p), force="mesh")
                if isinstance(loaded, trimesh.Scene):
                    loaded = loaded.dump(concatenate=True)
                print(f"Loaded '{key}': {p.name}  "
                      f"vertices={len(loaded.vertices)}  faces={len(loaded.faces)}")
                return loaded
    raise FileNotFoundError(f"No valid mesh found in {asset_config}")


def _hsv_color(hue: float, sat: float = 0.85, val: float = 0.95) -> list:
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, sat, val)
    return [int(r * 255), int(g * 255), int(b * 255), 255]


def _render(scene: trimesh.Scene, out: Path, resolution=(800, 600), show: bool = False):
    png = scene.save_image(resolution=resolution, visible=True)
    out.write_bytes(png)
    print(f"Saved → {out}")
    if show:
        scene.show()


def _mesh_scene(mesh: trimesh.Trimesh) -> trimesh.Scene:
    m = mesh.copy()
    m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=[180, 180, 200, 255])
    return trimesh.Scene([m])


def _add_cylinder(scene, p1, p2, radius, color):
    vec = np.array(p2) - np.array(p1)
    length = np.linalg.norm(vec)
    if length < 1e-8:
        return
    cyl = trimesh.creation.cylinder(radius=radius, height=length)
    z = np.array([0, 0, 1], float)
    ax = np.cross(z, vec / length)
    ax_n = np.linalg.norm(ax)
    if ax_n > 1e-6:
        cyl.apply_transform(trimesh.transformations.rotation_matrix(
            np.arccos(np.clip(np.dot(z, vec / length), -1, 1)), ax / ax_n))
    cyl.apply_translation((np.array(p1) + np.array(p2)) / 2)
    cyl.visual = trimesh.visual.ColorVisuals(mesh=cyl, vertex_colors=color)
    scene.add_geometry(cyl)


def _add_sphere(scene, centre, radius, color):
    dot = trimesh.creation.icosphere(radius=radius)
    dot.apply_translation(centre)
    dot.visual = trimesh.visual.ColorVisuals(mesh=dot, vertex_colors=color)
    scene.add_geometry(dot)


def _add_arrow(scene, origin, direction, length, radius, color):
    """Cylinder + cone arrow from origin along direction."""
    tip = np.array(origin) + np.array(direction) * length
    _add_cylinder(scene, origin, tip, radius, color)
    cone = trimesh.creation.cone(radius=radius * 2.5, height=radius * 5)
    z = np.array([0, 0, 1], float)
    d = np.array(direction, float)
    ax = np.cross(z, d)
    ax_n = np.linalg.norm(ax)
    if ax_n > 1e-6:
        cone.apply_transform(trimesh.transformations.rotation_matrix(
            np.arccos(np.clip(np.dot(z, d), -1, 1)), ax / ax_n))
    cone.apply_translation(tip)
    cone.visual = trimesh.visual.ColorVisuals(mesh=cone, vertex_colors=color)
    scene.add_geometry(cone)


def _sample_candidates(mesh, n_samples, filter_width=True):
    """Sample antipodal pairs and return list of (score, p1, p2, width).

    filter_width=False keeps pairs wider than GRIPPER_OPEN*2 (used for
    visualization so the GIF still shows a grasp attempt on large objects).
    """
    points, face_ids = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_ids]

    ANTIPODAL_THRESH = 0.7   # min cos(angle) for both normals vs squeeze axis (~45°)

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
        n2 = mesh.face_normals[tid2]   # outward normal at exit point

        width = float(np.linalg.norm(p2 - p1))
        if width <= nudge * 2:
            continue
        if filter_width and width > GRIPPER_OPEN * 2:
            continue
        axis = (p2 - p1) / width

        # Both normals must oppose the squeeze axis (true antipodal condition)
        score1 = float(np.dot(-n1, axis))   # n1 should point away from axis
        score2 = float(np.dot( n2, axis))   # n2 should point with axis (exit face)
        if score1 < ANTIPODAL_THRESH or score2 < ANTIPODAL_THRESH:
            continue

        score = (score1 + score2) / 2.0
        candidates.append((score, p1, p2, width))

    candidates.sort(key=lambda x: -x[0])
    return candidates, points, normals


def _grasp_frame(p1, p2):
    """Return (M, x_grip, y_grip, z_grip) for an antipodal pair."""
    M = (np.array(p1) + np.array(p2)) / 2
    x_grip = (np.array(p2) - np.array(p1))
    x_grip /= np.linalg.norm(x_grip)
    world_up = np.array([0.0, 1.0, 0.0])
    z_grip = world_up - np.dot(world_up, x_grip) * x_grip
    z_n = np.linalg.norm(z_grip)
    z_grip = z_grip / z_n if z_n > 1e-6 else np.array([0.0, 0.0, 1.0])
    y_grip = np.cross(z_grip, x_grip)
    y_grip /= np.linalg.norm(y_grip)
    return M, x_grip, y_grip, z_grip


def _finger_depth_ok(mesh, M, z_grip):
    """Check that object extent along z_grip (approach axis) ≤ FETCH_FINGER_DEPTH."""
    proj = mesh.vertices @ z_grip
    m_proj = float(np.array(M) @ z_grip)
    extent = proj.max() - proj.min()
    clearance = FETCH_FINGER_DEPTH - (m_proj - proj.min())
    return extent <= FETCH_FINGER_DEPTH, float(extent), float(clearance)


def _approach_clear(mesh, M, z_grip, n_steps=5):
    """Check straight-line approach from pre-grasp to grasp is collision-free.

    Casts a ray from pre-grasp toward M.  If it hits the mesh before
    reaching M the approach is blocked.  We also check two lateral rays
    offset by ±GRIPPER_OPEN along x_grip (finger tips) using only the
    centre line here (x_grip not available, conservative single-ray check).
    """
    pregrasp = np.array(M) + z_grip * PREGRASP_OFFSET
    approach_dir = np.array(M) - pregrasp
    approach_len = np.linalg.norm(approach_dir)
    if approach_len < 1e-6:
        return True, n_steps, n_steps
    approach_unit = approach_dir / approach_len

    locs, _, _ = mesh.ray.intersects_location(
        [pregrasp], [approach_unit], multiple_hits=False)

    if len(locs) == 0:
        return True, n_steps, n_steps

    hit_dist = np.linalg.norm(locs[0] - pregrasp)
    # The gripper fingers are expected to contact the object surface.
    # Only flag as blocked if an obstacle is encountered in the first half
    # of the approach (well before the contact surface) — this catches
    # overhanging geometry or surrounding obstacles, not the object itself.
    clearance_threshold = approach_len * 0.5
    if hit_dist >= clearance_threshold:
        return True, n_steps, n_steps

    n_clear = int(hit_dist / approach_len * n_steps)
    return False, n_clear, n_steps


# ── Steps 1–5 (unchanged logic, refactored to use helpers) ────────────────────

def step1(asset_config: Path, out: Path, show: bool = False, **_):
    mesh = _load_mesh(asset_config)
    print(f"Bounds: {mesh.bounds}")
    print(f"Watertight: {mesh.is_watertight}")
    _render(_mesh_scene(mesh), out, show=show)


def step2(asset_config: Path, out: Path, n_samples: int = 500, show: bool = False, **_):
    mesh = _load_mesh(asset_config)
    points, face_ids = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_ids]

    scene = _mesh_scene(mesh)
    r = mesh.scale * 0.012
    for p, n in zip(points[::5], normals[::5]):
        _add_sphere(scene, p, r, [255, 80, 80, 255])
        _add_arrow(scene, p, n, mesh.scale * 0.06, mesh.scale * 0.005, [80, 200, 80, 255])

    _render(scene, out, show=show)
    print(f"Sampled {len(points)} points")


def step3(asset_config: Path, out: Path, n_samples: int = 500, show: bool = False, **_):
    mesh = _load_mesh(asset_config)
    points, face_ids = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_ids]

    nudge = mesh.scale * 5e-3
    locs, ray_ids, _ = mesh.ray.intersects_location(
        points + normals * nudge, -normals, multiple_hits=True)
    hits = defaultdict(list)
    for loc, rid in zip(locs, ray_ids):
        hits[rid].append(loc)
    pairs = []
    for rid, hit_locs in hits.items():
        p1 = points[rid]
        dists = [np.linalg.norm(h - p1) for h in hit_locs]
        p2 = hit_locs[int(np.argmax(dists))]
        if np.linalg.norm(p2 - p1) > nudge * 2:
            pairs.append((p1, p2))

    print(f"Found {len(pairs)} antipodal pairs")
    if pairs:
        dists = [np.linalg.norm(b - a) for a, b in pairs]
        print(f"Distance: min={min(dists):.4f}  mean={np.mean(dists):.4f}  max={max(dists):.4f} m")

    scene = _mesh_scene(mesh)
    r = mesh.scale * 0.006
    shown = pairs[::max(1, len(pairs) // 40)]
    for idx, (p1, p2) in enumerate(shown):
        color = _hsv_color(idx / max(len(shown) - 1, 1))
        _add_sphere(scene, p1, r, color)
        _add_sphere(scene, p2, r, color)
        _add_cylinder(scene, p1, p2, r * 0.4, color)

    _render(scene, out, show=show)


def step4(asset_config: Path, out: Path, n_samples: int = 500, show: bool = False, **_):
    mesh = _load_mesh(asset_config)

    # Collect all pairs (no width filter) so we can show valid vs invalid
    all_candidates, _, _ = _sample_candidates(mesh, n_samples, filter_width=False)
    valid   = [(s, p1, p2, w) for s, p1, p2, w in all_candidates if w <= GRIPPER_OPEN * 2]
    invalid = [(s, p1, p2, w) for s, p1, p2, w in all_candidates if w >  GRIPPER_OPEN * 2]

    print(f"Width filter ({GRIPPER_OPEN*2*100:.1f} cm):  "
          f"valid={len(valid)}  invalid={len(invalid)}  total={len(all_candidates)}")
    if valid:
        scores = [c[0] for c in valid]
        print(f"Valid scores:   min={min(scores):.3f}  mean={np.mean(scores):.3f}  max={max(scores):.3f}")
    if invalid:
        scores = [c[0] for c in invalid]
        print(f"Invalid scores: min={min(scores):.3f}  mean={np.mean(scores):.3f}  max={max(scores):.3f}")

    scene = _mesh_scene(mesh)
    r = mesh.scale * 0.010

    # Show a representative subsample of each group
    def _show_pairs(pairs, color, max_shown=30):
        step = max(1, len(pairs) // max_shown)
        for s, p1, p2, w in pairs[::step]:
            _add_sphere(scene, p1, r, color)
            _add_sphere(scene, p2, r, color)
            _add_cylinder(scene, p1, p2, r * 0.4, color)

    _show_pairs(invalid, [220, 50,  50,  200])   # red   — too wide / fails antipodal
    _show_pairs(valid,   [50,  210, 80,  255])   # green — fits in gripper

    _render(scene, out, show=show)


def step5(asset_config: Path, out: Path, n_samples: int = 500, show: bool = False, **_):
    mesh = _load_mesh(asset_config)
    candidates, _, _ = _sample_candidates(mesh, n_samples, filter_width=False)
    if not candidates:
        print("No candidates — cannot compute grasp pose.")
        return
    if candidates[0][3] > GRIPPER_OPEN * 2:
        print(f"  [WARNING] Best candidate width={candidates[0][3]*100:.1f} cm > gripper span={GRIPPER_OPEN*200:.1f} cm — showing geometry only")

    _, p1, p2, _ = candidates[0]
    M, x_grip, y_grip, z_grip = _grasp_frame(p1, p2)
    print(f"Best candidate: score={candidates[0][0]:.3f}  width={np.linalg.norm(p2-p1)*100:.1f} cm")
    print(f"M={M.round(4)}  x={x_grip.round(3)}  y={y_grip.round(3)}  z={z_grip.round(3)}")

    scene = _mesh_scene(mesh)
    al = mesh.scale * 0.15
    r  = mesh.scale * 0.008
    _add_arrow(scene, M, x_grip,  al, r, [255, 60,  60,  255])
    _add_arrow(scene, M, y_grip,  al, r, [60,  255, 60,  255])
    _add_arrow(scene, M, z_grip,  al, r, [60,  60,  255, 255])
    _add_sphere(scene, M, r * 1.5, [255, 255, 60, 255])

    _render(scene, out, show=show)


# ── Step 6 — Finger depth check ───────────────────────────────────────────────

def step6(asset_config: Path, out: Path, n_samples: int = 500, show: bool = False, **_):
    """Filter by knuckle clearance: object extent along approach axis ≤ FETCH_FINGER_DEPTH."""
    mesh = _load_mesh(asset_config)
    candidates, _, _ = _sample_candidates(mesh, n_samples, filter_width=False)
    if not candidates:
        print("No candidates.")
        return

    passed, failed = [], []
    for c in candidates[:TOP_K * 3]:
        _, p1, p2, _ = c
        M, x_grip, y_grip, z_grip = _grasp_frame(p1, p2)
        ok, extent, clearance = _finger_depth_ok(mesh, M, z_grip)
        (passed if ok else failed).append((c, extent, clearance, M, z_grip))

    print(f"Finger depth check  (FETCH_FINGER_DEPTH={FETCH_FINGER_DEPTH*100:.0f} cm):")
    print(f"  Passed: {len(passed)}  Failed: {len(failed)}")
    for c, extent, clearance, M, _ in passed[:3]:
        print(f"  [PASS] score={c[0]:.3f}  extent={extent*100:.1f} cm  clearance={clearance*100:.1f} cm")
    for c, extent, clearance, M, _ in failed[:3]:
        print(f"  [FAIL] score={c[0]:.3f}  extent={extent*100:.1f} cm  clearance={clearance*100:.1f} cm")

    scene = _mesh_scene(mesh)
    r = mesh.scale * 0.012
    al = mesh.scale * 0.18
    for items, color in [(passed, [60, 220, 60, 255]), (failed, [220, 60, 60, 255])]:
        for (_, p1, p2, _), extent, clearance, M, z_grip in items[:5]:
            _add_sphere(scene, p1, r, color)
            _add_sphere(scene, p2, r, color)
            _add_cylinder(scene, p1, p2, r * 0.4, color)
            _add_arrow(scene, M, z_grip, al, r * 0.6, [60, 60, 220, 200])

    _render(scene, out, show=show)


# ── Step 7 — Approach clearance ───────────────────────────────────────────────

def step7(asset_config: Path, out: Path, n_samples: int = 500, show: bool = False, **_):
    """Filter by approach clearance: straight-line path from pre-grasp to grasp."""
    mesh = _load_mesh(asset_config)
    candidates, _, _ = _sample_candidates(mesh, n_samples, filter_width=False)
    if not candidates:
        print("No candidates.")
        return

    # Apply depth filter first, then approach filter
    depth_ok = []
    for c in candidates[:TOP_K * 3]:
        _, p1, p2, _ = c
        M, x_grip, y_grip, z_grip = _grasp_frame(p1, p2)
        ok, _, _ = _finger_depth_ok(mesh, M, z_grip)
        if ok:
            depth_ok.append((c, M, x_grip, y_grip, z_grip))

    passed, failed = [], []
    for c, M, x_grip, y_grip, z_grip in depth_ok:
        clear, clear_steps, total = _approach_clear(mesh, M, z_grip)
        (passed if clear else failed).append((c, M, x_grip, y_grip, z_grip, clear_steps, total))

    print(f"After depth filter: {len(depth_ok)}  After approach filter: {len(passed)}")
    for c, M, _, _, z_grip, cs, tot in passed[:3]:
        print(f"  [CLEAR] score={c[0]:.3f}  clear_steps={cs}/{tot}")
    for c, M, _, _, z_grip, cs, tot in failed[:3]:
        print(f"  [BLOCKED] score={c[0]:.3f}  clear_steps={cs}/{tot}")

    scene = _mesh_scene(mesh)
    r = mesh.scale * 0.012
    al = mesh.scale * 0.20
    for items, color in [(passed, [60, 220, 60, 255]), (failed, [220, 60, 60, 255])]:
        for (_, p1, p2, _), M, x_grip, y_grip, z_grip, cs, tot in items[:5]:
            _add_sphere(scene, p1, r, color)
            _add_sphere(scene, p2, r, color)
            _add_cylinder(scene, p1, p2, r * 0.4, color)
            pregrasp = M + z_grip * PREGRASP_OFFSET
            _add_sphere(scene, pregrasp, r * 0.8, [255, 200, 60, 200])
            _add_cylinder(scene, pregrasp, M, r * 0.3, [255, 200, 60, 150])

    _render(scene, out, show=show)


# ── Step 8 — IK solve in habitat-sim ─────────────────────────────────────────

def step8(asset_config: Path, out: Path, n_samples: int = 500, show: bool = False, **_):
    """Solve IK for surviving candidates; place robot in habitat-sim and render PNG."""
    import os, sys
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    os.environ.setdefault("GLOG_minloglevel", "5")
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "habitat-lab"))

    import magnum as mn
    import habitat_sim
    from habitat_object_benchmark.utils.sim_factory import make_sim, load_fetch_robot, FETCH_URDF
    from habitat_object_benchmark.checks.graspability_check import (
        FetchIKSolver, ARM_INIT, GRIPPER_OPEN as G_OPEN, FLOOR_Y,
    )

    mesh = _load_mesh(asset_config)
    candidates, _, _ = _sample_candidates(mesh, n_samples, filter_width=False)
    if not candidates:
        print("No candidates.")
        return
    if candidates[0][3] > GRIPPER_OPEN * 2:
        print(f"  [WARNING] All candidates wider than gripper span ({GRIPPER_OPEN*200:.1f} cm) — IK solve for geometry only")

    # Apply depth + approach filters
    surviving = []
    for c in candidates[:TOP_K * 3]:
        _, p1, p2, _ = c
        M, x_grip, y_grip, z_grip = _grasp_frame(p1, p2)
        depth_ok, _, _ = _finger_depth_ok(mesh, M, z_grip)
        if not depth_ok:
            continue
        clear, _, _ = _approach_clear(mesh, M, z_grip)
        if clear:
            surviving.append((c, M, x_grip, y_grip, z_grip))

    print(f"Candidates surviving geometry filters: {len(surviving)}")
    if not surviving:
        print("No candidates passed geometry filters — relaxing to top-K by score.")
        surviving = [(c, *_grasp_frame(c[1], c[2])) for c in candidates[:TOP_K]]

    # Set up minimal sim
    sim = make_sim(simple_floor=True, with_renderer=True)
    robot = load_fetch_robot(sim, FETCH_URDF)
    ik_solver = FetchIKSolver(FETCH_URDF)

    from habitat_object_benchmark.checks.graspability_check import GRASP_HEIGHT

    # Place the object at arm-workspace height (same as graspability_check.py).
    # GRASP_HEIGHT is the EE target height; we centre the object there.
    otm = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()
    otm.load_configs(str(asset_config))
    asset_id = asset_config.stem.replace(".object_config", "")
    handles = otm.get_template_handles(asset_id)
    obj = rom.add_object_by_template_handle(handles[0])
    mesh_centre_y = float((mesh.bounds[0][1] + mesh.bounds[1][1]) / 2)
    obj.translation = mn.Vector3(0, GRASP_HEIGHT - mesh_centre_y, 0)
    obj.motion_type = habitat_sim.physics.MotionType.STATIC

    best_result = None
    for rank, (c, M, x_grip, y_grip, z_grip) in enumerate(surviving[:TOP_K]):
        # World-space grasp midpoint (M is in mesh-local coords; offset to world)
        world_M = np.array([float(M[0]), GRASP_HEIGHT + float(M[1]) - mesh_centre_y, float(M[2])])

        # Robot base: fixed approach from +Z (angle=0 → base at +Z, faces -Z)
        # Use several approach angles and pick the best IK result
        best_angle_result = None
        for dist in [0.50, 0.60, 0.70, 0.80]:
            for approach_angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                base_x = float(world_M[0]) + dist * float(np.sin(approach_angle))
                base_z = float(world_M[2]) + dist * float(np.cos(approach_angle))
                base_xz_arr = np.array([base_x, base_z])
                base_yaw = float(approach_angle + np.pi)
                angles, err = ik_solver.solve(base_xz_arr, base_yaw, world_M)
                if angles is not None:
                    best_angle_result = (base_x, base_z, base_yaw, angles, world_M, err)
                    break
                if best_angle_result is None or err < best_angle_result[5]:
                    best_angle_result = (base_x, base_z, base_yaw, angles, world_M, err)
            if best_angle_result and best_angle_result[3] is not None:
                break

        base_x, base_z, base_yaw, arm_angles, target, ik_err = best_angle_result
        ok = arm_angles is not None
        print(f"  Candidate {rank}: IK={'OK' if ok else 'FAIL'}  err={ik_err:.4f}  target_y={target[1]:.3f}")
        if ok and best_result is None:
            best_result = (base_x, base_z, base_yaw, arm_angles, target)

    if best_result is None:
        print("No candidate had IK success — using best-effort (lowest error).")
        ik_solver.close()
        sim.close()
        return

    base_x, base_z, base_yaw, arm_angles, target = best_result
    hab_yaw = base_yaw - ik_solver._arm_forward_angle

    print(f"\n── Step 8 debug ──────────────────────────────────────")
    print(f"  IK target (world_M)    : {np.array(target).round(4)}")
    print(f"  Object translation     : {[round(float(obj.translation[i]),4) for i in range(3)]}")
    print(f"  Robot base             : ({base_x:.4f}, {FLOOR_Y:.4f}, {base_z:.4f})")
    print(f"  base_yaw={base_yaw:.4f}  hab_yaw={hab_yaw:.4f}  arm_forward_angle={ik_solver._arm_forward_angle:.4f}")
    print(f"  arm_angles             : {np.array(arm_angles).round(4)}")

    robot.base_pos = mn.Vector3(base_x, FLOOR_Y, base_z)
    robot.base_rot = hab_yaw
    robot.arm_joint_pos = arm_angles
    robot.gripper_joint_pos = np.array([G_OPEN, G_OPEN])
    robot.update()

    # Query actual EE position in habitat-sim after applying joint angles
    ee_tf  = robot.ee_transform(0)
    ee_pos = np.array(ee_tf.translation)
    print(f"  EE pos (habitat-sim)   : {ee_pos.round(4)}")
    print(f"  EE→target distance     : {np.linalg.norm(ee_pos - target):.4f} m")

    # Verify PyBullet FK matches habitat-sim EE
    import pybullet as _pb
    _pb.resetBasePositionAndOrientation(
        ik_solver._robot, (base_x, 0.0, base_z),
        _pb.getQuaternionFromEuler([0, hab_yaw, 0]),
        physicsClientId=ik_solver._client)
    from habitat_object_benchmark.checks.graspability_check import PB_ARM_JOINTS, PB_EE_LINK, PB_TORSO_JOINT, TORSO_HEIGHT
    _pb.resetJointState(ik_solver._robot, PB_TORSO_JOINT, TORSO_HEIGHT, physicsClientId=ik_solver._client)
    for k, ji in enumerate(PB_ARM_JOINTS):
        _pb.resetJointState(ik_solver._robot, ji, float(arm_angles[k]), physicsClientId=ik_solver._client)
    pb_ee = np.array(_pb.getLinkState(ik_solver._robot, PB_EE_LINK, physicsClientId=ik_solver._client)[4])
    print(f"  EE pos (PyBullet FK)   : {pb_ee.round(4)}")
    print(f"  PyBullet FK→target     : {np.linalg.norm(pb_ee - target):.4f} m")
    print(f"  PyBullet FK vs hab-sim : {np.linalg.norm(pb_ee - ee_pos):.4f} m")
    print(f"──────────────────────────────────────────────────────\n")

    # Place a bright red marker sphere at the IK target so it's visible in the render
    otm2 = sim.get_object_template_manager()
    rom2 = sim.get_rigid_object_manager()
    marker_handle = otm2.get_template_handles("cubeSolid")[0]
    tmpl_m = otm2.get_template_by_handle(marker_handle)
    tmpl_m.scale = mn.Vector3(0.03, 0.03, 0.03)
    otm2.register_template(tmpl_m, "__ik_target__")
    marker = rom2.add_object_by_template_handle("__ik_target__")
    marker.translation = mn.Vector3(float(target[0]), float(target[1]), float(target[2]))
    marker.motion_type = habitat_sim.physics.MotionType.STATIC

    # Camera: side view
    from habitat_object_benchmark.checks.graspability_check import _setup_camera
    cam_height = float(obj.translation[1]) + 0.3
    _setup_camera(sim,
                  asset_pos=np.array([float(obj.translation[0]), 0, float(obj.translation[2])]),
                  robot_base=np.array([base_x, FLOOR_Y, base_z]),
                  cam_height=cam_height,
                  lookat_y=float(obj.translation[1]),
                  view_angle_deg=45.0)

    obs = sim.get_sensor_observations()
    from PIL import Image
    img = Image.fromarray(obs["color"][:, :, :3])
    img.save(str(out))
    print(f"Saved → {out}")

    ik_solver.close()
    sim.close()


# ── Step 9 — Full execution GIF ───────────────────────────────────────────────

def step9(asset_config: Path, out: Path, n_samples: int = 500, show: bool = False, **_):
    """Full grasp GIF. Shows an interactive 3D geometry view first, then runs the sim."""
    import os, sys
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    os.environ.setdefault("GLOG_minloglevel", "5")
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "habitat-lab"))

    from habitat_object_benchmark.checks.graspability_check import _capture_static
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    asset_id = asset_config.stem.replace(".object_config", "")
    gif_path = out.with_suffix(".gif")
    os.makedirs(str(gif_path.parent), exist_ok=True)

    # Run sim and collect geometry data
    geo = _capture_static(str(asset_config), asset_id, str(gif_path),
                          return_geometry=True)
    print(f"GIF → {gif_path}")

    if geo is None:
        return

    # ── Interactive 3D view ───────────────────────────────────────────────────
    bb_min      = geo["bb_min"]
    bb_max      = geo["bb_max"]
    wp_grasp    = geo["wp_grasp"]
    wp_pregrasp = geo["wp_pregrasp"]
    wp_lift     = geo["wp_lift"]
    robot_base  = geo["robot_base"]
    table_top_y = geo["table_top_y"]

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    # AABB wireframe (X, Z, Y axes → plot as X, Z, height=Y)
    corners = np.array([[x, y, z]
                        for x in (bb_min[0], bb_max[0])
                        for y in (bb_min[1], bb_max[1])
                        for z in (bb_min[2], bb_max[2])])
    edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    for i, j in edges:
        a, b = corners[i], corners[j]
        ax.plot([a[0], b[0]], [a[2], b[2]], [a[1], b[1]], "b-", alpha=0.5, linewidth=1.5)

    # Table surface
    ty = table_top_y
    E  = 0.4
    ax.plot_surface(
        np.array([[-E, E], [-E, E]]),
        np.array([[-E, -E], [E, E]]),
        np.full((2, 2), ty),
        alpha=0.15, color="brown"
    )

    def _pt(p, color, marker, label, size=100):
        ax.scatter(p[0], p[2], p[1], c=color, marker=marker, s=size,
                   zorder=5, label=label)

    _pt(wp_grasp,    "red",    "o", "wp_grasp",    120)
    _pt(wp_pregrasp, "orange", "o", "wp_pregrasp", 120)
    _pt(wp_lift,     "green",  "o", "wp_lift",      80)
    _pt(robot_base,  "black",  "^", "robot base",   80)
    _pt((bb_min + bb_max) / 2, "blue", "x", "obj centre", 60)

    # Arrows: pregrasp → grasp → lift
    for a, b, col in [(wp_pregrasp, wp_grasp, "red"), (wp_grasp, wp_lift, "green")]:
        d = b - a
        ax.quiver(a[0], a[2], a[1], d[0], d[2], d[1],
                  color=col, arrow_length_ratio=0.2, linewidth=2)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_zlabel("Y / height (m)")
    ax.set_title(f"Grasp geometry — {asset_id}\nClose window to continue")
    ax.legend(loc="upper left", fontsize=8)
    mid = (bb_min + bb_max) / 2
    ax.set_xlim(mid[0] - 0.7, mid[0] + 0.7)
    ax.set_ylim(mid[2] - 0.7, mid[2] + 0.7)
    ax.set_zlim(0.0, 1.6)
    ax.view_init(elev=20, azim=-55)
    plt.tight_layout()
    plt.show()


# ── Step 10 — Loop top-K, report metrics ─────────────────────────────────────

def step10(asset_config: Path, out: Path, n_samples: int = 500, show: bool = False, **_):
    """Run full antipodal pipeline on top-K candidates, report success rates."""
    import os, sys
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    os.environ.setdefault("GLOG_minloglevel", "5")
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "habitat-lab"))

    import magnum as mn
    import habitat_sim
    from habitat_object_benchmark.utils.sim_factory import make_sim, load_fetch_robot, FETCH_URDF
    from habitat_object_benchmark.checks.graspability_check import (
        FetchIKSolver, ARM_INIT, GRIPPER_OPEN as G_OPEN,
        GRIPPER_CLOSED, FLOOR_Y, APPROACH_STEPS, CLOSE_STEPS,
        HOLD_STEPS, RELEASE_STEPS, DRIFT_THRESHOLD,
    )

    mesh = _load_mesh(asset_config)
    candidates, _, _ = _sample_candidates(mesh, n_samples)
    print(f"Total antipodal candidates: {len(candidates)}")

    surviving = []
    for c in candidates:
        _, p1, p2, _ = c
        M, x_grip, y_grip, z_grip = _grasp_frame(p1, p2)
        if not _finger_depth_ok(mesh, M, z_grip)[0]:
            continue
        if _approach_clear(mesh, M, z_grip)[0]:
            surviving.append((c, M, x_grip, y_grip, z_grip))

    print(f"After geometry filters: {len(surviving)}")

    from habitat_object_benchmark.checks.graspability_check import GRASP_HEIGHT

    sim = make_sim(simple_floor=True)
    robot = load_fetch_robot(sim, FETCH_URDF)
    ik_solver = FetchIKSolver(FETCH_URDF)
    otm = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()
    otm.load_configs(str(asset_config))
    asset_id = asset_config.stem.replace(".object_config", "")
    handles = otm.get_template_handles(asset_id)
    mesh_centre_y = float((mesh.bounds[0][1] + mesh.bounds[1][1]) / 2)

    ik_ok = 0
    grasp_ok = 0
    tried = 0

    for rank, (c, M, x_grip, y_grip, z_grip) in enumerate(surviving[:TOP_K]):
        tried += 1
        # World-space grasp target at arm workspace height
        world_M = np.array([float(M[0]), GRASP_HEIGHT + float(M[1]) - mesh_centre_y, float(M[2])])

        # Try 8 approach angles, pick best IK result
        best_angle_result = None
        for dist in [0.50, 0.60, 0.70, 0.80]:
            for approach_angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                base_x = float(world_M[0]) + dist * float(np.sin(approach_angle))
                base_z = float(world_M[2]) + dist * float(np.cos(approach_angle))
                base_xz_arr = np.array([base_x, base_z])
                base_yaw = float(approach_angle + np.pi)
                angles, err = ik_solver.solve(base_xz_arr, base_yaw, world_M)
                if angles is not None:
                    best_angle_result = (base_x, base_z, base_yaw, angles, world_M, err)
                    break
                if best_angle_result is None or err < best_angle_result[5]:
                    best_angle_result = (base_x, base_z, base_yaw, angles, world_M, err)
            if best_angle_result and best_angle_result[3] is not None:
                break
        base_x, base_z, base_yaw, arm_angles, target, ik_err = best_angle_result

        obj = rom.add_object_by_template_handle(handles[0])
        obj.translation = mn.Vector3(0, GRASP_HEIGHT - mesh_centre_y, 0)
        obj.motion_type = habitat_sim.physics.MotionType.STATIC

        arm_angles, ik_err = ik_solver.solve(np.array([base_x, base_z]), base_yaw, target)
        if arm_angles is None:
            print(f"  [{rank}] IK FAIL  err={ik_err:.4f}")
            rom.remove_object_by_id(obj.object_id)
            continue
        ik_ok += 1

        # Geometric success criterion (same as graspability_check.py):
        # object width along gripper's squeeze axis must fit within open span.
        hab_yaw = base_yaw - ik_solver._arm_forward_angle
        robot.base_pos = mn.Vector3(base_x, FLOOR_Y, base_z)
        robot.base_rot = hab_yaw
        robot.arm_joint_pos = arm_angles
        robot.gripper_joint_pos = np.array([G_OPEN, G_OPEN])
        robot.update()

        # Squeeze axis = x_grip (antipodal direction, known from geometry).
        # Project mesh vertices onto x_grip to get object width at contact.
        verts = mesh.vertices  # mesh-local coords
        proj = verts @ x_grip
        obj_width = float(proj.max() - proj.min())
        width_ok = obj_width <= G_OPEN * 2
        success = width_ok
        if success:
            grasp_ok += 1
        print(f"  [{rank}] IK OK  obj_width={obj_width*100:.1f}cm  limit={G_OPEN*200:.1f}cm  {'GRASP OK' if success else 'TOO WIDE'}")

        rom.remove_object_by_id(obj.object_id)
        robot.base_pos = mn.Vector3(50, FLOOR_Y, 0)
        robot.arm_joint_pos = ARM_INIT.copy()
        robot.update()

    ik_solver.close()
    sim.close()

    print(f"\n── Results ─────────────────────────────")
    print(f"  Tried:             {tried}")
    print(f"  IK reachable:      {ik_ok}/{tried}  ({ik_ok/max(tried,1)*100:.0f}%)")
    print(f"  Grasp success:     {grasp_ok}/{tried}  ({grasp_ok/max(tried,1)*100:.0f}%)")
    print(f"  ik_reachable_rate: {ik_ok/max(tried,1):.3f}")
    print(f"  grasp_success_rate:{grasp_ok/max(tried,1):.3f}")


# ── Display helper ────────────────────────────────────────────────────────────

def _display_blocking(path: Path, title: str) -> None:
    """Show a PNG or GIF in a blocking matplotlib window.  Close to continue."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from PIL import Image

    img = Image.open(path)

    if getattr(img, "is_animated", False) or path.suffix.lower() == ".gif":
        frames = []
        try:
            while True:
                frames.append(img.copy().convert("RGBA"))
                img.seek(img.tell() + 1)
        except EOFError:
            pass

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.canvas.manager.set_window_title(title)
        ax.axis("off")
        im = ax.imshow(frames[0])

        def _update(i):
            im.set_data(frames[i % len(frames)])
            return (im,)

        duration_ms = img.info.get("duration", 80)
        ani = animation.FuncAnimation(
            fig, _update, frames=len(frames),
            interval=duration_ms, blit=True, repeat=True,
        )
        plt.tight_layout()
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.canvas.manager.set_window_title(title)
        ax.imshow(np.array(img))
        ax.axis("off")
        plt.tight_layout()
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

STEPS = {1: step1, 2: step2, 3: step3, 4: step4, 5: step5,
         6: step6, 7: step7, 8: step8, 9: step9, 10: step10}

STEP_DESCRIPTIONS = {
    1:  "Load mesh + render",
    2:  "Surface points + normals",
    3:  "Ray casting → antipodal pairs",
    4:  "Width filter + antipodal score",
    5:  "6-DOF grasp frame (best pair)",
    6:  "Finger depth check",
    7:  "Approach clearance",
    8:  "IK solve (habitat-sim)",
    9:  "Full execution GIF",
    10: "Metrics loop (no image)",
}

def main():
    parser = argparse.ArgumentParser(description="Antipodal grasp sampler — debug steps")
    parser.add_argument("--asset-config", required=True, type=Path)
    parser.add_argument("--step", type=int, choices=list(STEPS), default=1,
                        help="Which step to run (ignored when --all is set)")
    parser.add_argument("--all", action="store_true",
                        help="Run all steps sequentially, showing each result "
                             "in a blocking window (close to advance)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (auto-set per step when --all is used)")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--show", action="store_true",
                        help="Open interactive 3D viewer after saving PNG (steps 1–7)")
    args = parser.parse_args()

    import tempfile, os
    tmp_dir = Path(tempfile.mkdtemp(prefix="antipodal_"))

    if args.all:
        steps_to_run = sorted(STEPS.keys())
    else:
        steps_to_run = [args.step]

    for s in steps_to_run:
        fn = STEPS[s]
        desc = STEP_DESCRIPTIONS[s]
        print(f"\n{'='*60}")
        print(f"  Step {s}: {desc}")
        print(f"{'='*60}")

        if args.all:
            ext = ".gif" if s == 9 else ".png"
            out = tmp_dir / f"step{s:02d}{ext}"
        else:
            out = args.out or Path(f"/tmp/antipodal_step{s}.png")

        # Steps 1-7 have a trimesh 3D viewer; steps 8-9 produce hab-sim images.
        use_show = args.show or (args.all and s <= 7)
        fn(asset_config=args.asset_config, out=out,
           n_samples=args.n_samples, show=use_show)

        if not use_show and s != 10 and out.exists():
            _display_blocking(out, f"Step {s} — {desc}  (close to continue)")


if __name__ == "__main__":
    main()
