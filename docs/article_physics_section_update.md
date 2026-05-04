# Article Update — Physics Stability Experiment

## Summary of Changes Since Last Description

The physics stability experiment was substantially redesigned. The core change is the removal of a two-phase protocol in favour of a single, consistent drop from a fixed height. This affects how the experiment is described, what metrics are reported, and what conclusions can be drawn.

---

## What Changed

### 1. Experiment Protocol — Remove Two-Phase Drop

**Old description (do not use):**
The experiment had two phases. First, a "snap" phase where the object was dropped and its resting position was recorded as a reference point. Second, the object was raised 20 cm above that reference point and dropped again. All vertical metrics were computed relative to the snap reference position.

**New description (use this):**
The experiment uses a single drop protocol. Each object's collision mesh AABB is computed, and the object is spawned with its lowest collision vertex 10 cm above the floor. It is then released under gravity and simulated for 10 seconds (600 steps at 60 Hz). The floor at Y=0 is the sole reference for all metrics. This ensures every asset starts from a geometrically consistent initial condition, independent of its shape.

---

### 2. Metrics — Complete Replacement of Vertical Offset Fields

**Remove all mention of these fields:**

- `final_y_offset_m` — final vertical offset from snap reference
- `min_y_offset_m` — minimum vertical offset from snap reference
- `sinks_permanently` — final offset below −10 cm threshold
- `sinks_below_floor` — minimum offset below −15 cm threshold

**These fields were relative to the snap reference position, which was not physically meaningful** (a tall box that tips during the snap phase would produce a large negative reference offset, corrupting all subsequent measurements).

**Replace with:**

- `penetration_y_m` — the minimum world Y reached by any vertex of the collision mesh over the full simulation, clamped to ≤ 0 (i.e., zero if the object never goes below the floor, negative if it penetrates). Computed as `min(min_vertex_Y, 0)`.
- `floor_penetration` — boolean: `True` if `penetration_y_m < −0.05 m`. This flags objects whose collision mesh passes more than 5 cm below the floor plane, indicating a degenerate collision mesh or excessive tunnelling.

---

### 3. Primary Pass Criterion — Updated Definition

**Old:** `physics_settles` was the primary criterion, defined as: no large displacement AND no excessive sinking AND no excessive final offset from snap reference.

**New:**

- `physics_settles` — the object's linear velocity fell and remained below 0.01 m/s at some point during simulation. This is the settling criterion.
- `physics_stable` (new field) — composite: `physics_settles AND NOT flies_away AND NOT floor_penetration`. This is the quality criterion used for asset filtering.

For reporting, **use `physics_stable` as the primary pass rate**, not `physics_settles`. An object can settle (stop moving) at the wrong place or after penetrating the floor — `physics_stable` catches both cases.

---

### 4. Displacement Metric — Reference Changed

**Old:** `displacement_m` was XZ distance from the snap reference position to the final position.

**New:** `displacement_m` is the XZ (horizontal) distance from the spawn position (directly above where the object was placed at the start) to the final resting position. Since all objects spawn at XZ = (0, 0), this is simply the absolute horizontal displacement from origin.

The threshold for `flies_away` remains 1.5 m.

---

### 5. Simulation Duration

Duration increased from 5 seconds (300 steps) to **10 seconds (600 steps)** to allow slower or heavier objects more time to come to rest. The settling velocity threshold remains 0.01 m/s linear, with an additional 0.1 rad/s angular velocity threshold.

---

### 6. New Timing Metric — `wall_time_s` and Simulation Readiness

A new field `wall_time_s` records the wall-clock time (in seconds) taken to complete the simulation for each asset. The timer starts at object instantiation and covers: collision mesh loading (trimesh), AABB normalization, Habitat template registration and hull building, and the full 600-step simulation. It does **not** include GIF rendering (only relevant when `--save-images` is used, which should not be used for paper runs). All paper results were produced without image saving, so `wall_time_s` reflects pure simulation cost and is comparable across datasets.

**Why it matters:** An asset that takes more than 60 seconds to simulate is not practically usable in large-scale pipelines, even if it passes the stability checks. This captures a different dimension of asset quality — computational tractability — beyond geometric stability.

**Derived statistic for reporting:**

- **Within-60s rate** — the fraction of assets whose simulation completed within 60 seconds. Use this as a "simulation readiness" metric: a dataset where only 70% of assets simulate within 60 seconds is not ready for large-scale use, regardless of how many pass the stability check.

**Suggested reporting:** Report `wall_time_s` mean, median, and max per collision mode, alongside the within-60s rate. For cross-dataset comparison, the within-60s rate directly quantifies simulation tractability.

Note: assets that timed out at the batch level (exceeded the per-asset timeout, default 20 s) have `wall_time_s = None` and are excluded from timing statistics. They are counted as errors in the error rate but not in the within-60s denominator.

---

### 7. What Remains Unchanged

- `flies_away` — boolean, `displacement_m > 1.5 m`
- `contact_points_at_rest` — number of contact points at the final simulation step
- `settle_time_s` — time at which the object first reached and sustained the velocity threshold
- `collision_mode` — convex hull vs. V-HACD decomposition
- The overall purpose: filtering out assets whose geometry causes unstable physics simulation

---

## Suggested Rewrite Points for the Article

1. **Protocol paragraph:** Replace any mention of "snap position" or "two-phase" with the single-drop protocol. Emphasise that the consistent 10 cm spawn height makes results comparable across all assets regardless of shape.

2. **Metrics table:** Replace the four removed fields with `penetration_y_m` and `floor_penetration`. Update `physics_settles` to `physics_stable` as the primary reported metric.

3. **Pass rate numbers:** The reported pass rate should be `physics_stable` (settles + no flying + no floor penetration), not raw `physics_settles`. This is a stricter criterion and the numbers will differ.

4. **Motivation for floor penetration metric:** The floor penetration check catches assets with degenerate convex hull decompositions that allow the simulation to tunnel through the ground plane — a different failure mode from flying away or not settling.

5. **Remove any comparison to snap reference:** Any sentence like "offset from the initial resting position" or "sinking below the reference point" should be removed or replaced with "penetration below the floor plane."

6. **Add simulation tractability paragraph:** Report `wall_time_s` statistics (mean, median, max) and the within-60s completion rate per dataset. Frame the within-60s rate as a "simulation readiness" criterion separate from physics stability — an asset can be geometrically stable but computationally intractable at scale. This is particularly relevant for cross-dataset comparison (e.g. Objaverse vs. Amara vs. YCB).
