# Isaac Lab API findings (Plan B Task 1 spike)

Confirmed against the **live local install** — Isaac Sim 5.1 / Isaac Lab 2.3.2,
RTX 4060 Ti, CUDA torch 2.7.0+cu128 — via `scripts`-level spike (a single
primitive cuboid `RigidObject`, headless). These are the calls the
`IsaacArticulationView` adapter (`apply_isaac.py`) is built against.

> Spike method: spawn one `RigidObject` cuboid (mass 13.5 kg) with gravity off,
> read its state, apply a +X **body-frame** force, step, and confirm it moves +X.
> Reproduce with `scratchpad/isaac_spike.py` (see commit history / dev notes).

## Q1 — Read API (per-body world pose + velocity)

`asset.data` exposes both root- and body-indexed views. For the multi-link
articulation case the adapter uses the **`body_*`** fields:

| Field | Shape | Frame |
|---|---|---|
| `data.body_pos_w` | `(num_envs, num_bodies, 3)` | world |
| `data.body_quat_w` | `(num_envs, num_bodies, 4)` | world, **scalar-first `(w,x,y,z)`** |
| `data.body_lin_vel_w` | `(num_envs, num_bodies, 3)` | **world** |
| `data.body_ang_vel_w` | `(num_envs, num_bodies, 3)` | **world** |

- **Quaternion order confirmed:** a body at rest reads `[1, 0, 0, 0]` → scalar-first
  `(w, x, y, z)`, body→world. Matches lighthill's convention exactly; no reorder needed.
- **Velocities are WORLD-frame.** lighthill's kernels expect **body-frame** twist
  `[u,v,w,p,q,r]`, so the adapter must convert: `v_body = frames.world_vec_to_body(v_world, quat)`
  for both linear and angular parts, then pack into `[E,B,6]`.
- (`RigidObject` single-body also exposes `root_pos_w (E,3)` / `root_quat_w (E,4)` etc.;
  the adapter standardizes on the `body_*` `(E,B,*)` shape so single-body and
  multi-body share one path.)

## Q2 — Per-body external wrench

```python
asset.set_external_force_and_torque(
    forces,            # (num_envs, num_bodies, 3)
    torques,           # (num_envs, num_bodies, 3)
    positions=None,    # optional application point; None = body CoM
    body_ids=None,     # None = all bodies
    env_ids=None,      # None = all envs
    is_global=False,   # False = forces/torques in BODY/LINK-LOCAL frame
)
asset.write_data_to_sim()   # MUST be called before sim.step() or the buffer is ignored
```

- **This is the current API, NOT deprecated.** The plan/design speculated a newer
  "composable wrench system" had superseded `set_external_force_and_torque`; the live
  install shows it is the standard call on both `RigidObject` and `Articulation` in
  Isaac Lab 2.3. The design note about avoiding a deprecated call is **moot** — use this.
- **Frame confirmed by experiment:** with `is_global=False` (the default), a `+X`
  force vector moved the body along `+X` → forces are interpreted in the **body-local**
  frame.
- **lighthill integration choice:** `UnderwaterHydrodynamics.apply()` already converts the
  body-frame wrench to **world** frame and hands `set_external_wrench` a world wrench. The
  adapter therefore calls `set_external_force_and_torque(f_world, m_world, is_global=True)`.
  (Body-local with `is_global=False` would also work and skip the rotation, but that would
  require changing the merged CPU core's `apply()`; not worth it — the world path is correct.)
- **Persistence:** the external-wrench buffer **persists across `sim.step()`** until
  overwritten. lighthill re-sets it every step in `apply()`, so this is harmless, but worth
  knowing (do not assume it auto-clears).

## Q3 — Mass / inertia (for diagonal added-mass augmentation)

- **Read:** `data.default_mass` `(num_envs, num_bodies)`; `data.default_inertia`
  `(num_envs, num_bodies, 9)` — the inertia is a **flattened 3×3** (row-major), not a
  3-vector. The adapter exposes `mass` `[E,B]` and `inertia_diag` `[E,B,3]` (per the
  `ArticulationView` Protocol) by reading these and taking the principal diagonal
  `(indices 0,4,8)` for `inertia_diag`.
- **Write:** via the PhysX tensor view — `asset.root_physx_view` (a `RigidBodyView`;
  `ArticulationView` for articulations) exposes `set_masses(masses)` and
  `set_inertias(inertias)`. `get_masses()` → `(count, num_bodies)`, `get_inertias()` →
  `(count, num_bodies, 9)`.
- **Adapter `set_body_inertias(mass[E,B], inertia_diag[E,B,3])`** therefore: `set_masses(mass)`,
  and builds a flattened diagonal 3×3 from `inertia_diag` (zeros off-diagonal, diag at 0/4/8)
  → `set_inertias(...)`. This is an **init-time** call (the diagonal added-mass augmentation
  is computed once); runtime per-step updates are not needed.

## Q4 — Step hook + dt

- **dt:** `sim.get_physics_dt()` returns the physics step (equals `SimulationCfg.dt`,
  `0.005` in the spike). The adapter/validation passes this to `apply(dt)`.
- **Apply hook:** call `hydro.apply(dt)` → `write_data_to_sim()` **before** each
  `sim.step()`. In a standalone validation loop that is an explicit sequence; in a managed
  Isaac Lab env it goes in `pre_physics_step` (or a physics callback). The wrench must be
  written into the buffer in the same step it is meant to act.

## Operational note (headless teardown)

`SimulationApp` / Omniverse Kit reliably **hangs on `simulation_app.close()`** in headless
runs — the process lingers and never returns. Standalone Isaac scripts here end with
`os._exit(0)` after writing results (and are run foreground with a timeout) to force a clean
exit and avoid zombie processes. Do not rely on a clean `close()`.

## What this pins for the adapter (`apply_isaac.py`)

`IsaacArticulationView(ArticulationView)` wrapping an Isaac `Articulation`/`RigidObject`:
- `body_states()` → read `body_pos_w / body_quat_w / body_lin_vel_w / body_ang_vel_w`;
  convert the two world velocities to body frame via `frames.world_vec_to_body`, pack
  `vel = [E,B,6]`. Return `(pos, quat, vel)`.
- `set_external_wrench(world_wrench[E,B,6])` → split into forces/torques and call
  `set_external_force_and_torque(f, m, is_global=True)` then `write_data_to_sim()`.
- `set_body_inertias(mass[E,B], inertia_diag[E,B,3])` → `root_physx_view.set_masses` +
  `set_inertias` (diag→flattened 3×3).
- Expose `mass [E,B]` and `inertia_diag [E,B,3]` read attributes (Protocol contract) from
  `data.default_mass` / `data.default_inertia` (diagonal).
