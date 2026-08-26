# Plan B — handoff (resume Task 6)

**Temporary doc.** Hands a fresh agent everything needed to finish Plan B (the Isaac Lab
integration). Delete once Task 6 lands. Last updated end of the session that completed
Task 5 and de-risked Task 6.

## TL;DR

`lighthill` is a GPU-vectorized per-link Fossen hydrodynamics layer for Isaac Lab (turns the
dry rigid-body sim into an underwater UVMS sim). Plan B wires the Plan A physics core into
Isaac. **Tasks 1–5 are done and validated; Task 6 (the arm-swing coupling gate) is the only
thing left, and its hard unknowns are already resolved — what remains is careful physics
modeling (a reference + a clean gate scenario).**

- Plan: `docs/plans/2026-06-29-plan-b-isaac-integration.md`
- Spec: `docs/design/2026-06-28-hydrodynamics-design.md`
- API facts (pinned from a live spike): `docs/isaac-api-findings.md`
- Findings for the paper (READ THIS): `docs/paper-notes.md`
- Progress ledger (SDD shorthand): `.superpowers/sdd/progress.md` (git-ignored scratch)

## Branch / PR state

- **PR #3** — `feat/plan-b-isaac-integration` → `main`: the CPU core (Tasks 1–4, Isaac-free
  engine). Open, gate green. Whole-branch reviewed.
- **`feat/plan-b-isaac-adapter`** (pushed): stacked on the PR #3 branch. Holds Task 1 spike +
  Task 5 (adapter + 3 in-sim validations) + the `apply.py` GPU device fix. **You work here.**
  When PR #3 merges, rebase this onto `main` (clean — no file overlap with the CPU core).

Resume git-wise: `git checkout feat/plan-b-isaac-adapter`.

## Running Isaac (operational recipe — non-obvious, learn this first)

The lighthill CPU dev venv has CPU torch; Isaac lives in a **separate** env:

```bash
export OMNI_KIT_ACCEPT_EULA=YES   # REQUIRED or Isaac hangs on an interactive EULA prompt
ISAAC_PY="E:/workspaces/research/isaac-lab-env/env_isaaclab/Scripts/python.exe"
"$ISAAC_PY" sim_validation/drag_terminal.py   # ~1-2 min launch (shader cache) + run
```

- **Isaac reliably HANGS on `simulation_app.close()`** (headless and GUI). Every standalone
  script ends with `os._exit(0)` in a `finally:` instead. If a run times out / a window won't
  close, the process is zombied on teardown — kill it:
  ```powershell
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'env_isaaclab|<scriptname>' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
  ```
- lighthill is installed **editable** into the Isaac env, so source edits are picked up live.
- The CPU gate (`just check`) runs in the lighthill venv and SKIPS the real_sim tests.
- Run the in-sim gate deliberately: `LIGHTHILL_REAL_SIM_OK=1 uv run pytest
  tests/test_apply_isaac_marker.py -p no:xdist` (each scenario subprocess launches its own
  Isaac; serial to avoid GPU contention).
- GPU: RTX 4060 Ti 16GB; one Isaac instance at a time (don't run two — they fight for the GPU).

## What's done (Tasks 1–5)

- **Tasks 1–4 (CPU core, PR #3):** `articulation.py` (ArticulationView Protocol + FakeArticulation),
  `inertia.py` (added-mass routing), `accel.py` (EMA accel filter), `apply.py`
  (UnderwaterHydrodynamics). 57 tests, 90.8% coverage.
- **Task 1 spike:** `docs/isaac-api-findings.md`. Key facts: read `data.body_{pos,quat,lin_vel,
  ang_vel}_w` (world frame, scalar-first wxyz quats; convert velocities to body via
  `frames.world_vec_to_body`); apply `set_external_force_and_torque(f, m, is_global=True)` +
  `write_data_to_sim()`; mass/inertia via `root_physx_view.set_masses/set_inertias`; dt via
  `sim.get_physics_dt()`.
- **Task 5 (adapter + validation):** `apply_isaac.py` (`IsaacArticulationView`), and three
  passing in-sim scenarios in `sim_validation/` vs the Plan A CPU reference:
  drag-terminal **0.41%**, free-decay **0.02%**, restoring **0.07°**. `real_sim` marker test
  wired. See `sim_validation/README.md`.

## Task 6 — what remains (the only open work)

The arm-swing reaction gate: command an arm trajectory on a vehicle+arm articulation with the
base free, and verify the measured base reaction matches an analytical Fossen+Featherstone
reference. **Do not loosen the tolerance to pass — a fail means the coupling is wrong.**

**De-risked this session (the hard unknowns — all confirmed working):**
- A 2-body (base+arm, revolute joint) articulation can be authored procedurally (no external
  USD) and wrapped as an Isaac Lab `Articulation`.
- lighthill's **multi-body** path (B=2) runs through the adapter end-to-end:
  `IsaacArticulationView(robot)` + `UnderwaterHydrodynamics(view, coeffs)` works; adapter shapes
  `(E,2)`/`(E,2,3)`; `set_body_inertias` adapts to the Articulation physx view.
- Dry sim: arm swing → free base counter-rotates (momentum coupling). Underwater: damped
  reaction. **The coupling is physically present and behaves correctly.**

**Remaining (3 pieces), in order:**
1. **Coupled Featherstone CPU reference** (pure Python, NO Isaac — do this first, fresh).
   Extend the Plan A single-body integrator (`validation/reference.py`) to a floating-base
   2-body chain (vehicle + 1 arm link) with per-link hydro and a commanded joint angle, to
   compute the analytical base reaction to an arm swing. Document the derivation + tolerance
   rationale in `sim_validation/reference_featherstone.md`. This is the correctness-critical
   piece — get it right.
2. **`sim_validation/arm_swing_reaction.py`** — the in-sim scenario. Reuse the proven
   articulation-authoring code (below). Critical scenario-design note: the free assembly is
   **buoyancy-untrimmed** in the naive setup (a ~25° tipping transient swamps the reaction).
   Trim the assembly (combined CoB = combined CoM) and/or start trimmed and exclude the
   transient so the gate isolates the arm-swing reaction. Same DOF-isolation discipline as the
   single-body scenarios (see `sim_validation/README.md`).
3. **The gate test** — add a `real_sim` case to `tests/test_apply_isaac_marker.py` asserting
   `peak_rel_error < <tol from reference_featherstone.md>`. Then update `docs/paper-notes.md`
   §4 results + §5, and finish the ledger.

### Proven articulation-authoring code (reuse verbatim)

This spawned correctly and produced the right multi-body coupling. Gravity shown off for the
dry-coupling check; turn it on (`(0,0,-9.81)`) + `PhysxCfg(enable_external_forces_every_iteration=True)`
for the hydro gate.

```python
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext

sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0", gravity=(0,0,-9.81),
        physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True)))
stage = omni.usd.get_context().get_stage()
root = "/World/Robot"
UsdGeom.Xform.Define(stage, root)
UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(root))

def make_link(path, scale, mass, pos):
    cube = UsdGeom.Cube.Define(stage, path); cube.GetSizeAttr().Set(1.0)
    x = UsdGeom.Xformable(cube)
    x.AddTranslateOp().Set(Gf.Vec3d(*pos)); x.AddScaleOp().Set(Gf.Vec3f(*scale))
    p = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(p); UsdPhysics.RigidBodyAPI.Apply(p)
    UsdPhysics.MassAPI.Apply(p).GetMassAttr().Set(mass)
    return p

base = make_link(root+"/base", (0.5,0.4,0.3), 13.7, (0,0,0))
arm  = make_link(root+"/arm",  (0.08,0.08,0.5), 0.6, (0,0,-0.55))
j = UsdPhysics.RevoluteJoint.Define(stage, root+"/joint")
j.CreateBody0Rel().SetTargets([base.GetPath()]); j.CreateBody1Rel().SetTargets([arm.GetPath()])
j.CreateAxisAttr().Set("Y")
j.CreateLocalPos0Attr().Set(Gf.Vec3f(0,0,-0.15)); j.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0.25))
d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
d.CreateTypeAttr().Set("force"); d.CreateStiffnessAttr().Set(1500.0); d.CreateDampingAttr().Set(80.0)
d.CreateTargetPositionAttr().Set(0.0)

robot = Articulation(ArticulationCfg(prim_path=root, spawn=None,
    init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,0)),
    actuators={"joint": ImplicitActuatorCfg(joint_names_expr=["joint"], stiffness=1500.0, damping=80.0)}))
sim.reset()
# body order is ['base','arm'] -> coeffs config links must be (base, arm) in that order.
# command:  robot.set_joint_position_target(torch.tensor([[target_rad]], device=dev))
#           hydro.apply(dt); robot.write_data_to_sim(); sim.step(); robot.update(dt)
# NOTE: the drive at k=1500 only tracked ~8.5deg of a fast +-50deg sinusoid -> stiffen it,
#       or command a slower/smaller swing, for a clean commanded trajectory.
```

Full working spikes were in the prior session's scratchpad (`spike_articulation.py`,
`spike_articulation_hydro.py`) — session-specific, may be gone; the snippet above is the
distilled, proven version.

## Critical findings to respect (full detail in `docs/paper-notes.md`)

- **Device-pinning bug class:** the CPU gate cannot catch GPU device-mixing; 5 instances were
  found (4 by review, 1 by the first in-sim run). Any new tensor must be on the view's device.
- **Munk directional instability:** a free thrust-driven slender body tumbles — real physics;
  why translational scenarios pin attitude.
- **Isaac external-force accuracy:** enable `PhysxCfg.enable_external_forces_every_iteration=True`
  or applied buoyancy is under-integrated vs gravity (spurious sink). Required for the gate.
- **Inertia parity:** when comparing rotational dynamics to the CPU reference, feed it the body's
  ACTUAL inertia read from Isaac (see `restoring.py`), or oscillation phase diverges.
- **`set_external_force_and_torque` is deprecation-flagged** toward `permanent_wrench_composer`
  — still works in Isaac Lab 2.3; pin the version.

## Suggested first move

Build the Featherstone reference (piece 1) entirely in Python/tests first — it needs no Isaac,
it's the correctness-critical artifact, and having it ready makes the in-sim gate a
straightforward comparison. Use the `superpowers:subagent-driven-development` discipline if
continuing the SDD flow (the ledger and prior task structure are already set up).
