# lighthill — underwater hydrodynamics for Isaac Lab (AUV → UVMS) — design

> **Status: design (brainstormed 2026-06-28).** The design doc for the `lighthill` Isaac Lab
> plugin — GPU-vectorized per-link hydrodynamics for articulated underwater robots. Originated
> to serve an external research project's environment need (arm-state-predictive RL for a UVMS)
> and built as a standalone, releasable tool spanning single-body AUV → coupled UVMS. Next
> step: implementation plan (writing-plans).

## Goal & success bar

Turn Isaac Lab's dry rigid-body sim into an **underwater** sim by computing hydrodynamic forces
and applying them as **per-link external wrenches**. It is **general across underwater platforms**:
a single-body **AUV (no arm) is first-class**, and a coupled **vehicle-manipulator system (UVMS)**
is the headline configuration this paper uses — where the articulation solver then produces the
**vehicle↔arm reaction coupling**. **The arm is optional**, so other researchers can run ordinary
AUV experiments with the same tool.

**Success bar = coupling-faithful (not reality-faithful).** The module must reproduce the
*structure* of the coupling correctly and stay internally consistent, **validated against the
analytical Fossen+Featherstone reference** (Kolano 2022, `kolano2022coupling`). Coefficients may
be literature/shape-derived, not experimentally identified. The reality gap is deferred to I002
(sim-to-real). Sim-only.

**Tool-grade.** Built general, per-link, and config-driven so **any underwater platform plugs in —
a vehicle alone (AUV) or a vehicle + arm (UVMS)**. This is what makes it a releasable artifact and
the basis of I004, not throwaway scaffolding.

## Why (the gap)

No simulator is simultaneously GPU-RL-fast **and** spans single-body **AUV through coupled UVMS**
**and** has currents. MarineGym = fast + currents but **AUV-only, base-link only** (confirmed in
code); Angler/Stonefish have arms but no RL / too slow. This extension is **one general, per-link
underwater hydrodynamics tool for Isaac Lab**: it **subsumes the AUV case** (Isaac-Lab-native, more
general than MarineGym) **and** fills the **per-link coupled-UVMS gap** that is net-new — the
contribution. Researchers doing ordinary AUV work and those doing UVMS work use the same tool.

## Architecture

A single component, `UnderwaterHydrodynamics`, attachable to **any underwater articulation with
1..N links — a single-body AUV through a coupled vehicle+arm UVMS**. Each `pre_physics_step`:

1. Read every body's state (world pose, linear+angular velocity) from the Isaac Lab
   `Articulation` view (GPU tensors).
2. For each link, compute its Fossen wrench in the body frame; transform to the apply frame.
3. Apply the wrench **per link** via the Isaac Lab wrench API (the current composable wrench
   system, *not* the deprecated `set_external_force_and_torque`).
4. **The articulation (Featherstone) solver propagates the inter-link dynamic coupling.** We do
   **not** compute hydro coupling between links — that emerges from the rigid-body chain (matches
   DAVE practice and Kolano's method). **For a single-body AUV there is no chain — the same
   per-link path simply runs over one link**, so the AUV case needs no special handling.

Vectorized over `(num_envs × num_links × 6)`; all tensors stay on GPU; no CPU round-trips.

### Module boundaries (one responsibility each)
- `config` — per-link parameter schema (dataclass/YAML), validated at load (matrix symmetry, etc.).
- `coefficients` — resolve per-link coefficients: explicit 6×6 **or** shape-based (analytic).
- `forces` — pure functions: `buoyancy()`, `drag()`, `added_mass_residual()`, `coriolis()`,
  given batched body states + coefficients → wrench tensors. No Isaac dependency (unit-testable).
- `current` — the flow field + per-link relative-velocity computation.
- `apply` — the Isaac Lab glue: read articulation state, call `forces`, apply per-link wrench,
  set augmented inertias at init.
- `validation` — the analytical reference + the gate tests.

### Generality — topology-agnostic (any underwater articulation)
The per-link-over-the-articulation design is **topology-agnostic** — the same path handles:
- **AUV** (1 body) · **UVMS** (vehicle + 1 arm) · **multi-arm UVMS** (vehicle + 2+ arms). A
  second arm is a *branched* tree; Featherstone handles branches, so it's just more config
  blocks — **no special code**.
- **Hyper-redundant swimmers** (underwater **snake/eel** robots) — a long serial chain of
  cylinder segments. The cylinder shape model's **anisotropic drag** (transverse ≫ tangential)
  plus per-segment added mass *is* the **resistive-force / elongated-body model** used for
  undulatory-swimming control, so bio-inspired swimmers are covered by construction.
  - **Caveat:** for a swimmer whose *propulsion is the hydro itself* (no thrusters),
    "coupling-faithful" extends to "thrust-from-undulation-faithful." The per-segment model gives
    **resistive-force-theory-level** fidelity (the standard for control/RL), **not** vortex/wake-
    resolved CFD — a future high-fidelity tier.

We **validate + ship AUV + UVMS** configs now (the paper's need); **multi-arm and snake/eel are
supported by construction**, offered as future example configs — a large audience expansion for
the tool (I004) at no extra design cost.

## Force model (per link)

Fossen terms, per body, with current giving relative velocity `v_r = v_body − R_bodyᵀ·v_current`:

- **Buoyancy / restoring:** `F_b = ρ·g·V` (up, world frame) at the 3D **center of buoyancy**;
  the CoB↔CoM offset produces the restoring moment. Per-link V and CoB from config.
- **Drag (damping):** `D(v_r)·v_r` with `D = D_lin + |v_r|·D_quad` (linear + quadratic).
- **Added-mass Coriolis:** `C_A(v_r)·v_r` from the skew form of `M_A·v_r` (cheap; matters for the
  vehicle's angular motion; small for slow arm links but included).
- **Added mass:** see below (the resolved fork).

### Added mass — hybrid (the key decision)
- **Diagonal** added-mass → **inertia augmentation**: set each body's effective mass/inertia to
  `M_RB + diag(M_A)` via Isaac Lab `set_body_inertias` at init. Stable, needs no acceleration
  estimate. (Stonefish's stability, but per-axis — *not* its scalar-average mistake.)
- **Off-diagonal** added-mass → a small **residual wrench** `−M_A_offdiag·a_filtered`, with
  `a_filtered` from finite-diff + low-pass (α≈0.05–0.1). Only the off-diagonal residual goes
  through the noisy path, so the oscillation risk UUV-Sim warns about is minimized.
- For arm links modeled as **shapes** (below), M_A is diagonal-dominant → inertia augmentation
  alone; no residual wrench needed.

### Shape-based hydro for arm links (solves the coefficient problem)
Slender arm links (Reach Alpha) have no identified coefficients. Model them analytically via a
config `type`:
- **cylinder** (R, L, axis): `M_A_transverse = ρπR²L`, `M_A_axial ≈ 0`; drag from cross-section.
- **sphere** (R): `M_A = (2/3)πρR³` (isotropic).
- **box** (with C_d): form-drag + analytic M_A.
Defensible, no invented numbers, and a clean config option alongside explicit `type: fossen`.

## Current model
Uniform, **global** per env (one vector), randomized magnitude+direction in **[0, 0.5] m/s** at
reset (domain randomization), optional small per-step Gaussian noise. Enters **drag + Coriolis
only** via `v_r`; the added-mass term uses *absolute* body acceleration (a constant current adds
nothing to it — a commonly-mis-ported subtlety). Spatially-varying/sheared currents are out of
scope (future).

## Frames & conventions
**NWU throughout** (Isaac/PhysX convention). Fossen's equations are NED — all damping signs,
restoring directions, and coefficient axes are translated to NWU **explicitly** and asserted in
tests. (UUV-Sim's `ToNED` + MarineGym's undocumented axis sign-flips are the cautionary tales.)

## Config schema (the tool surface)
Per link (Python dataclass / YAML), borrowed from UUV-Sim's proven schema:
```
link:
  name: <body name in the articulation>
  volume: <m^3>                      # for buoyancy
  center_of_buoyancy: [x, y, z]      # body frame, m (3D, not scalar)
  neutrally_buoyant: <bool>          # skip buoyancy if true
  added_mass:                        # one of:
    type: fossen        # explicit 6x6 (36 floats) OR 6-diagonal
    matrix: [...]
    # --- or ---
    type: cylinder      # analytic: radius, length, axis
    radius: 0.025 ; length: 0.15 ; axis: z
  linear_damping: [6 or 36]
  quadratic_damping: [6 or 36]
```
A config is **1..N link blocks**: an **AUV config = one vehicle link**; a **UVMS config = the
vehicle link** (explicit `fossen`, full 6×6) **+ one block per arm link** (usually `cylinder`).
Ship **two** example configs: an **AUV-only BlueROV2** (coefficients from MarineGym's
`BlueROV.yaml`, MIT — a drop-in MarineGym-equivalent) **and** a **UVMS BlueROV2 + cylinder-modeled
Reach Alpha** arm.

## Borrow vs. build (with provenance)
- **Borrow (MarineGym, MIT — `chu2025marinegym`):** the vehicle Fossen formulas (damping,
  buoyancy, Coriolis), the BlueROV2 diagonal coefficients, the GPU `(N,6,6)@(N,6,1)` batching.
- **Borrow (UUV-Sim/DAVE, Apache-2.0):** the per-link config schema, the shape sub-models
  (cylinder/sphere/box), the per-link-independent-forces pattern.
- **Build fresh:** per-link application over the **articulation** (not a single base link), the
  **hybrid added-mass**, the **Isaac Lab** (not Isaac Sim/OmniDrones) wrench API, NWU frame
  handling, the config/dataclass layer, and the **validation harness**.

## Validation (the Phase-1.3 gate — shipped with the tool)
Against the analytical Fossen+Featherstone reference (Kolano 2022):
1. **Free-decay** — a displaced, released body's oscillation/settle matches the reference.
2. **Drag terminal velocity** — a thrust-loaded body reaches the analytically-predicted terminal
   speed.
3. **Buoyancy/restoring** — static tilt → restoring moment matches `(CoB−CoM)×F_b`.
4. **Arm-swing reaction (the crux)** — commanding an arm trajectory with the base free, the
   measured base reaction (force/displacement) matches the Featherstone+per-link-hydro reference
   within tolerance. **Do not proceed past a failing arm-swing test** — wrong coupling
   invalidates the science.
**Tests 1–3 validate the AUV-only (single-body) case; test 4 (arm-swing) applies only when an arm
is configured** and is the UVMS coupling gate. Pass criteria: relative error under a stated
tolerance per test; tests ship with the tool so users trust it.

## Scope
**In:** **AUV-only (single-body) and coupled UVMS** configurations; per-link
buoyancy/drag/added-mass (+ shape models), uniform current, NWU, GPU-vectorized, config-driven,
the validation harness, **AUV-only BlueROV2 + UVMS BlueROV2+Alpha** example configs.
**Out (future):** full per-link off-diagonal 6×6 added-mass for arm links (shapes suffice now);
spatially-varying/turbulent currents; free-surface/waves; **vortex/wake-resolved propulsion
fidelity for undulatory swimmers** (the per-segment resistive model suffices for control/RL);
experimental coefficient identification and sim-to-real (I002); the RL task/gym wrapper and
benchmark suite (the *env* layer that sits on top of this module — separate spec, but this module
is its foundation).

## Risks / open questions (resolve at implementation)
- **Isaac Lab wrench API specifics** — confirm the current composable per-body wrench call +
  whether `set_body_inertias` accepts the augmented values at runtime vs init only.
- **Residual-wrench stability** at the chosen `dt` and α — pilot-tune; fall back to
  diagonal-only added mass if the off-diagonal residual misbehaves.
- **Buoyancy tuning per arm link** — calibrate volumes so arm links are ~neutral (a 0.002 m³ link
  is positively buoyant against a 0.1 kg mass); document the convention.
- **Articulation state read cost** — verify per-link state reads don't bottleneck the step at
  high env counts.
