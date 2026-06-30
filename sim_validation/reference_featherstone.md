# Coupled Featherstone reference — derivation & tolerance (Task 6)

The arm-swing reaction gate asks one question: **when the arm is commanded through a
trajectory and the vehicle is free, does the vehicle react the way the coupled
dynamics say it should?** This is the headline UVMS claim. The gate compares the
in-sim base reaction (`sim_validation/arm_swing_reaction.py`) against the analytical
reference implemented in `src/lighthill/validation/reference_coupled.py`. This doc
derives that reference and justifies the gate tolerance.

## What is (and isn't) under test

lighthill applies a **per-link** wrench and lets the rigid-body solver propagate the
vehicle↔arm coupling (we never compute inter-link hydro coupling — design decision,
matches DAVE / Kolano 2022). The single-body scenarios (drag-terminal 0.41 %,
free-decay 0.02 %, restoring 0.07°) already certified the **force law** and the
**glue** (state read, frame conversion, wrench/inertia application) on one body. So
this gate must isolate the one thing they don't exercise: **the coupling** — the
reaction transmitted between a free base and a moving arm through the articulated
chain.

To isolate it, the reference uses the **same** per-link modeling as the sim, so any
discrepancy is coupling, not force law:

* Per-link hydro wrench is computed by the **same kernels** the Isaac adapter drives
  — `buoyancy_wrench + drag_wrench + added_mass_coriolis + added_mass_residual` — with
  the same EMA acceleration filter (`accel.AccelerationFilter`, α≈0.08) feeding the
  residual.
* Diagonal added mass folds into each link's **effective mass/inertia** exactly as
  `inertia.split_added_mass` + `set_body_inertias` do in sim (isotropic linear part →
  scalar PhysX mass; angular diagonal → inertia tensor; anisotropic-linear remainder
  + off-diagonal → the residual wrench).
* Gravity acts on the **effective** mass, matching PhysX: `set_body_inertias` writes
  the augmented mass via `set_masses`, and PhysX applies `gravity = m_eff · g`. (This
  is a real gotcha — see "Trim" below.)

The **only** new physics is the floating-base coupling solve.

## The floating-base reaction equation

System: a free base `B0` (6 DOF, unactuated — only hydro + gravity act on it) and one
arm link `B1` on a revolute joint whose angle `q(t)` is **prescribed** (commanded in
sim by a stiff drive). Generalized coordinates are the base pose + `q`; because the
base is unactuated, its 6 equations of motion contain **no joint-actuator torque**
(that torque is conjugate to `q` only). So the base acceleration follows from the
base-coordinate block alone — no need to model the drive torque.

Working in the world frame with the base CoM as the reference point `O`, write
Newton–Euler for both bodies and eliminate the internal joint wrench by taking the
**total** linear and angular momentum rates (internal wrenches cancel). With the arm
CoM offset `r = R₀ d(q)` (world), arm angular velocity `ω₁ = ω₀ + n̂ q̇`
(`n̂ = R₀·axis`), and the joint-driven acceleration biases `k₁` (linear), `k₂`
(angular) from the kinematics, the result is a 6×6 system

```
[ (m0+m1) I3      −m1 [r]×              ] [ a0 ]   [ ΣF − m1 k1            ]
[  m1 [r]×        I0ʷ + I1ʷ − m1 [r]×²  ] [ α0 ] = [ T_O − bias_ang       ]
```

where `[r]×` is the skew matrix of `r`, `Iiʷ = Ri diag(Ii_eff) Riᵀ`, `ΣF` is the
total external force, and `T_O` the total external torque about `O`
(`= Σ τi + Σ ρi×Fi`, with `ρ0 = 0`, `ρ1 = r`). The angular bias collects the
velocity-product (gyroscopic) and joint-driven coupling terms:
`ω0×(I0ʷ ω0) + I1ʷ k2 + ω1×(I1ʷ ω1) + m1(ṙ×v1 + r×k1)`.

**This left-hand matrix is exactly the composite spatial inertia of the two-body
system about the base CoM** — its lower-right block `I0ʷ + I1ʷ + m1(|r|²I − rrᵀ)` is
the arm's parallel-axis term, and the off-diagonal `m1[r]×` is `m·skew(c)` for the
system CoM offset `c = m1 r/(m0+m1)`. The matrix is **symmetric by construction**
(top-right `−m1[r]×` = bottom-left transpose), which is the structural correctness
signal we lean on: a sign error in the coupling breaks the symmetry and the
conservation tests below. Solving gives the base linear/angular acceleration; the
base is then advanced by semi-implicit Euler, and the arm follows kinematically from
`q(t)`.

Full derivation, term by term, is in the module docstring and the commit message.

## How the reference is certified (no Isaac)

`tests/test_reference_coupled.py` (10 tests, in the CPU gate) certifies the physics
before any GPU is involved:

1. **Arm kinematics** — hand-computed values for the offset `r`, world axis `n̂`, arm
   velocity, and the `k₁/k₂` acceleration biases at chosen `q, q̇, q̈`.
2. **Momentum conservation under an arm swing** — with no external forces, total
   linear *and* angular momentum stay at zero while the base visibly recoils. This is
   the direct test of the coupling.
3. **First-order-in-dt convergence** — the momentum residual **halves when dt
   halves** (verified across dt ∈ {4,2,1,0.5} ms, ratios ≈ 2.0). This proves the
   residual is semi-implicit-Euler truncation, **not** a coupling/sign error (a
   modeling bug leaves a dt-independent residual). This is the strongest correctness
   evidence.
4. **Single-body limit** — as the arm mass/inertia → 0 the reaction it transmits → 0,
   so the base stays at rest. Recovers the one-body behaviour.
5. **Trimmed static equilibrium** + a **gravity-sink control** — confirms the
   gravity + buoyancy wiring (balanced → base holds; buoyancy off → base sinks).
6. **Hydro-laden swing** — drag + anisotropic-added-mass residual measurably break
   conservation and push the system CoM, exercising every per-link kernel in the
   coupled path.

## Scenario design (DOF isolation) — `arm_swing_reaction.py`

Same isolation discipline as the single-body scenarios:

* **Feed the sim's *actual* joint angle to the reference.** A stiff PD drive does not
  track a commanded trajectory perfectly (the spike saw a 1500-stiffness drive track
  only ~8.5° of a fast ±50° swing). Rather than assume perfect tracking, the scenario
  records the realized `q(t)` from the sim and differentiates it for the reference —
  exactly as `restoring.py` reads the body's actual inertia. This removes
  drive-tracking error from the comparison, leaving only the coupling + discretization.
  Command a **slow, moderate** swing so the drive tracks cleanly anyway.
* **Trim the assembly.** The free assembly is buoyancy-untrimmed in the naive setup
  (PhysX applies gravity on `m_eff = rigid + added-mass bump`, so an untrimmed base
  sinks/tips — a ~25° transient that swamps the reaction). Run the gate with **gravity
  off** (cleanest isolation of the momentum coupling — the reaction we test is present
  with or without gravity) or trim each link to `buoyancy = m_eff·g` with `CoB = CoM`.
  Gravity-off is preferred: it removes the augmented-mass-gravity confound entirely.
* **Enable `PhysxCfg.enable_external_forces_every_iteration=True`** (Finding C) — the
  coupling is force-balance-sensitive, so the under-integrated-external-force bias must
  be off.
* **Exclude the startup transient** (first ~0.2 s) before scoring.

## The gate metric and tolerance

The measured signal is the base reaction (CoM displacement / velocity / angular
velocity) over the swing. Relative error is normalized by the **signal scale**, not
pointwise, to avoid the divide-by-near-zero blow-up at reaction zero-crossings:

```
peak_rel_error = max_t |sim(t) − ref(t)|  /  max_t |ref(t)|
```

**Tolerance: `peak_rel_error < 0.15` (15 %).** Rationale — with the force law and glue
already certified to <0.5 % single-body and the *actual* `q(t)` fed to the reference,
the residual budget is:

| Source | Contribution |
|---|---|
| PhysX TGS solver vs the reference's semi-implicit Euler, on the coupled DOF at sim dt | dominant; a few–several % |
| Residual-lag (anisotropic added mass via the lagged EMA filter) — **same approximation both sides**, so largely cancels | small |
| Stiff-drive joint-tracking residual (mitigated by feeding actual q) | small |
| Low SNR of a small reaction signal near zero-crossings | inflates peak-relative |

15 % is a **ceiling that still falsifies a wrong coupling**: the conservation +
convergence tests show a correct coupling tracks the reference to truncation order, so
a frame-sign error, a wrong wrench frame, a missing inertia augmentation, or a
mis-routed added-mass term produces errors far larger than 15 % (typically a wrong
sign or a factor, i.e. ≥100 %). **Do not loosen this to pass.** A failure means the
coupling is wrong — investigate frame sign, residual α, wrench frame, or inertia
augmentation (per the plan), not the threshold. If the realized error is comfortably
under 15 % (expected), record the actual number in `docs/paper-notes.md` §4 and
tighten the documented expectation accordingly.
