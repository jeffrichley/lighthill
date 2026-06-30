# lighthill — running paper notes (lab notebook)

**Living document.** Accumulates design rationale, validation methodology, results,
and findings as the tool is built, in paper-ready framing. Append-only log at the
bottom; structured sections above are kept current.

**Feeds:**
- **I004** — the open hydrodynamics tool / benchmark paper (`lighthill` is the artifact).
- **I003** — `uvms-arm-state-rl` (RA-L target): this is its simulation methodology +
  validation evidence (the environment the RL results run on).

**Why keep this:** the most publishable parts of a tools paper are the *non-obvious*
findings discovered while building it — physical (the Munk instability), methodological
(dual-reference validation), and engineering (the device-pinning bug class, the Isaac
force-application gotcha). They are easy to lose if not logged when found.

---

## 1. Claim / contribution (draft)

A GPU-vectorized, per-link Fossen hydrodynamics layer for NVIDIA Isaac Lab that turns its
dry rigid-body sim into an underwater UVMS sim, **spanning single-body AUV → coupled
vehicle+manipulator**, with currents, validated **coupling-faithful** against an analytical
Fossen+Featherstone reference. Net-new vs prior art: MarineGym is fast + currents but
AUV/base-link only; Angler/Stonefish have arms but are not RL-fast. (Full gap analysis:
`docs/design/2026-06-28-hydrodynamics-design.md`.)

## 2. Architecture & design decisions (with rationale — paper §Method)

- **Per-link wrench over the articulation; let Featherstone propagate coupling.** We never
  compute inter-link hydro coupling; it emerges from the rigid-body chain (matches DAVE /
  Kolano 2022). The same per-link path runs over 1 body (AUV) or N (UVMS, snake) with no
  special-casing — topology-agnostic.
- **Isaac-free core behind a Protocol seam.** All physics + assembly logic depends only on
  an `ArticulationView` Protocol and is unit-tested on a CPU fake with **no Isaac
  installed**; a single thin adapter (`apply_isaac.py`) is the only module touching Isaac.
  This is what makes the core testable in CI and portable; it is also a reproducibility
  argument (the force law is verifiable without a GPU cluster).
- **Hybrid added-mass routing (the key numerical decision).** Diagonal added mass →
  **inertia augmentation** (`set_body_inertias` at init; stable, no acceleration estimate).
  Off-diagonal → a small **residual wrench** `−M_A_offdiag · a_filtered` (finite-diff +
  low-pass, α≈0.05–0.1). Anisotropic linear remainder that cannot be a scalar PhysX mass
  also goes to the residual. Rationale: Stonefish's stability *without* its scalar-average
  error; UUV-Sim's accuracy *without* its noisy full-acceleration path.
- **Shape-based hydro for arm links** (cylinder/sphere/box analytic added mass + drag) so
  un-identified manipulator links need no invented coefficients.
- **NWU throughout, scalar-first (w,x,y,z) quaternions**, body-frame wrench/twist. Frame
  conversions at the Isaac boundary are explicit and asserted.
- **Neutrality is expressed by volume (V=m/ρ), never a flag** (decision D1, `DECISIONS.md`).
  Buoyancy is always applied at the center of buoyancy so the CoB↔CoM offset preserves the
  restoring couple. The inverted `neutrally_buoyant` flag we removed would have silently
  dropped that couple — a cautionary example of porting a schema without its semantics.

## 3. Validation methodology (paper §Validation — this is itself a contribution)

**Dual-reference cross-validation.** Every force law is checked twice:
1. An on-CPU analytical Fossen integrator (`validation/reference.py`) vs closed-form
   expectations (e.g. quadratic-drag terminal velocity), and
2. The **in-sim** Isaac run vs that same CPU integrator, body-for-body, same inputs.
Agreement of (2) certifies the *glue* (state read, frame conversion, wrench application,
inertia augmentation), independently of whether the physics is right — (1) certifies the
physics. Separating these two is what makes a failure diagnosable.

**DOF isolation for unit-level force validation.** A free, thrust-driven body couples all
6 DOF, so a "terminal velocity" measurement on a free body is confounded by attitude
dynamics (see Finding A). To validate a single force law we deliberately constrain the
confounding DOF — e.g. drag-terminal **pins attitude** and zeroes the CoB so the surge
drag law is measured in isolation; restoring/coupling dynamics are validated by separate
scenarios that *release* those DOF. State the isolation explicitly per scenario.

**Scenarios** (primitive stand-ins; lighthill coefficients from config, not Isaac geometry):
drag-terminal ✅, free-decay ✅, restoring ✅, and the headline **arm-swing base-reaction
gate** ✅ — base pitch reaction to a commanded arm trajectory vs a floating-base Featherstone +
per-link-hydro reference; this is the UVMS coupling claim, validated at 7.7%.

## 4. Results so far

| Scenario | Metric | Result | Tolerance |
|---|---|---|---|
| drag-terminal (in-sim vs CPU ref) | surge terminal velocity | u_sim 0.95260 vs u_ref 0.94871, **0.41%** | < 5% |
| free-decay (in-sim vs CPU ref) | surge-decay trajectory | max **0.02%** (0.084/0.030/0.013/0.006) | < 5% |
| restoring (in-sim vs CPU ref) | roll(t) oscillation + damping | max **0.07°** over a 30°→0 decay | < 3° |
| arm-swing reaction (the gate) | base pitch reaction vs Featherstone ref | **7.7%** | < 15% |
| CPU core gate | unit tests / coverage | 67 passed, 89.7% | ≥ 78% |

The three single-body scenarios jointly certify translation (drag, steady + transient) and
rotation (restoring couple + angular drag) against the analytical reference; the arm-swing
gate (Task 6) certifies the multi-body coupling that is the headline claim. The arm-swing gate
compares the **free base's pitch reaction** to a commanded arm swing against the floating-base
Featherstone reference (`sim_validation/reference_featherstone.md`), fed the sim's actual
`q(t)`, masses/inertias and joint geometry; it isolates the inertial coupling (gravity/buoyancy/
drag off) and matches to **7.7%** (pitch tracks −0.4/−1.4/−2.1/−1.8° vs ref −0.4/−1.5/−2.2/−2.0°).

(Closed-form anchor for the CPU ref itself, from Plan A: terminal velocity sim 1.118032 vs
√(F/Dq) 1.118034.)

## 5. Findings worth reporting (each ≈ one paragraph in the paper)

**A. A free, thrust-driven slender AUV is directionally (Munk) unstable.** With asymmetric
added mass (M_A,surge ≠ M_A,heave), a body under pure surge thrust and no control develops a
destabilizing Munk moment: any heave perturbation pitches it, which redirects thrust into
more heave — a limit cycle (we observed a steady pitch oscillation, body tumbling, in
Isaac). The idealized symmetric CPU integrator sits on the unstable equilibrium and never
leaves it; the PhysX solver's numerical asymmetry tips it off. Implications: (i) it is
*real physics*, a correctness check that the added-mass Coriolis term is active and
asymmetric; (ii) unit-validating translational drag requires constraining rotation; (iii)
RL/control on these vehicles must contend with open-loop directional instability — relevant
to the I003 control task.

**B. A latent device-placement bug class in GPU-vectorized hydro.** Tensors silently
allocated on CPU (filter state, current buffer, coefficient tensors, index tensors) work in
a CPU test suite but crash — or, worse, silently copy — on the GPU hot path. We found and
fixed **five** independent instances (an `arange` index, the EMA filter state, the current
buffer, the inertia fallbacks, and the coefficient tensors), several only surfaced by the
first real-GPU in-sim run. Lesson for the paper's reproducibility section: a CPU-only test
gate structurally cannot certify a "GPU-resident" claim; either a GPU smoke test or a
device-genericity lint is required. The adversarial per-task + whole-branch review caught
four of five before in-sim; the fifth (coeffs) needed the sim.

**C. Isaac applies external forces less accurately than gravity by default.** With
`PhysxCfg.enable_external_forces_every_iteration=False` (the default), an applied buoyancy
wrench is under-integrated relative to PhysX's internal gravity, leaving a residual net
force (we saw ~0.49 m/s spurious sink on a body that should be neutral). Harmless for a
decoupled-axis terminal-velocity check, but it biases force balance — **must** be enabled
for the coupling-sensitive arm-swing gate. A concrete porting gotcha worth documenting for
anyone applying Fossen wrenches in Isaac Lab.

**E. USD joint `localPos` is interpreted in the *scaled* link frame — a coupling-magnitude
trap.** When a link is authored as a unit primitive plus a non-uniform scale (the standard
`UsdGeom.Cube` + `AddScaleOp` idiom), a `RevoluteJoint`'s `LocalPos0/1` anchors are expressed
in that **scaled** local frame, so the *effective* anchor is `authored_anchor × body_scale`.
For the UVMS gate this turned an authored arm moment arm of 0.40 m into a real 0.17 m, and a
Featherstone reference built on the unscaled anchors over-predicted the base reaction ~2.5×
(gate error ~50–65%). The vehicle↔arm coupling magnitude is set by exactly this moment arm, so
the bug is invisible to single-body validation and only surfaces in the coupling gate. The fix
is the same read-actual-from-sim discipline as inertia parity: feed the reference the scaled
anchors and **assert** them against the arm CoM offset measured from the sim (a `GEOMCHECK`).
Lesson for the paper's reproducibility section: for articulated UVMS authoring, never trust
authored joint frames against a scaled link — verify the realized geometry. (Diagnosis was
clean because the sim itself was correct: it conserved momentum — base recoil velocity matched
`m0 v0 = −m1 v1` — so the discrepancy had to be in the reference's geometry, not the physics.)

**D. Wrench-API frame + lifecycle specifics (Isaac Lab 2.3).**
`set_external_force_and_torque(forces, torques, …, is_global=False)` interprets forces in
the **body/link-local** frame by default; `write_data_to_sim()` must be called before
`sim.step()`; the buffer **persists** across steps until overwritten. The call is flagged
for future deprecation toward `permanent_wrench_composer.set_forces_and_torques` — pin the
version. (Full spike: `docs/isaac-api-findings.md`.) Headless Kit reliably hangs on
`simulation_app.close()`; standalone scripts force-exit.

## 6. Reproducibility

- **Stack:** Isaac Sim 5.1.0, Isaac Lab 2.3.2.post1, torch 2.7.0+cu128, CUDA 12.8, RTX 4060
  Ti (sm_89). Local dev (laptop) for the CPU core; in-sim validation on the same GPU.
- **Gate:** `just check` — ruff + mypy + pytest, coverage floor 78%. Isaac-only modules
  excluded from coverage; `real_sim`-marked tests opt-in via `LIGHTHILL_REAL_SIM_OK=1`.
- **Provenance:** frozen experiment runs pin the lighthill commit; the tool is the source of
  truth, deployments are downstream.

---

## Running log

- **2026-06-30 (Task 6)** — **Arm-swing coupling gate PASSED at 7.7%** — the headline UVMS
  claim is validated. Built the floating-base 2-body Featherstone CPU reference
  (`validation/reference_coupled.py`): the base-DOF accelerations solve from the system
  composite spatial inertia about the base CoM (symmetric by construction), with the prescribed
  joint acceleration as a known bias; certified by 10 CPU tests incl. a momentum-conservation +
  first-order-in-dt convergence proof. The in-sim gate (`arm_swing_reaction.py`) first failed at
  ~50–65%; systematic debugging found the sim was correct (it conserves momentum) and the bug
  was the reference's joint geometry — **USD `localPos` is scaled by body-scale** (Finding E),
  giving a 0.40 m vs real 0.17 m moment arm. Fixed by feeding scaled anchors + a `GEOMCHECK`.
  Gate isolates the inertial coupling (gravity/buoyancy/drag off) and keys on base pitch.
- **2026-06-30** — Isaac phase start. Spike pinned the read/wrench/inertia/step API on the
  live install (§5D, findings doc). Built `IsaacArticulationView` adapter. Fixed the coeff
  device-placement bug (Finding B, 5th instance) found by the first in-sim run. **drag-terminal
  in-sim PASSED at 0.41%** vs the CPU reference — first end-to-end proof of the glue on GPU.
  Discovered + characterized the Munk directional instability (Finding A) while debugging why
  a free body wouldn't hold surge. Noted the Isaac external-force-accuracy gotcha (Finding C).
- **(earlier, Plan A/B CPU core)** — Removed the inverted `neutrally_buoyant` flag (D1).
  Built + reviewed the Isaac-free engine (Protocol seam, added-mass routing, accel filter,
  orchestration); whole-branch reviewed; PR #3. Device-pinning bug class (Finding B,
  instances 1–4) caught by the review loop.
