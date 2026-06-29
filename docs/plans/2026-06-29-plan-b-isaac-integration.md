# Plan B — Isaac Lab Integration + In-Sim Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the Plan A physics core into Isaac Lab — an `UnderwaterHydrodynamics` component that reads an `Articulation`'s per-body state each physics step, applies the per-link Fossen wrenches, augments inertias for the diagonal added mass, and pushes the off-diagonal added mass through a filtered residual wrench — then prove the coupled dynamics against the analytical Fossen+Featherstone reference, culminating in the arm-swing reaction gate.

**Architecture:** A dependency-injected articulation interface (`ArticulationView` Protocol) lets the assembly, added-mass split, and acceleration-filter logic be unit-tested on a CPU fake with **no Isaac installed**; only a thin adapter and the in-sim validation scripts require a real Isaac Lab + GPU on the HPC. Per-link wrenches are applied via Isaac Lab's composable wrench API; the Featherstone articulation solver propagates the vehicle↔arm coupling (we do not compute inter-link hydro coupling). Diagonal added mass folds into effective mass/inertia at init; off-diagonal added mass becomes a small residual wrench from a low-pass-filtered acceleration estimate.

**Tech Stack:** Everything from Plan A, plus NVIDIA Isaac Sim + **Isaac Lab** (HPC, GPU). CPU-testable tasks use a fake `ArticulationView`; in-sim tasks run under a `real_sim` pytest marker (opt-in via `LIGHTHILL_REAL_SIM_OK=1`) or as standalone Isaac scripts.

## Global Constraints

- **Plan A is a hard dependency.** This plan consumes `forces`, `coefficients`, `current`, `frames`, `constants` exactly as Plan A defined them. Do not modify Plan A signatures; if a change is unavoidable, it is a Plan A change with its own test cycle.
- **Frame: NWU throughout.** Same wrench `[...,6]=[F(3),M(3)]` and twist `[...,6]=[u,v,w,p,q,r]` body-frame conventions as Plan A. The Isaac wrench API may want world-frame or link-frame wrenches — convert explicitly in the adapter and assert the convention in a test.
- **No CPU round-trips in the hot path.** `apply()` operates on GPU tensors end-to-end; the only `.cpu()`/`.item()` allowed is in init-time setup and validation logging.
- **The arm-swing reaction gate is a hard gate.** Per the spec: *do not proceed past a failing arm-swing test* — wrong coupling invalidates the science. Plan B is not "done" until Task 6 passes within tolerance.
- **Isaac-touching code is isolated behind `ArticulationView`.** Anything that imports `isaaclab`/`isaacsim`/`omni` lives in `apply_isaac.py` (the adapter) or under `sim_validation/`. Core logic imports only the Protocol.
- **Every CPU-testable task ends green through `just check`.** In-sim tasks (5, 6) run on the HPC and are excluded from the CI gate via the `real_sim` marker.

---

## File Structure

```
src/lighthill/
  articulation.py    NEW  ArticulationView Protocol (the interface apply.py depends on) + a CPU FakeArticulation for tests
  inertia.py         NEW  split a 6x6 added-mass matrix into (mass bump, inertia-tensor augmentation, off-diagonal residual matrix)
  accel.py           NEW  per-body acceleration estimator (finite-diff) + low-pass filter (alpha)
  apply.py           NEW  UnderwaterHydrodynamics: orchestrates read->kernels->wrench, Isaac-free (depends on ArticulationView)
  apply_isaac.py     NEW  IsaacArticulationView adapter (the ONLY core module importing isaaclab); thin
tests/
  test_inertia.py  test_accel.py  test_apply.py   (CPU, fake articulation)
  test_apply_isaac_marker.py                        (real_sim marker; skipped off-HPC)
sim_validation/      NEW  standalone Isaac scripts (run on HPC, not in the pytest gate)
  README.md  free_decay.py  drag_terminal.py  restoring.py  arm_swing_reaction.py
docs/
  isaac-api-findings.md   NEW  Task 1 spike output (pinned API calls + decisions)
```

The `real_sim` marker must be registered. Add to `pyproject.toml` `[tool.pytest.ini_options]` (the scaffold already left a commented stub):
```toml
markers = [
    "real_sim: requires a real Isaac Sim + GPU. Skipped without LIGHTHILL_REAL_SIM_OK=1.",
]
```

---

### Task 1: Isaac Lab API spike (`docs/isaac-api-findings.md`, `articulation.py` Protocol)

The spec lists four "resolve at implementation" unknowns. Resolve them against the live API **before** building the adapter, and pin the answers. This task's deliverable is a findings doc + the `ArticulationView` Protocol whose shape is dictated by what the live API actually offers.

**Files:**
- Create: `docs/isaac-api-findings.md`
- Create: `src/lighthill/articulation.py` (Protocol + CPU fake)
- Test: `tests/test_articulation_fake.py`

**Spike questions to answer in the findings doc (run a throwaway Isaac Lab script on the HPC):**
1. **Read API:** the exact `Articulation` calls for per-body world pose (position + quaternion, and the quaternion order) and per-body linear+angular velocity. Record tensor shapes `(num_envs, num_bodies, …)` and the quaternion convention (confirm `w,x,y,z`).
2. **Per-link wrench API:** the current composable per-body external-wrench call (NOT the deprecated `set_external_force_and_torque`). Record its name, the frame it expects (world vs link), and whether forces+torques go in one call. Capture a minimal working snippet that pushes a constant per-body force and visibly moves the bodies.
3. **`set_body_inertias` semantics:** whether body mass and inertia tensor can be set at init and whether runtime updates are honored. **Critical sub-question:** PhysX body mass is a *scalar* — anisotropic *linear* added mass (different per axis) cannot be a scalar mass. Decide the routing (see Task 2): isotropic linear part → mass bump; angular part → inertia tensor; anisotropic linear remainder + all off-diagonal → residual wrench.
4. **Step hook:** the correct place to run `apply()` each step (`pre_physics_step` vs a physics callback) and how to get `dt`.

- [ ] **Step 1: Run the spike on the HPC and write `docs/isaac-api-findings.md`**

Write the doc with: each question, the confirmed call/signature, the captured snippet, and the decision. This is a real artifact other contributors read — not a scratch note. Include the resolved added-mass routing decision explicitly.

- [ ] **Step 2: Write the failing test for the fake articulation** (`tests/test_articulation_fake.py`)

```python
import torch
from lighthill.articulation import FakeArticulation


def test_fake_round_trips_state_and_records_wrenches():
    art = FakeArticulation(num_envs=2, num_bodies=3)
    pos, quat, vel = art.body_states()
    assert pos.shape == (2, 3, 3)
    assert quat.shape == (2, 3, 4)
    assert vel.shape == (2, 3, 6)
    w = torch.zeros(2, 3, 6)
    w[..., 2] = 1.0
    art.set_external_wrench(w)
    assert torch.allclose(art.last_wrench, w)


def test_fake_lets_tests_set_state():
    art = FakeArticulation(num_envs=1, num_bodies=1)
    art.set_body_velocity(torch.tensor([[[1.0, 0, 0, 0, 0, 0]]]))
    _, _, vel = art.body_states()
    assert vel[0, 0, 0] == 1.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_articulation_fake.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lighthill.articulation'`.

- [ ] **Step 4: Implement `articulation.py`** (Protocol + CPU fake; shape per spike findings)

```python
"""The articulation interface apply.py depends on, plus a CPU fake for tests.

The real Isaac Lab adapter (apply_isaac.py) implements this Protocol; the fake
lets all assembly logic be tested without Isaac installed."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor


class ArticulationView(Protocol):
    """Per-body state read + per-body wrench/inertia write. Shapes: (num_envs, num_bodies, ...)."""

    num_envs: int
    num_bodies: int

    def body_states(self) -> tuple[Tensor, Tensor, Tensor]:
        """(pos [E,B,3] world, quat [E,B,4] wxyz body->world, vel [E,B,6] body twist)."""
        ...

    def set_external_wrench(self, wrench_world: Tensor) -> None:
        """Apply per-body external wrench [E,B,6] = [F(3), M(3)] (frame per adapter)."""
        ...

    def set_body_inertias(self, mass: Tensor, inertia_diag: Tensor) -> None:
        """Set per-body scalar mass [E,B] and principal inertia [E,B,3] (init-time)."""
        ...


class FakeArticulation:
    """In-memory CPU stand-in for tests. Records wrenches; lets tests set state."""

    def __init__(self, num_envs: int, num_bodies: int) -> None:
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self._pos = torch.zeros(num_envs, num_bodies, 3)
        self._quat = torch.zeros(num_envs, num_bodies, 4)
        self._quat[..., 0] = 1.0
        self._vel = torch.zeros(num_envs, num_bodies, 6)
        self.last_wrench = torch.zeros(num_envs, num_bodies, 6)
        self.mass = torch.ones(num_envs, num_bodies)
        self.inertia_diag = torch.ones(num_envs, num_bodies, 3)

    def body_states(self) -> tuple[Tensor, Tensor, Tensor]:
        return self._pos, self._quat, self._vel

    def set_external_wrench(self, wrench_world: Tensor) -> None:
        self.last_wrench = wrench_world.clone()

    def set_body_inertias(self, mass: Tensor, inertia_diag: Tensor) -> None:
        self.mass = mass.clone()
        self.inertia_diag = inertia_diag.clone()

    # test helpers
    def set_body_velocity(self, vel: Tensor) -> None:
        self._vel = vel.clone()

    def set_body_quat(self, quat: Tensor) -> None:
        self._quat = quat.clone()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_articulation_fake.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add docs/isaac-api-findings.md src/lighthill/articulation.py tests/test_articulation_fake.py
git commit -m "feat(articulation): ArticulationView Protocol + CPU fake; Isaac API findings"
```

---

### Task 2: Added-mass routing split (`inertia.py`)

Split each link's 6×6 added-mass matrix into the three destinations the spike decided: an isotropic **mass bump**, an **inertia-tensor augmentation** (angular diagonal), and an **off-diagonal residual matrix** (everything that cannot be a scalar mass / principal inertia). Pure, fully CPU-testable.

**Files:**
- Create: `src/lighthill/inertia.py`
- Test: `tests/test_inertia.py`

**Interfaces:**
- Produces:
  - `@dataclass class AddedMassRouting` — `mass_bump: Tensor [N]`, `inertia_bump: Tensor [N,3]`, `residual: Tensor [N,6,6]`
  - `split_added_mass(added_mass: Tensor) -> AddedMassRouting` where `added_mass` is `[N,6,6]`:
    - `mass_bump[i]` = min of the three linear-diagonal entries (the isotropic part that is safe as a scalar mass)
    - `inertia_bump[i]` = the three angular-diagonal entries `diag[3:6]`
    - `residual` = `added_mass` minus the part accounted for by mass_bump (on the linear diagonal) and inertia_bump (on the angular diagonal); i.e. the anisotropic linear remainder + all off-diagonal terms
  - `effective_inertia(rigid_mass: Tensor, rigid_inertia: Tensor, routing: AddedMassRouting) -> tuple[Tensor, Tensor]` → `(mass+mass_bump, inertia+inertia_bump)` for `set_body_inertias`

- [ ] **Step 1: Write the failing test** (`tests/test_inertia.py`)

```python
import torch
from lighthill.inertia import split_added_mass, effective_inertia


def test_isotropic_linear_added_mass_goes_entirely_to_mass_bump():
    M = torch.diag(torch.tensor([5.0, 5.0, 5.0, 0.2, 0.3, 0.4])).unsqueeze(0)
    r = split_added_mass(M)
    assert torch.isclose(r.mass_bump[0], torch.tensor(5.0))
    assert torch.allclose(r.inertia_bump[0], torch.tensor([0.2, 0.3, 0.4]))
    # nothing left on the linear diagonal of the residual
    assert torch.allclose(torch.diagonal(r.residual[0])[:3], torch.zeros(3), atol=1e-6)


def test_anisotropic_linear_remainder_goes_to_residual():
    M = torch.diag(torch.tensor([6.0, 7.0, 18.0, 0.1, 0.1, 0.1])).unsqueeze(0)
    r = split_added_mass(M)
    assert torch.isclose(r.mass_bump[0], torch.tensor(6.0))  # min of 6,7,18
    diag_res = torch.diagonal(r.residual[0])[:3]
    assert torch.allclose(diag_res, torch.tensor([0.0, 1.0, 12.0]), atol=1e-6)


def test_off_diagonal_preserved_in_residual():
    M = torch.zeros(1, 6, 6)
    M[0, 0, 4] = 2.0
    M[0, 4, 0] = 2.0
    M[0, range(6), range(6)] = torch.tensor([3.0, 3.0, 3.0, 0.0, 0.0, 0.0])
    r = split_added_mass(M)
    assert r.residual[0, 0, 4] == 2.0


def test_effective_inertia_adds_bumps():
    M = torch.diag(torch.tensor([5.0, 5.0, 5.0, 0.2, 0.2, 0.2])).unsqueeze(0)
    r = split_added_mass(M)
    mass = torch.tensor([10.0])
    inertia = torch.tensor([[1.0, 1.0, 1.0]])
    m_eff, i_eff = effective_inertia(mass, inertia, r)
    assert torch.isclose(m_eff[0], torch.tensor(15.0))
    assert torch.allclose(i_eff[0], torch.tensor([1.2, 1.2, 1.2]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inertia.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lighthill.inertia'`.

- [ ] **Step 3: Implement `inertia.py`**

```python
"""Route a 6x6 added-mass matrix to mass bump / inertia bump / residual wrench."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class AddedMassRouting:
    mass_bump: Tensor      # [N] isotropic scalar mass addition
    inertia_bump: Tensor   # [N,3] principal inertia addition
    residual: Tensor       # [N,6,6] anisotropic linear remainder + off-diagonal


def split_added_mass(added_mass: Tensor) -> AddedMassRouting:
    n = added_mass.shape[0]
    diag = torch.diagonal(added_mass, dim1=-2, dim2=-1)  # [N,6]
    lin_diag = diag[:, 0:3]
    ang_diag = diag[:, 3:6]
    mass_bump = lin_diag.min(dim=-1).values  # isotropic safe part
    inertia_bump = ang_diag.clone()
    residual = added_mass.clone()
    idx = torch.arange(6)
    # zero the angular diagonal (moved to inertia) and subtract the isotropic mass on linear diagonal
    residual[:, idx, idx] = 0.0
    # restore the anisotropic linear remainder on the linear diagonal
    remainder = lin_diag - mass_bump.unsqueeze(-1)
    for k in range(3):
        residual[:, k, k] = remainder[:, k]
    return AddedMassRouting(mass_bump=mass_bump, inertia_bump=inertia_bump, residual=residual)


def effective_inertia(rigid_mass: Tensor, rigid_inertia: Tensor,
                      routing: AddedMassRouting) -> tuple[Tensor, Tensor]:
    return rigid_mass + routing.mass_bump, rigid_inertia + routing.inertia_bump
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_inertia.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/lighthill/inertia.py tests/test_inertia.py
git commit -m "feat(inertia): route added mass to mass/inertia bumps + off-diagonal residual"
```

---

### Task 3: Acceleration estimate + low-pass filter (`accel.py`)

The off-diagonal residual needs body acceleration; the spec mandates finite-diff + low-pass (α≈0.05–0.1) to tame noise. Pure, CPU-testable.

**Files:**
- Create: `src/lighthill/accel.py`
- Test: `tests/test_accel.py`

**Interfaces:**
- Produces:
  - `class AccelerationFilter` — `__init__(self, shape: tuple[int,...], alpha: float = 0.08)`; `update(self, twist: Tensor, dt: float) -> Tensor` returns filtered body acceleration `[...,6]`. First call returns zeros (no previous sample). EMA: `a_filt = (1-α)·a_filt + α·a_raw`, `a_raw = (twist − prev_twist)/dt`.
  - `reset(self, mask: Tensor | None = None) -> None` — clear state (per-env reset; mask `[E]` bool).

- [ ] **Step 1: Write the failing test** (`tests/test_accel.py`)

```python
import torch
from lighthill.accel import AccelerationFilter


def test_first_update_returns_zero():
    f = AccelerationFilter(shape=(2, 3), alpha=0.1)
    a = f.update(torch.ones(2, 3, 6), dt=0.01)
    assert torch.allclose(a, torch.zeros(2, 3, 6))


def test_constant_acceleration_is_tracked_after_warmup():
    f = AccelerationFilter(shape=(1, 1), alpha=0.5)
    dt = 0.01
    twist = torch.zeros(1, 1, 6)
    a = torch.zeros(1, 1, 6)
    for k in range(200):
        twist = twist + 0.02  # constant accel of 2.0 per axis (0.02/0.01)
        a = f.update(twist, dt)
    assert torch.allclose(a, torch.full((1, 1, 6), 2.0), atol=0.05)


def test_low_alpha_attenuates_single_spike():
    f = AccelerationFilter(shape=(1, 1), alpha=0.08)
    f.update(torch.zeros(1, 1, 6), dt=0.01)            # seed prev
    spike = torch.zeros(1, 1, 6)
    spike[..., 0] = 1.0                                  # one big jump
    a = f.update(spike, dt=0.01)
    raw = 1.0 / 0.01                                     # 100
    assert a[0, 0, 0] < 0.2 * raw                        # heavily attenuated


def test_reset_clears_state():
    f = AccelerationFilter(shape=(2, 1), alpha=0.5)
    f.update(torch.ones(2, 1, 6), dt=0.01)
    f.reset()
    a = f.update(torch.ones(2, 1, 6), dt=0.01)
    assert torch.allclose(a, torch.zeros(2, 1, 6))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_accel.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lighthill.accel'`.

- [ ] **Step 3: Implement `accel.py`**

```python
"""Finite-difference + EMA low-pass body-acceleration estimator for the residual term."""

from __future__ import annotations

import torch
from torch import Tensor


class AccelerationFilter:
    def __init__(self, shape: tuple[int, ...], alpha: float = 0.08) -> None:
        self.alpha = alpha
        self._prev_twist: Tensor | None = None
        self._a_filt = torch.zeros(*shape, 6)

    def update(self, twist: Tensor, dt: float) -> Tensor:
        if self._prev_twist is None:
            self._prev_twist = twist.clone()
            return self._a_filt
        a_raw = (twist - self._prev_twist) / dt
        self._a_filt = (1.0 - self.alpha) * self._a_filt + self.alpha * a_raw
        self._prev_twist = twist.clone()
        return self._a_filt

    def reset(self, mask: Tensor | None = None) -> None:
        if mask is None:
            self._prev_twist = None
            self._a_filt = torch.zeros_like(self._a_filt)
            return
        self._a_filt[mask] = 0.0
        if self._prev_twist is not None:
            self._prev_twist[mask] = 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_accel.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/lighthill/accel.py tests/test_accel.py
git commit -m "feat(accel): finite-diff + low-pass acceleration estimator"
```

---

### Task 4: `UnderwaterHydrodynamics` orchestration (`apply.py`)

The component that ties it together — Isaac-free, depends only on `ArticulationView`. Tested end-to-end on the CPU fake.

**Files:**
- Create: `src/lighthill/apply.py`
- Test: `tests/test_apply.py`

**Interfaces:**
- Consumes: `ArticulationView`, `ResolvedCoefficients` (Plan A), `forces.*`, `current.relative_velocity`, `inertia.*`, `accel.AccelerationFilter`, `frames`
- Produces:
  - `class UnderwaterHydrodynamics`:
    - `__init__(self, view: ArticulationView, coeffs: ResolvedCoefficients, *, current: CurrentField | None = None, alpha: float = 0.08)` — splits added mass, sets effective inertias on the view once, allocates the accel filter and per-env current buffer.
    - `reset(self, current_world: Tensor | None = None) -> None` — sample/seed the current, reset the filter.
    - `compute_wrench(self, dt: float) -> Tensor` — read state, build per-link body-frame wrench `[E,B,6]` = buoyancy + drag + added-mass Coriolis + off-diagonal residual; returns body-frame wrench (the adapter converts frames).
    - `apply(self, dt: float) -> None` — `compute_wrench` then `view.set_external_wrench(world_wrench)` (convert body→world here using body quats).
- Behavior: buoyancy/drag/Coriolis use `relative_velocity` against the current; residual uses the filtered acceleration and `routing.residual`.

- [ ] **Step 1: Write the failing test** (`tests/test_apply.py`)

```python
import torch
from lighthill.articulation import FakeArticulation
from lighthill.config import RobotHydroConfig
from lighthill.coefficients import resolve_coefficients
from lighthill.apply import UnderwaterHydrodynamics
from lighthill import example_config_path


def _auv_coeffs():
    cfg = RobotHydroConfig.from_yaml(example_config_path("bluerov2_auv.yaml"))
    return resolve_coefficients(cfg)


def test_inertias_are_augmented_at_init():
    coeffs = _auv_coeffs()
    art = FakeArticulation(num_envs=2, num_bodies=1)
    base_mass = art.mass.clone()
    UnderwaterHydrodynamics(art, coeffs)
    assert (art.mass > base_mass).all()  # mass bumped by isotropic added mass


def test_buoyant_body_gets_upward_world_wrench():
    coeffs = _auv_coeffs()
    art = FakeArticulation(num_envs=1, num_bodies=1)
    hydro = UnderwaterHydrodynamics(art, coeffs)
    hydro.reset(current_world=torch.zeros(1, 3))
    hydro.apply(dt=0.01)
    # positive-buoyancy base_link -> +Z world force recorded on the body
    assert art.last_wrench[0, 0, 2] > 0


def test_drag_opposes_forward_motion():
    coeffs = _auv_coeffs()
    art = FakeArticulation(num_envs=1, num_bodies=1)
    art.set_body_velocity(torch.tensor([[[2.0, 0, 0, 0, 0, 0]]]))  # surge +x
    hydro = UnderwaterHydrodynamics(art, coeffs)
    hydro.reset(current_world=torch.zeros(1, 3))
    w = hydro.compute_wrench(dt=0.01)
    assert w[0, 0, 0] < 0  # body-frame surge drag opposes motion


def test_wrench_shape_matches_bodies():
    cfg = RobotHydroConfig.from_yaml(example_config_path("bluerov2_alpha_uvms.yaml"))
    coeffs = resolve_coefficients(cfg)
    nb = len(cfg.links)
    art = FakeArticulation(num_envs=4, num_bodies=nb)
    hydro = UnderwaterHydrodynamics(art, coeffs)
    hydro.reset()
    w = hydro.compute_wrench(dt=0.01)
    assert w.shape == (4, nb, 6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_apply.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lighthill.apply'`.

- [ ] **Step 3: Implement `apply.py`**

```python
"""UnderwaterHydrodynamics: per-link Fossen wrench over an articulation. Isaac-free."""

from __future__ import annotations

import torch
from torch import Tensor

from .accel import AccelerationFilter
from .articulation import ArticulationView
from .coefficients import ResolvedCoefficients
from .current import CurrentField, relative_velocity
from .forces import added_mass_coriolis, added_mass_residual, buoyancy_wrench, drag_wrench
from .frames import quat_to_rotation_matrix
from .inertia import effective_inertia, split_added_mass


class UnderwaterHydrodynamics:
    def __init__(self, view: ArticulationView, coeffs: ResolvedCoefficients, *,
                 current: CurrentField | None = None, alpha: float = 0.08) -> None:
        self.view = view
        self.coeffs = coeffs
        self.current_field = current or CurrentField()
        self.routing = split_added_mass(coeffs.added_mass)  # [B,...]
        self._filter = AccelerationFilter(shape=(view.num_envs, view.num_bodies), alpha=alpha)
        self._current_world = torch.zeros(view.num_envs, 3)
        # augment inertias once (broadcast per-body routing across envs)
        mass0 = view.mass if hasattr(view, "mass") else torch.ones(view.num_envs, view.num_bodies)
        inertia0 = (view.inertia_diag if hasattr(view, "inertia_diag")
                    else torch.ones(view.num_envs, view.num_bodies, 3))
        m_eff, i_eff = effective_inertia(
            mass0, inertia0,
            _broadcast_routing(self.routing, view.num_envs),
        )
        view.set_body_inertias(m_eff, i_eff)

    def reset(self, current_world: Tensor | None = None) -> None:
        if current_world is not None:
            self._current_world = current_world
        else:
            self._current_world = self.current_field.sample(self.view.num_envs)
        self._filter.reset()

    def compute_wrench(self, dt: float) -> Tensor:
        _pos, quat, twist = self.view.body_states()  # [E,B,*]
        cur = self._current_world.unsqueeze(1).expand(-1, self.view.num_bodies, -1)
        v_rel = relative_velocity(twist, quat, cur)
        c = self.coeffs
        buoy = buoyancy_wrench(quat, c.volume, c.center_of_buoyancy,
                               c.neutrally_buoyant, c.density)
        drag = drag_wrench(v_rel, c.linear_damping, c.quadratic_damping)
        cor = added_mass_coriolis(c.added_mass, v_rel)
        a_filt = self._filter.update(twist, dt)
        resid = added_mass_residual(self.routing.residual, a_filt)
        return buoy + drag + cor + resid

    def apply(self, dt: float) -> None:
        w_body = self.compute_wrench(dt)
        quat = self.view.body_states()[1]
        R = quat_to_rotation_matrix(quat)  # [E,B,3,3]
        f_world = (R @ w_body[..., 0:3].unsqueeze(-1)).squeeze(-1)
        m_world = (R @ w_body[..., 3:6].unsqueeze(-1)).squeeze(-1)
        self.view.set_external_wrench(torch.cat([f_world, m_world], dim=-1))


def _broadcast_routing(routing, num_envs: int):
    from .inertia import AddedMassRouting
    return AddedMassRouting(
        mass_bump=routing.mass_bump.unsqueeze(0).expand(num_envs, -1),
        inertia_bump=routing.inertia_bump.unsqueeze(0).expand(num_envs, -1, -1),
        residual=routing.residual,
    )
```

> The buoyancy/drag/Coriolis kernels broadcast over the leading `[E,B]` dims because Plan A wrote them with `[...,6]` / `[...,6,6]` batching. `c.volume [B]`, `c.center_of_buoyancy [B,3]`, etc. broadcast against `quat [E,B,4]` via standard right-aligned broadcasting; if a shape mismatch arises, `unsqueeze(0).expand` the per-body coeffs to `[E,B,...]` in `__init__` once.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_apply.py -v`
Expected: PASS (4 passed). Resolve any broadcasting shape error by pre-expanding coeffs to `[E,B,...]` as noted.

- [ ] **Step 5: Run the full CPU gate**

Run: `just check`
Expected: ruff + mypy clean, all CPU tests pass, coverage ≥78%.

- [ ] **Step 6: Commit**

```bash
git add src/lighthill/apply.py tests/test_apply.py
git commit -m "feat(apply): UnderwaterHydrodynamics orchestration over ArticulationView"
```

---

### Task 5: Isaac adapter + in-sim validation tests 1–3 (HPC)

The thin real-Isaac adapter and the single-body validation scenarios (free-decay, drag-terminal, restoring) re-run **inside Isaac** to confirm the glue matches the Plan A CPU reference.

**Files:**
- Create: `src/lighthill/apply_isaac.py` (the only core module importing `isaaclab`)
- Create: `sim_validation/README.md`, `sim_validation/free_decay.py`, `sim_validation/drag_terminal.py`, `sim_validation/restoring.py`
- Create: `tests/test_apply_isaac_marker.py` (marked `real_sim`)

**Interfaces:**
- Produces: `class IsaacArticulationView` implementing `ArticulationView` using the spike-confirmed Isaac Lab calls (read body states, set external wrench, set body inertias). Frame conversion to the API's expected frame happens here, asserted against the convention recorded in `docs/isaac-api-findings.md`.

- [ ] **Step 1: Implement `apply_isaac.py`** using the pinned API from Task 1. Keep it thin — only state read, wrench write, inertia write, and frame conversion. No physics.

- [ ] **Step 2: Write `sim_validation/drag_terminal.py`** — spawn one BlueROV2 body in Isaac with `UnderwaterHydrodynamics`, apply constant thruster force, record terminal surge speed, compare to `terminal_velocity_quadratic` from Plan A. Print PASS/FAIL with the relative error.

- [ ] **Step 3: Write `sim_validation/free_decay.py` and `restoring.py`** — the displaced-release decay and static-tilt restoring scenarios, each comparing the in-sim trajectory to the Plan A CPU reference (`validation.reference.simulate`) within a stated tolerance.

- [ ] **Step 4: Write `tests/test_apply_isaac_marker.py`**

```python
import os
import pytest

real_sim = pytest.mark.skipif(
    os.environ.get("LIGHTHILL_REAL_SIM_OK") != "1",
    reason="needs a real Isaac Sim + GPU (set LIGHTHILL_REAL_SIM_OK=1 on the HPC)",
)


@real_sim
def test_drag_terminal_velocity_in_sim_matches_reference():
    from sim_validation.drag_terminal import run
    rel_err = run(steps=4000, dt=0.005)
    assert rel_err < 0.05
```

- [ ] **Step 5: Run on the HPC**

Run: `LIGHTHILL_REAL_SIM_OK=1 uv run pytest tests/test_apply_isaac_marker.py -v` (on the cluster, in the Isaac env)
Expected: PASS. Off-HPC, the same test is SKIPPED — confirm `just check` still passes on the laptop (skip, not fail).

- [ ] **Step 6: Commit**

```bash
git add src/lighthill/apply_isaac.py sim_validation/ tests/test_apply_isaac_marker.py pyproject.toml
git commit -m "feat(sim): Isaac adapter + in-sim validation tests 1-3 (free-decay/drag/restoring)"
```

---

### Task 6: The arm-swing reaction gate (HPC) — the crux

The UVMS coupling gate. Command an arm trajectory with the base free; the measured base reaction must match the Featherstone + per-link-hydro reference. **Do not declare Plan B done until this passes.**

**Files:**
- Create: `sim_validation/arm_swing_reaction.py`
- Create: `sim_validation/reference_featherstone.md` (how the reference is computed + the tolerance rationale)
- Modify: `tests/test_apply_isaac_marker.py` (add the marked gate test)

**Interfaces:**
- `sim_validation/arm_swing_reaction.py::run(...) -> dict` returns the measured base reaction (force/displacement time series) and the reference, plus the peak/RMS relative error.

- [ ] **Step 1: Build the reference** — document in `reference_featherstone.md` how the analytical Fossen+Featherstone base reaction to a commanded arm swing is computed (cf. Kolano 2022). Either an offline closed-form for a simple 1-arm-link swing, or the Plan A reference extended to a 2-body chain. State the tolerance and why.

- [ ] **Step 2: Implement `arm_swing_reaction.py`** — load `bluerov2_alpha_uvms.yaml`, import the UVMS articulation, attach `UnderwaterHydrodynamics`, command a deterministic arm trajectory (low/med/high aggressiveness), record base force/pose, compare to the reference.

- [ ] **Step 3: Add the gate test** (`tests/test_apply_isaac_marker.py`)

```python
@real_sim
def test_arm_swing_base_reaction_matches_featherstone_reference():
    from sim_validation.arm_swing_reaction import run
    result = run(aggressiveness="med")
    assert result["peak_rel_error"] < 0.15  # tolerance per reference_featherstone.md
```

- [ ] **Step 4: Run the gate on the HPC**

Run: `LIGHTHILL_REAL_SIM_OK=1 uv run pytest tests/test_apply_isaac_marker.py::test_arm_swing_base_reaction_matches_featherstone_reference -v`
Expected: PASS within tolerance. **If it fails:** do not loosen the tolerance — investigate (frame sign, residual α, wrench frame, inertia augmentation). A failing arm-swing gate means the coupling is wrong; fix the cause. Pilot-tune the residual `α` (0.05–0.1) and `dt` here per the spec's open question.

- [ ] **Step 5: Commit**

```bash
git add sim_validation/arm_swing_reaction.py sim_validation/reference_featherstone.md tests/test_apply_isaac_marker.py
git commit -m "feat(sim): arm-swing base-reaction gate vs Featherstone reference"
```

---

## Self-Review

**Spec coverage** (against `docs/design/2026-06-28-hydrodynamics-design.md`):

| Spec element | Task |
|---|---|
| `apply` — read articulation state, call forces, per-link wrench | Task 4 (logic), Task 5 (Isaac adapter) |
| Per-link wrench via composable API (not deprecated call) | Task 1 (confirm), Task 5 (use) |
| Featherstone propagates coupling (we don't compute inter-link hydro) | Task 4/6 (design honored — wrenches per link only) |
| Added mass: diagonal → inertia augmentation | Task 2 + Task 4 (`set_body_inertias`) |
| Added mass: off-diagonal → filtered residual wrench (α≈0.05–0.1) | Tasks 2, 3, 4 |
| Scalar-mass subtlety (anisotropic linear) | Task 1 (decide) + Task 2 (route to residual) |
| Current per env, randomized at reset, drag/Coriolis only | Task 4 (`reset` samples; uses `relative_velocity`) |
| Validation test 1 free-decay | Task 5 |
| Validation test 2 drag terminal | Task 5 |
| Validation test 3 restoring | Task 5 |
| Validation test 4 arm-swing reaction (the gate) | Task 6 |
| NWU frame conversion at the Isaac boundary, asserted | Task 5 (adapter) |
| Risk: `set_body_inertias` runtime vs init | Task 1 (spike resolves) |
| Risk: residual stability at dt/α | Task 6 (pilot-tune) |
| Risk: articulation read cost at high env counts | Task 5 (note: profile in-sim) |

**Placeholder scan:** The Isaac-touching steps (Tasks 1, 5, 6) describe concrete deliverables with candidate calls and explicit pass criteria, not "TODO". The genuinely unknowable-until-HPC pieces (exact API call names) are resolved by Task 1's spike and pinned in `docs/isaac-api-findings.md` before any adapter code — this is a real de-risking task per the spec's own "resolve at implementation" list, not a deferred placeholder. All CPU tasks (2, 3, 4) have complete code and real assertions.

**Type consistency:** `ArticulationView` (Task 1) is consumed by `UnderwaterHydrodynamics` (Task 4) and implemented by both `FakeArticulation` (Task 1) and `IsaacArticulationView` (Task 5) with identical method signatures. `AddedMassRouting` fields (`mass_bump`, `inertia_bump`, `residual`) match across Tasks 2 and 4. `ResolvedCoefficients` fields are consumed exactly as Plan A defined them. `terminal_velocity_quadratic` (Plan A) is reused in Task 5.

**Cross-plan integrity:** Plan B adds only new modules + the `real_sim` marker; it does not alter any Plan A signature. The CPU gate stays green on the laptop (in-sim tests skip); the full coupling validation runs on the HPC.
