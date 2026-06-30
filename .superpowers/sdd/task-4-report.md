# Task 4 Report: `UnderwaterHydrodynamics` orchestration (`apply.py`)

## Status: COMPLETE

## What was implemented

Created two files:
- `src/lighthill/apply.py` — the `UnderwaterHydrodynamics` class and `_broadcast_routing` helper
- `tests/test_apply.py` — four TDD tests from the brief (import-sorted for ruff I001)

### `UnderwaterHydrodynamics` class

- `__init__`: Calls `split_added_mass` on `coeffs.added_mass [B,6,6]` to get `routing [B,...]`. Pre-expands all per-body coefficient tensors to `[E,B,...]` once (see broadcasting note below). Calls `effective_inertia` with `_broadcast_routing` to bump the view's masses and inertias. Allocates an `AccelerationFilter(shape=(E,B))` and a `_current_world [E,3]` buffer.
- `reset`: Seeds `_current_world` from the passed tensor or from `current_field.sample(num_envs)`, then calls `_filter.reset()` (full global reset).
- `compute_wrench`: Reads `body_states()`, expands `_current_world` to `[E,B,3]`, calls `relative_velocity`, then sums `buoyancy_wrench + drag_wrench + added_mass_coriolis + added_mass_residual`, all using pre-expanded `[E,B,...]` coefficient tensors. Returns `[E,B,6]` body-frame wrench.
- `apply`: Calls `compute_wrench`, rotates force/moment vectors to world frame with `quat_to_rotation_matrix` and matmul (`R @ f_body`), then calls `view.set_external_wrench`.

### `_broadcast_routing` helper

Takes routing `[B,...]` and creates an `AddedMassRouting` with `mass_bump [E,B]`, `inertia_bump [E,B,3]` (expanded via `unsqueeze(0).expand`). `residual` left as `[B,6,6]` — it is stored separately as `self._residual [E,B,6,6]` in `__init__`.

---

## TDD evidence

### RED phase

Command: `uv run pytest tests/test_apply.py -v`

Result:
```
ERROR collecting tests/test_apply.py
  ModuleNotFoundError: No module named 'lighthill.apply'
```
All 4 tests failed to collect — confirmed RED.

### GREEN phase (after implementation)

Command: `uv run pytest tests/test_apply.py -v`

Result: **4 passed** — all tests green.

---

## Broadcasting issue encountered and resolution

The brief warned about `torch.cross` requiring identical number of dimensions. The root mismatch:

- `cob_body` from `coeffs.center_of_buoyancy` is `[B,3]`
- `f_body` produced by `buoyancy_wrench` (after `world_vec_to_body(f_world [B,3], quat [E,B,4])`) is `[E,B,3]`

`torch.cross(cob_body [B,3], f_body [E,B,3])` raises:
`RuntimeError: linalg.cross: inputs must have the same number of dimensions`

Note: `@` (matmul) tolerates this via batch-broadcasting (e.g., `[B,6,6] @ [E,B,6,1]` → `[E,B,6,1]`), but `torch.cross` does not.

**Resolution**: Pre-expanded all per-body coefficient tensors to `[E,B,...]` once in `__init__`, per the brief's guidance:

```python
self._volume    = coeffs.volume.unsqueeze(0).expand(E, B)
self._cob       = coeffs.center_of_buoyancy.unsqueeze(0).expand(E, B, 3)
self._lin_damp  = coeffs.linear_damping.unsqueeze(0).expand(E, B, 6, 6)
self._quad_damp = coeffs.quadratic_damping.unsqueeze(0).expand(E, B, 6, 6)
self._added_mass= coeffs.added_mass.unsqueeze(0).expand(E, B, 6, 6)
self._residual  = self.routing.residual.unsqueeze(0).expand(E, B, 6, 6)
```

`compute_wrench` uses these `_*` attributes instead of `self.coeffs.*` directly.

---

## Import sort fix

The brief's test code had `import torch` followed immediately by `from lighthill...` imports with no blank-line separation. `ruff` raised `I001` (import block unsorted). Applied `ruff check --fix` to insert the blank line between third-party and project imports. This is a cosmetic ruff requirement, not a logic change.

---

## Files changed

- `src/lighthill/apply.py` (new, 76 lines)
- `tests/test_apply.py` (new, 47 lines, import-sorted)

No other files were touched or committed.

---

## `just check` result

```
uv lock --check          OK
ruff check .             All checks passed!
mypy src                 Success: no issues found in 14 source files
pytest                   56 passed, 1 skipped — coverage 90.78% (threshold 78%)
```

Gate: GREEN.

---

## Self-review findings

1. **Correctness — body→world conversion**: `apply` uses `R @ w_body[...,0:3].unsqueeze(-1)` where R is `[E,B,3,3]` (body→world rotation). This is correct: `R @ f_body` gives `f_world`. The test `test_buoyant_body_gets_upward_world_wrench` (identity quaternion → R=I → world wrench = body wrench = +Z) confirms this.

2. **Broadcasting**: All `[B,...]` coefficients pre-expanded to `[E,B,...]` in `__init__`. No hidden shape mismatches remain.

3. **Inertia augmentation**: `_broadcast_routing` creates an `AddedMassRouting` with `[E,B]` mass_bump and `[E,B,3]` inertia_bump. `effective_inertia` adds these to the view's `[E,B]` and `[E,B,3]` inertias — correct.

4. **YAGNI**: No extra abstractions. `_broadcast_routing` is the only helper; it is lean and purposeful.

5. **Naming**: Instance attributes follow the `_` prefix convention for internal state (`_filter`, `_current_world`, `_volume`, etc.). Public API surface matches the brief exactly.

6. **No Isaac imports**: Verified — `apply.py` imports only from within the `lighthill` package and `torch`. No `isaaclab` dependency.

7. **Residual placement**: `routing.residual` is used via `self._residual` (pre-expanded), not `self.routing.residual` directly in `compute_wrench`. Consistent with the expand-once strategy.

---

## Concerns

None. All tests pass, gate is green, broadcast issue was correctly resolved per brief's guidance.

---

## Fix: device + frame test

### Files changed

- `src/lighthill/apply.py` — device-source fix for init-time tensors
- `tests/test_apply.py` — new tilted-body test locking the body→world frame direction

### Change 1 — `apply.py` device fix

Added `_dev = view.body_states()[0].device` immediately after storing `self.view`, then threaded `device=_dev` into:

- `self._current_world = torch.zeros(view.num_envs, 3, device=_dev)`
- `mass0` fallback: `torch.ones(E, B, device=_dev)`
- `inertia0` fallback: `torch.ones(E, B, 3, device=_dev)`

`body_states()[0]` (position tensor) is the canonical, Protocol-safe device source — the `ArticulationView` Protocol guarantees it; `view.mass` is NOT used as it may not exist on the real Isaac adapter.

### Change 2 — `test_apply.py` frame-direction test

Added `test_apply_converts_body_wrench_to_world_frame`. The test uses a 90° roll quaternion `(w=0.707, x=0.707, y=0, z=0)`.

**Axis correction required:** The brief's specified assertions (`w[0] > 0`, `abs(w[2]) < 1e-3`) were wrong. Running the test as specified produced:

```
AssertionError: assert tensor(0.) > 0   # w[0] was 0.0, not positive
```

**Diagnosis:** `buoyancy_wrench` converts world-up `[0,0,F]` into body frame via `R^T`, giving `f_body = [0, F, 0]` for this roll. `apply()` then applies `R @ f_body = R @ R^T @ [0,0,F] = [0,0,F]`. Buoyancy **always emerges as world +z** — the two rotations cancel regardless of body orientation. The brief's comment "body +z maps to world +x" is geometrically incorrect for 90° roll about +x (it maps body +z to world −y, not +x).

**Why this IS still a meaningful test:** If `apply()` used `R^T` (wrong direction) the chain becomes `R^T @ R^T @ [0,0,F] = R_x(−180°) @ [0,0,F] = [0,0,−F]`, making `w[2] < 0` — the test would catch it. The assertion `w[2] > 0` on a non-identity quaternion correctly distinguishes `R` from `R^T` (the identity-quat test cannot, since `I = I^T`).

**Corrected assertions:**
```python
assert w[2] > 0      # world +z carries buoyancy (correct body->world: R, not R.T)
assert abs(w[0]) < 1e-3  # world +x is ~zero (no spurious x-component)
```

### Commands run and output

```
uv run pytest tests/test_apply.py -v
# → 5 passed (after axis correction); 1 failed before correction (w[0] == 0)

just check
# lock-check: OK
# ruff check: All checks passed!
# mypy src:   Success: no issues found in 14 source files
# pytest:     57 passed, 1 skipped — coverage 90.80% (threshold 78%)
# Gate: GREEN
```

### Commit

`8500c89  fix(apply): device-source init tensors for GPU views; lock body->world frame test`

---

## Whole-branch review fixes

### Files changed

- `src/lighthill/apply.py` — Important #1, Important #2, Minor A
- `src/lighthill/articulation.py` — Important #2 (Protocol declaration)
- `src/lighthill/accel.py` — Minor B, Minor C

### Fix: Important #1 — coerce sampled/explicit current to orchestrator's device (`apply.py`)

**Problem:** `CurrentField.sample` allocates CPU tensors. On a CUDA view, `_current_world` would be CPU while `twist`/`quat` are CUDA, causing a device-mismatch crash in `relative_velocity`.

**Change:** Stored `self._device = _dev` in `__init__`. In `reset`, both paths now call `.to(self._device)`:

```python
self._current_world = current_world.to(self._device)          # explicit arg
self._current_world = self.current_field.sample(...).to(self._device)  # default sample
```

`.to(device)` is a no-op when already on the target device, so CPU-only runs are unaffected. `CurrentField` (Plan A) was not touched.

### Fix: Important #2 — honest mass/inertia contract: Protocol declaration + drop silent fallback (`apply.py` + `articulation.py`)

**Problem:** `apply.__init__` read `view.mass`/`view.inertia_diag` via `hasattr` guards with silent `torch.ones` fallbacks. Neither attribute was declared on the `ArticulationView` Protocol. A real Isaac adapter that didn't expose those exact names would silently set every link's mass/inertia to 1.0 with no error — physically wrong augmented inertias.

**Changes:**

In `articulation.py`, added two required read-attribute annotations to `ArticulationView`:

```python
mass: Tensor          # [E,B] per-body rigid mass (read attribute)
inertia_diag: Tensor  # [E,B,3] per-body principal rigid inertia (read attribute)
```

`FakeArticulation` already exposes both (`self.mass = torch.ones(...)` and `self.inertia_diag = torch.ones(...)`), so it continues to satisfy the Protocol — confirmed by reading the class.

In `apply.__init__`, dropped the `hasattr` guards and fallback `torch.ones` tensors:

```python
mass0 = view.mass
inertia0 = view.inertia_diag
```

### Fix: Minor A — hoist `AddedMassRouting` import + annotate `_broadcast_routing` (`apply.py`)

`AddedMassRouting` was imported inside the `_broadcast_routing` function body despite `inertia` already being imported at module top. Hoisted to the module-level import line and added full type annotation to the helper:

```python
from .inertia import AddedMassRouting, effective_inertia, split_added_mass
...
def _broadcast_routing(routing: AddedMassRouting, num_envs: int) -> AddedMassRouting:
```

### Fix: Minor B — document `dt` contract in `update` (`accel.py`)

Added a short docstring to `AccelerationFilter.update` noting that `dt` must be > 0 (zero produces `inf` accelerations; the sim step is always positive so this is a caller error, not a runtime branch).

### Fix: Minor C — document mask device coupling in `reset` (`accel.py`)

Appended one sentence to the existing `reset` docstring noting that a provided `mask` must reside on the same device as `_a_filt`/`_prev_twist`; a device mismatch raises at the index-assignment step.

---

### Commands run and output

```
uv run pytest tests/test_apply.py tests/test_articulation_fake.py tests/test_accel.py -v
# → 14 passed, 1 skipped (CUDA device test, expected on CPU-only host)

just check
# uv lock --check:      OK
# ruff check .:         All checks passed!
# mypy src:             Success: no issues found in 14 source files
# pytest (full suite):  57 passed, 1 skipped — coverage 90.84% (threshold 78%)
# Gate: GREEN
```

### Commit

`c90da01  fix(apply): coerce current to device; declare mass/inertia on Protocol, drop silent ones-fallback`
