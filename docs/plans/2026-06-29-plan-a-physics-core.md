# Plan A — Hydrodynamics Physics Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Isaac-free underwater hydrodynamics core for `lighthill` — config schema, coefficient resolution (explicit + shape-based), the per-link Fossen force kernels (buoyancy/restoring, drag, added-mass Coriolis + residual), the current model, and a numerical validation harness that proves the physics against closed-form references — all runnable and tested on a laptop with CPU torch.

**Architecture:** Pure batched **torch** functions, body-frame convention throughout, vectorized over `(num_links, 6)` (and broadcastable to `(num_envs, num_links, 6)`). No Isaac/Isaac-Sim import anywhere in this plan — the kernels take plain tensors in and return wrench tensors out, so they are unit- and property-testable on CPU. A minimal in-house single-rigid-body Fossen integrator drives the validation scenarios, so the science is de-risked before the Isaac Lab glue (Plan B) exists.

**Tech Stack:** Python ≥3.10 (3.12 dev), PyTorch (CPU build for tests), PyYAML (config), pytest + hypothesis + coverage (`just check` gate: ruff, mypy, pytest@78% cover). Package layout `src/lighthill/`, tests in `tests/`, `--import-mode=importlib`.

## Global Constraints

- **Frame: NWU throughout** (Isaac/PhysX convention). Up = +Z. Gravity acts −Z. Buoyancy acts +Z. Fossen's equations are NED — every sign that comes from a Fossen source must be translated to NWU explicitly and asserted in a test.
- **Wrench convention:** every force kernel returns a body-frame wrench `[..., 6]` ordered `[Fx, Fy, Fz, Mx, My, Mz]`. Body twist input is `[..., 6]` ordered `[u, v, w, p, q, r]` (body-frame linear then angular). This ordering is fixed for the whole package.
- **No Isaac import in Plan A.** Nothing under `src/lighthill/` in this plan may `import isaaclab`/`isaacsim`/`omni`. The kernels are pure tensor math.
- **Tensors are torch, dtype `float32` default, device-agnostic** (tests run on CPU). Never call `.cpu()`/`.numpy()` inside a kernel (Plan B runs these on GPU; round-trips are forbidden).
- **Constants:** `RHO_SEAWATER = 1025.0` kg/m³ (default), `GRAVITY = 9.81` m/s². Density is configurable per call.
- **Quaternion convention: `(w, x, y, z)`, scalar-first** (Isaac Lab convention), representing body→world orientation.
- **Every task ends green through `just check`** (ruff + mypy + pytest@≥78% branch coverage). Steps below run the focused test; the task's final commit assumes `just check` passes.

---

## File Structure

```
src/lighthill/
  __init__.py          (exists) → export the public API at the end (Task 9)
  constants.py         NEW  physical constants + the canonical 6-vector index slices
  frames.py            NEW  quaternion→rotation-matrix, world↔body transforms (NWU)
  config.py            NEW  LinkConfig/RobotHydroConfig dataclasses + YAML loader + validation
  coefficients.py      NEW  resolve configs → batched coefficient tensors; shape models
  forces.py            NEW  buoyancy, drag, added_mass_coriolis, added_mass_residual (pure)
  current.py           NEW  CurrentField + relative-velocity
  configs/             NEW  shipped example configs (assets)
    bluerov2_auv.yaml        AUV-only BlueROV2
    bluerov2_alpha_uvms.yaml BlueROV2 + cylinder-modeled Reach Alpha arm
  validation/
    __init__.py        NEW
    reference.py       NEW  single-rigid-body Fossen integrator + closed-form expectations
tests/
  test_frames.py       test_config.py       test_coefficients.py
  test_forces_buoyancy_drag.py   test_forces_addedmass.py
  test_current.py      test_validation.py    test_example_configs.py
```

Each file has one responsibility: `frames` = rotations, `config` = schema/IO, `coefficients` = parameter resolution + shapes, `forces` = the force law, `current` = flow, `validation` = the reference integrator. They depend forward-only: `frames → forces`, `coefficients → forces`, `config → coefficients`, everything → `validation`.

---

### Task 1: Constants + frame utilities (`constants.py`, `frames.py`)

Foundational tensor utilities every later task uses. This task also adds the runtime dependencies.

**Files:**
- Modify: `pyproject.toml` (add `torch`, `pyyaml` to `dependencies`; add `types-PyYAML` to dev group; note torch now present in test env)
- Create: `src/lighthill/constants.py`
- Create: `src/lighthill/frames.py`
- Test: `tests/test_frames.py`

**Interfaces:**
- Produces:
  - `constants.RHO_SEAWATER: float`, `constants.GRAVITY: float`
  - `constants.LIN = slice(0, 3)`, `constants.ANG = slice(3, 6)` (index the 6-vector)
  - `frames.quat_to_rotation_matrix(quat: Tensor) -> Tensor` — `quat [...,4]` (w,x,y,z) → `R [...,3,3]` (body→world)
  - `frames.world_vec_to_body(vec_world: Tensor, quat_wb: Tensor) -> Tensor` — `[...,3] → [...,3]`, computes `Rᵀ @ v`
  - `frames.skew(v: Tensor) -> Tensor` — `[...,3] → [...,3,3]` skew-symmetric matrix

- [ ] **Step 1: Add dependencies**

Edit `pyproject.toml`: change `dependencies = []` to:

```toml
dependencies = [
    "torch>=2.2",
    "pyyaml>=6.0",
]
```

In `[dependency-groups].dev` add `"types-PyYAML>=6.0"`. The mypy override block already lists `torch` under `ignore_missing_imports`; leave it (harmless when torch is installed). Run `uv sync` so torch (CPU) lands in the env.

- [ ] **Step 2: Write the failing test** (`tests/test_frames.py`)

```python
import math
import torch
from lighthill import constants, frames


def test_identity_quat_gives_identity_matrix():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # (w,x,y,z)
    R = frames.quat_to_rotation_matrix(q)
    assert torch.allclose(R, torch.eye(3).unsqueeze(0), atol=1e-6)


def test_90deg_about_z_rotates_x_to_y():
    # +90° about world/body z: body-x axis maps to world +y
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    q = torch.tensor([[c, 0.0, 0.0, s]])  # rotation pi/2 about z
    R = frames.quat_to_rotation_matrix(q)
    x_body = torch.tensor([[1.0, 0.0, 0.0]])
    x_world = (R @ x_body.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(x_world, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)


def test_world_vec_to_body_is_transpose_rotation():
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    q = torch.tensor([[c, 0.0, 0.0, s]])
    v_world = torch.tensor([[0.0, 1.0, 0.0]])
    v_body = frames.world_vec_to_body(v_world, q)
    # world +y came from body +x, so body frame sees it as +x
    assert torch.allclose(v_body, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)


def test_skew_matches_cross_product():
    a = torch.tensor([[1.0, 2.0, 3.0]])
    b = torch.tensor([[4.0, 5.0, 6.0]])
    out = (frames.skew(a) @ b.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(out, torch.cross(a, b, dim=-1), atol=1e-6)


def test_constants_are_nwu_sane():
    assert constants.RHO_SEAWATER == 1025.0
    assert constants.GRAVITY == 9.81
    assert (constants.LIN, constants.ANG) == (slice(0, 3), slice(3, 6))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_frames.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lighthill.frames'`.

- [ ] **Step 4: Implement `constants.py`**

```python
"""Physical constants and the canonical 6-vector layout for lighthill."""

RHO_SEAWATER: float = 1025.0  # kg/m^3, seawater
GRAVITY: float = 9.81  # m/s^2

# A wrench/twist is [linear(3), angular(3)] in body frame.
LIN = slice(0, 3)
ANG = slice(3, 6)
```

- [ ] **Step 5: Implement `frames.py`**

```python
"""Rotation/frame utilities (NWU, scalar-first quaternions). Pure torch."""

import torch
from torch import Tensor


def quat_to_rotation_matrix(quat: Tensor) -> Tensor:
    """(w,x,y,z) body->world quaternion ``[...,4]`` -> rotation matrix ``[...,3,3]``."""
    quat = quat / quat.norm(dim=-1, keepdim=True)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return R.reshape(*quat.shape[:-1], 3, 3)


def world_vec_to_body(vec_world: Tensor, quat_wb: Tensor) -> Tensor:
    """Express a world-frame vector in the body frame: ``R^T @ v``."""
    R = quat_to_rotation_matrix(quat_wb)
    return (R.transpose(-1, -2) @ vec_world.unsqueeze(-1)).squeeze(-1)


def skew(v: Tensor) -> Tensor:
    """Skew-symmetric matrix ``[...,3,3]`` such that ``skew(a) @ b == a x b``."""
    zero = torch.zeros_like(v[..., 0])
    row0 = torch.stack([zero, -v[..., 2], v[..., 1]], dim=-1)
    row1 = torch.stack([v[..., 2], zero, -v[..., 0]], dim=-1)
    row2 = torch.stack([-v[..., 1], v[..., 0], zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_frames.py -v`
Expected: PASS (5 passed).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/lighthill/constants.py src/lighthill/frames.py tests/test_frames.py
git commit -m "feat(frames): NWU rotation/skew utilities + physical constants"
```

---

### Task 2: Config schema + YAML loader (`config.py`)

The tool surface: declare per-link hydro parameters, validate at load.

**Files:**
- Create: `src/lighthill/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `class ConfigError(ValueError)`
  - `@dataclass(frozen=True) class AddedMassSpec` — fields: `kind: Literal["matrix","cylinder","sphere","box"]`, `matrix: list[float] | None` (len 6 or 36), `radius: float | None`, `length: float | None`, `axis: Literal["x","y","z"] | None`, `cd: float | None`
  - `@dataclass(frozen=True) class LinkConfig` — `name: str`, `volume: float`, `center_of_buoyancy: tuple[float,float,float]`, `neutrally_buoyant: bool`, `added_mass: AddedMassSpec`, `linear_damping: list[float]` (len 6 or 36), `quadratic_damping: list[float]` (len 6 or 36)
  - `@dataclass(frozen=True) class RobotHydroConfig` — `links: tuple[LinkConfig, ...]`, `density: float = RHO_SEAWATER`
  - `RobotHydroConfig.from_yaml(path: str | Path) -> RobotHydroConfig`
- Consumes: `constants.RHO_SEAWATER`

- [ ] **Step 1: Write the failing test** (`tests/test_config.py`)

```python
import textwrap
import pytest
from lighthill.config import RobotHydroConfig, ConfigError


def _write(tmp_path, body):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(body))
    return p


VALID = """
    density: 1025.0
    links:
      - name: base
        volume: 0.0134
        center_of_buoyancy: [0.0, 0.0, 0.02]
        neutrally_buoyant: false
        added_mass: {kind: matrix, matrix: [6.4, 7.1, 18.0, 0.2, 0.2, 0.2]}
        linear_damping: [4.0, 6.2, 5.2, 0.07, 0.07, 0.07]
        quadratic_damping: [18.0, 21.0, 36.0, 1.5, 1.5, 1.5]
"""


def test_loads_valid_config(tmp_path):
    cfg = RobotHydroConfig.from_yaml(_write(tmp_path, VALID))
    assert cfg.density == 1025.0
    assert len(cfg.links) == 1
    link = cfg.links[0]
    assert link.name == "base"
    assert link.added_mass.kind == "matrix"
    assert len(link.linear_damping) == 6


def test_rejects_wrong_length_damping(tmp_path):
    bad = VALID.replace("linear_damping: [4.0, 6.2, 5.2, 0.07, 0.07, 0.07]",
                        "linear_damping: [4.0, 6.2, 5.2]")
    with pytest.raises(ConfigError, match="linear_damping"):
        RobotHydroConfig.from_yaml(_write(tmp_path, bad))


def test_rejects_asymmetric_full_matrix(tmp_path):
    asym = [0.0] * 36
    asym[1] = 5.0  # M[0,1]=5 but M[1,0]=0 -> asymmetric
    bad = VALID.replace(
        "added_mass: {kind: matrix, matrix: [6.4, 7.1, 18.0, 0.2, 0.2, 0.2]}",
        f"added_mass: {{kind: matrix, matrix: {asym}}}",
    )
    with pytest.raises(ConfigError, match="symmetric"):
        RobotHydroConfig.from_yaml(_write(tmp_path, bad))


def test_rejects_negative_volume(tmp_path):
    bad = VALID.replace("volume: 0.0134", "volume: -0.1")
    with pytest.raises(ConfigError, match="volume"):
        RobotHydroConfig.from_yaml(_write(tmp_path, bad))


def test_cylinder_added_mass_requires_radius_and_length(tmp_path):
    bad = VALID.replace(
        "added_mass: {kind: matrix, matrix: [6.4, 7.1, 18.0, 0.2, 0.2, 0.2]}",
        "added_mass: {kind: cylinder, radius: 0.025}",  # missing length/axis
    )
    with pytest.raises(ConfigError, match="cylinder"):
        RobotHydroConfig.from_yaml(_write(tmp_path, bad))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lighthill.config'`.

- [ ] **Step 3: Implement `config.py`**

```python
"""Per-link hydrodynamics config schema + validated YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from .constants import RHO_SEAWATER


class ConfigError(ValueError):
    """Raised when a hydro config is structurally invalid."""


@dataclass(frozen=True)
class AddedMassSpec:
    kind: Literal["matrix", "cylinder", "sphere", "box"]
    matrix: tuple[float, ...] | None = None
    radius: float | None = None
    length: float | None = None
    axis: Literal["x", "y", "z"] | None = None
    cd: float | None = None


@dataclass(frozen=True)
class LinkConfig:
    name: str
    volume: float
    center_of_buoyancy: tuple[float, float, float]
    neutrally_buoyant: bool
    added_mass: AddedMassSpec
    linear_damping: tuple[float, ...]
    quadratic_damping: tuple[float, ...]


@dataclass(frozen=True)
class RobotHydroConfig:
    links: tuple[LinkConfig, ...]
    density: float = RHO_SEAWATER

    @staticmethod
    def from_yaml(path: str | Path) -> "RobotHydroConfig":
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict) or "links" not in data:
            raise ConfigError("config must be a mapping with a 'links' list")
        density = float(data.get("density", RHO_SEAWATER))
        links = tuple(_parse_link(raw) for raw in data["links"])
        if not links:
            raise ConfigError("config must declare at least one link")
        return RobotHydroConfig(links=links, density=density)


def _parse_link(raw: dict) -> LinkConfig:
    name = str(raw.get("name", "<unnamed>"))
    volume = float(raw.get("volume", 0.0))
    if volume < 0:
        raise ConfigError(f"link '{name}': volume must be >= 0, got {volume}")
    cob = raw.get("center_of_buoyancy", [0.0, 0.0, 0.0])
    if len(cob) != 3:
        raise ConfigError(f"link '{name}': center_of_buoyancy must have 3 elements")
    am = _parse_added_mass(name, raw.get("added_mass", {}))
    lin = _validate_damping(name, "linear_damping", raw.get("linear_damping", [0.0] * 6))
    quad = _validate_damping(name, "quadratic_damping", raw.get("quadratic_damping", [0.0] * 6))
    return LinkConfig(
        name=name,
        volume=volume,
        center_of_buoyancy=tuple(float(c) for c in cob),  # type: ignore[arg-type]
        neutrally_buoyant=bool(raw.get("neutrally_buoyant", False)),
        added_mass=am,
        linear_damping=tuple(float(v) for v in lin),
        quadratic_damping=tuple(float(v) for v in quad),
    )


def _validate_damping(name: str, key: str, vals: list) -> list:
    if len(vals) not in (6, 36):
        raise ConfigError(f"link '{name}': {key} must have 6 or 36 elements, got {len(vals)}")
    return vals


def _parse_added_mass(name: str, raw: dict) -> AddedMassSpec:
    kind = raw.get("kind", "matrix")
    if kind == "matrix":
        m = raw.get("matrix")
        if m is None or len(m) not in (6, 36):
            raise ConfigError(f"link '{name}': matrix added_mass needs 6 or 36 floats")
        if len(m) == 36:
            _require_symmetric(name, m)
        return AddedMassSpec(kind="matrix", matrix=tuple(float(v) for v in m))
    if kind == "cylinder":
        if raw.get("radius") is None or raw.get("length") is None or raw.get("axis") is None:
            raise ConfigError(f"link '{name}': cylinder added_mass needs radius, length, axis")
        return AddedMassSpec(kind="cylinder", radius=float(raw["radius"]),
                             length=float(raw["length"]), axis=raw["axis"])
    if kind == "sphere":
        if raw.get("radius") is None:
            raise ConfigError(f"link '{name}': sphere added_mass needs radius")
        return AddedMassSpec(kind="sphere", radius=float(raw["radius"]))
    if kind == "box":
        if raw.get("radius") is None or raw.get("cd") is None:
            raise ConfigError(f"link '{name}': box added_mass needs radius (half-extent) and cd")
        return AddedMassSpec(kind="box", radius=float(raw["radius"]), cd=float(raw["cd"]))
    raise ConfigError(f"link '{name}': unknown added_mass kind '{kind}'")


def _require_symmetric(name: str, m: list) -> None:
    for i in range(6):
        for j in range(6):
            if abs(m[i * 6 + j] - m[j * 6 + i]) > 1e-9:
                raise ConfigError(f"link '{name}': added-mass matrix must be symmetric")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/lighthill/config.py tests/test_config.py
git commit -m "feat(config): validated per-link hydro config schema + YAML loader"
```

---

### Task 3: Coefficient resolution + shape models (`coefficients.py`)

Turn a `RobotHydroConfig` into stacked, ready-to-use coefficient tensors. This is where the analytic shape formulas live.

**Files:**
- Create: `src/lighthill/coefficients.py`
- Test: `tests/test_coefficients.py`

**Interfaces:**
- Produces:
  - `@dataclass class ResolvedCoefficients` — torch tensors stacked over N links:
    `added_mass: Tensor [N,6,6]`, `linear_damping: Tensor [N,6,6]`, `quadratic_damping: Tensor [N,6,6]`,
    `volume: Tensor [N]`, `center_of_buoyancy: Tensor [N,3]`, `neutrally_buoyant: Tensor [N] bool`,
    `density: float`, `names: tuple[str, ...]`
  - `resolve_coefficients(config: RobotHydroConfig, dtype=torch.float32) -> ResolvedCoefficients`
  - `cylinder_added_mass(radius, length, axis, density) -> Tensor [6,6]`
  - `sphere_added_mass(radius, density) -> Tensor [6,6]`
- Consumes: `config.RobotHydroConfig`, `config.LinkConfig`, `config.AddedMassSpec`

**Shape physics (translate to NWU body axes, axis ∈ {x,y,z}):**
- **cylinder**, slender, transverse added mass per the 2D strip result `m_A = ρ·π·R²·L` on the two axes ⟂ the cylinder axis; ≈0 on the axis; rotational added mass ≈0 (slender-link approximation; document it).
- **sphere**, isotropic translational `m_A = (2/3)·π·ρ·R³` on all three linear axes; rotational ≈0.

- [ ] **Step 1: Write the failing test** (`tests/test_coefficients.py`)

```python
import math
import torch
from lighthill.config import RobotHydroConfig, LinkConfig, AddedMassSpec
from lighthill.coefficients import (
    resolve_coefficients, cylinder_added_mass, sphere_added_mass,
)


def _link(**kw):
    base = dict(
        name="l", volume=0.001, center_of_buoyancy=(0.0, 0.0, 0.0),
        neutrally_buoyant=False,
        added_mass=AddedMassSpec(kind="matrix", matrix=tuple([1.0] * 6)),
        linear_damping=tuple([2.0] * 6), quadratic_damping=tuple([3.0] * 6),
    )
    base.update(kw)
    return LinkConfig(**base)


def test_diagonal_matrix_expands_to_6x6_diag():
    cfg = RobotHydroConfig(links=(_link(),))
    rc = resolve_coefficients(cfg)
    assert rc.added_mass.shape == (1, 6, 6)
    assert torch.allclose(rc.added_mass[0], torch.eye(6))
    assert torch.allclose(rc.linear_damping[0], 2.0 * torch.eye(6))
    assert torch.allclose(rc.quadratic_damping[0], 3.0 * torch.eye(6))


def test_full_36_matrix_round_trips():
    m = [0.0] * 36
    for i in range(6):
        m[i * 6 + i] = float(i + 1)
    m[1], m[6] = 0.5, 0.5  # symmetric off-diagonal
    cfg = RobotHydroConfig(links=(_link(added_mass=AddedMassSpec(kind="matrix", matrix=tuple(m))),))
    rc = resolve_coefficients(cfg)
    assert rc.added_mass[0, 0, 1].item() == 0.5
    assert rc.added_mass[0, 1, 0].item() == 0.5


def test_cylinder_transverse_added_mass_matches_formula():
    R, L, rho = 0.025, 0.15, 1025.0
    expected = rho * math.pi * R * R * L
    M = cylinder_added_mass(R, L, "z", rho)  # axis z -> transverse on x,y
    assert math.isclose(M[0, 0].item(), expected, rel_tol=1e-6)
    assert math.isclose(M[1, 1].item(), expected, rel_tol=1e-6)
    assert M[2, 2].item() == 0.0  # ~no axial added mass


def test_sphere_added_mass_is_isotropic():
    R, rho = 0.1, 1025.0
    expected = (2.0 / 3.0) * math.pi * rho * R**3
    M = sphere_added_mass(R, rho)
    for i in range(3):
        assert math.isclose(M[i, i].item(), expected, rel_tol=1e-6)


def test_stacks_multiple_links():
    cfg = RobotHydroConfig(links=(_link(name="a"), _link(name="b"), _link(name="c")))
    rc = resolve_coefficients(cfg)
    assert rc.added_mass.shape == (3, 6, 6)
    assert rc.names == ("a", "b", "c")
    assert rc.volume.shape == (3,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_coefficients.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lighthill.coefficients'`.

- [ ] **Step 3: Implement `coefficients.py`**

```python
"""Resolve per-link configs into stacked coefficient tensors (+ shape models)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .config import AddedMassSpec, LinkConfig, RobotHydroConfig

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass
class ResolvedCoefficients:
    added_mass: Tensor          # [N,6,6]
    linear_damping: Tensor      # [N,6,6]
    quadratic_damping: Tensor   # [N,6,6]
    volume: Tensor              # [N]
    center_of_buoyancy: Tensor  # [N,3]
    neutrally_buoyant: Tensor   # [N] bool
    density: float
    names: tuple[str, ...]


def _to_6x6(vals: tuple[float, ...], dtype: torch.dtype) -> Tensor:
    t = torch.tensor(vals, dtype=dtype)
    if t.numel() == 6:
        return torch.diag(t)
    return t.reshape(6, 6)


def cylinder_added_mass(radius: float, length: float, axis: str, density: float,
                        dtype: torch.dtype = torch.float32) -> Tensor:
    """Slender-cylinder added mass: transverse = rho*pi*R^2*L, axial ~0, rotational ~0."""
    m_t = density * math.pi * radius * radius * length
    diag = [m_t, m_t, m_t, 0.0, 0.0, 0.0]
    diag[_AXIS_INDEX[axis]] = 0.0  # no added mass along the slender axis
    return torch.diag(torch.tensor(diag, dtype=dtype))


def sphere_added_mass(radius: float, density: float,
                      dtype: torch.dtype = torch.float32) -> Tensor:
    m = (2.0 / 3.0) * math.pi * density * radius**3
    return torch.diag(torch.tensor([m, m, m, 0.0, 0.0, 0.0], dtype=dtype))


def _resolve_added_mass(spec: AddedMassSpec, density: float, dtype: torch.dtype) -> Tensor:
    if spec.kind == "matrix":
        assert spec.matrix is not None
        return _to_6x6(spec.matrix, dtype)
    if spec.kind == "cylinder":
        assert spec.radius and spec.length and spec.axis
        return cylinder_added_mass(spec.radius, spec.length, spec.axis, density, dtype)
    if spec.kind == "sphere":
        assert spec.radius is not None
        return sphere_added_mass(spec.radius, density, dtype)
    # box: isotropic translational added mass ~ that of the bounding sphere; form drag via cd
    assert spec.radius is not None
    return sphere_added_mass(spec.radius, density, dtype)


def resolve_coefficients(config: RobotHydroConfig,
                         dtype: torch.dtype = torch.float32) -> ResolvedCoefficients:
    links: tuple[LinkConfig, ...] = config.links
    added = torch.stack([_resolve_added_mass(l.added_mass, config.density, dtype) for l in links])
    lin = torch.stack([_to_6x6(l.linear_damping, dtype) for l in links])
    quad = torch.stack([_to_6x6(l.quadratic_damping, dtype) for l in links])
    vol = torch.tensor([l.volume for l in links], dtype=dtype)
    cob = torch.tensor([l.center_of_buoyancy for l in links], dtype=dtype)
    neutral = torch.tensor([l.neutrally_buoyant for l in links], dtype=torch.bool)
    return ResolvedCoefficients(
        added_mass=added, linear_damping=lin, quadratic_damping=quad,
        volume=vol, center_of_buoyancy=cob, neutrally_buoyant=neutral,
        density=config.density, names=tuple(l.name for l in links),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_coefficients.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/lighthill/coefficients.py tests/test_coefficients.py
git commit -m "feat(coefficients): resolve configs to tensors + cylinder/sphere shape models"
```

---

### Task 4: Buoyancy/restoring + drag kernels (`forces.py` part 1)

**Files:**
- Create: `src/lighthill/forces.py`
- Test: `tests/test_forces_buoyancy_drag.py`

**Interfaces:**
- Produces (all return body-frame wrench `[...,6]`):
  - `buoyancy_wrench(quat_wb: Tensor, volume: Tensor, cob_body: Tensor, neutrally_buoyant: Tensor, density: float, gravity: float = GRAVITY) -> Tensor`
    — `quat_wb [...,4]`, `volume [...]`, `cob_body [...,3]`, `neutrally_buoyant [...] bool`. Buoyancy force is `+density*gravity*volume` along world +Z, rotated into body frame; moment `= cob_body × F_body`. Neutrally-buoyant links contribute zero.
  - `drag_wrench(v_rel_body: Tensor, linear_damping: Tensor, quadratic_damping: Tensor) -> Tensor`
    — `v_rel_body [...,6]`, damping `[...,6,6]`. Returns `-(D_lin @ v + D_quad @ (|v|⊙v))`.
- Consumes: `frames.world_vec_to_body`, `frames.skew`, `constants.GRAVITY`

- [ ] **Step 1: Write the failing test** (`tests/test_forces_buoyancy_drag.py`)

```python
import torch
from lighthill.forces import buoyancy_wrench, drag_wrench
from lighthill.constants import RHO_SEAWATER, GRAVITY

IDQUAT = torch.tensor([1.0, 0.0, 0.0, 0.0])


def test_buoyancy_upright_pushes_world_up_in_body_frame():
    V = torch.tensor(0.01)
    w = buoyancy_wrench(IDQUAT, V, torch.zeros(3), torch.tensor(False), RHO_SEAWATER)
    expected_fz = RHO_SEAWATER * GRAVITY * 0.01
    assert torch.allclose(w[:3], torch.tensor([0.0, 0.0, expected_fz]), atol=1e-4)
    assert torch.allclose(w[3:], torch.zeros(3), atol=1e-6)  # cob at origin -> no moment


def test_neutrally_buoyant_link_contributes_nothing():
    w = buoyancy_wrench(IDQUAT, torch.tensor(0.01), torch.zeros(3),
                        torch.tensor(True), RHO_SEAWATER)
    assert torch.allclose(w, torch.zeros(6), atol=1e-9)


def test_buoyancy_with_cob_offset_makes_restoring_moment():
    # cob 0.02 m above origin (+z); upright -> force +z through a point on +z axis -> zero moment
    V = torch.tensor(0.01)
    cob = torch.tensor([0.0, 0.0, 0.02])
    w_up = buoyancy_wrench(IDQUAT, V, cob, torch.tensor(False), RHO_SEAWATER)
    assert torch.allclose(w_up[3:], torch.zeros(3), atol=1e-6)
    # roll 90 deg about x: world-up now along body -y; force x cob -> nonzero moment about... check magnitude
    import math
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    q_roll = torch.tensor([c, s, 0.0, 0.0])  # 90 deg about x
    w_tilt = buoyancy_wrench(q_roll, V, cob, torch.tensor(False), RHO_SEAWATER)
    assert w_tilt[3:].norm() > 1e-3  # a restoring moment appears when tilted


def test_drag_zero_velocity_is_zero():
    D = torch.eye(6)
    w = drag_wrench(torch.zeros(6), D, D)
    assert torch.allclose(w, torch.zeros(6), atol=1e-9)


def test_drag_opposes_motion_and_has_quadratic_term():
    v = torch.zeros(6)
    v[0] = 2.0  # surge
    D_lin = torch.eye(6)
    D_quad = torch.eye(6)
    w = drag_wrench(v, D_lin, D_quad)
    # -(1*2 + 1*(|2|*2)) = -(2 + 4) = -6 on surge axis
    assert torch.isclose(w[0], torch.tensor(-6.0), atol=1e-5)
    assert torch.allclose(w[1:], torch.zeros(5), atol=1e-6)


def test_batched_shapes_broadcast():
    q = IDQUAT.expand(4, 4)
    V = torch.full((4,), 0.01)
    cob = torch.zeros(4, 3)
    neutral = torch.zeros(4, dtype=torch.bool)
    w = buoyancy_wrench(q, V, cob, neutral, RHO_SEAWATER)
    assert w.shape == (4, 6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_forces_buoyancy_drag.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lighthill.forces'`.

- [ ] **Step 3: Implement `forces.py` (buoyancy + drag)**

```python
"""Per-link Fossen force kernels. Pure torch; body-frame wrench [...,6] = [F(3), M(3)]."""

from __future__ import annotations

import torch
from torch import Tensor

from .constants import GRAVITY
from .frames import skew, world_vec_to_body


def buoyancy_wrench(quat_wb: Tensor, volume: Tensor, cob_body: Tensor,
                    neutrally_buoyant: Tensor, density: float,
                    gravity: float = GRAVITY) -> Tensor:
    """Buoyancy at the center of buoyancy, expressed as a body-frame wrench."""
    mag = density * gravity * volume  # [...]
    f_world = torch.zeros(*volume.shape, 3, dtype=volume.dtype, device=volume.device)
    f_world[..., 2] = mag  # +Z world (NWU up)
    f_body = world_vec_to_body(f_world, quat_wb)  # [...,3]
    f_body = torch.where(neutrally_buoyant.unsqueeze(-1), torch.zeros_like(f_body), f_body)
    moment = torch.cross(cob_body, f_body, dim=-1)  # r x F
    return torch.cat([f_body, moment], dim=-1)


def drag_wrench(v_rel_body: Tensor, linear_damping: Tensor,
                quadratic_damping: Tensor) -> Tensor:
    """-(D_lin @ v + D_quad @ (|v| * v)), body frame."""
    v = v_rel_body.unsqueeze(-1)  # [...,6,1]
    quad_term = (v_rel_body.abs() * v_rel_body).unsqueeze(-1)
    drag = linear_damping @ v + quadratic_damping @ quad_term
    return -drag.squeeze(-1)
```

Note: `skew` is imported for Task 5; keep the import (it is used there). If lint flags it as unused before Task 5 lands, add `# noqa: F401` and remove the noqa in Task 5.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_forces_buoyancy_drag.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/lighthill/forces.py tests/test_forces_buoyancy_drag.py
git commit -m "feat(forces): buoyancy/restoring + linear+quadratic drag kernels"
```

---

### Task 5: Added-mass Coriolis + off-diagonal residual (`forces.py` part 2)

**Files:**
- Modify: `src/lighthill/forces.py` (add two functions; drop the `skew` noqa if added)
- Test: `tests/test_forces_addedmass.py`

**Interfaces:**
- Produces (body-frame wrench `[...,6]`):
  - `added_mass_coriolis(added_mass: Tensor, v_rel_body: Tensor) -> Tensor` — `-C_A(ν) @ ν` where `C_A` is built from the 6×6 `added_mass` per Fossen's skew construction.
  - `added_mass_residual(added_mass_offdiag: Tensor, accel_body: Tensor) -> Tensor` — `-(M_A_offdiag @ a)`; consumed by `apply.py` in Plan B (diagonal added mass goes to inertia augmentation there; here it is a pure, tested function).
- Consumes: `frames.skew`

**Fossen Coriolis construction** (`M_A` blocks `A11,A12,A21,A22` are 3×3; `ν=[ν1,ν2]`):
```
C_A(ν) = [[ 0,            -skew(A11 ν1 + A12 ν2) ],
          [ -skew(A11 ν1 + A12 ν2), -skew(A21 ν1 + A22 ν2) ]]
```

- [ ] **Step 1: Write the failing test** (`tests/test_forces_addedmass.py`)

```python
import torch
from lighthill.forces import added_mass_coriolis, added_mass_residual


def test_coriolis_zero_velocity_is_zero():
    M = torch.diag(torch.tensor([10.0, 20.0, 30.0, 1.0, 2.0, 3.0]))
    w = added_mass_coriolis(M, torch.zeros(6))
    assert torch.allclose(w, torch.zeros(6), atol=1e-9)


def test_coriolis_pure_translation_gives_moment_only():
    # Diagonal added mass, surge+sway velocity -> coupling produces a yaw moment, no net force.
    M = torch.diag(torch.tensor([10.0, 20.0, 30.0, 0.0, 0.0, 0.0]))
    v = torch.zeros(6)
    v[0], v[1] = 1.0, 1.0  # u, v
    w = added_mass_coriolis(M, v)
    assert torch.allclose(w[:3], torch.zeros(3), atol=1e-6)  # no net force
    assert w[3:].abs().sum() > 1e-6                          # some moment


def test_coriolis_is_power_neutral():
    # Coriolis/centripetal forces do no work: w · v == 0 for any state.
    torch.manual_seed(0)
    M = torch.rand(6, 6)
    M = M + M.T  # symmetric
    v = torch.randn(6)
    w = added_mass_coriolis(M, v)
    assert torch.isclose(torch.dot(w, v), torch.tensor(0.0), atol=1e-4)


def test_residual_is_minus_matrix_times_accel():
    M_off = torch.zeros(6, 6)
    M_off[0, 4] = 2.0
    M_off[4, 0] = 2.0
    a = torch.zeros(6)
    a[4] = 3.0
    w = added_mass_residual(M_off, a)
    assert torch.isclose(w[0], torch.tensor(-6.0), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_forces_addedmass.py -v`
Expected: FAIL — `ImportError: cannot import name 'added_mass_coriolis'`.

- [ ] **Step 3: Implement (append to `forces.py`)**

```python
def added_mass_coriolis(added_mass: Tensor, v_rel_body: Tensor) -> Tensor:
    """-C_A(nu) @ nu, with C_A from Fossen's skew construction. Body frame."""
    a11 = added_mass[..., 0:3, 0:3]
    a12 = added_mass[..., 0:3, 3:6]
    a21 = added_mass[..., 3:6, 0:3]
    a22 = added_mass[..., 3:6, 3:6]
    nu1 = v_rel_body[..., 0:3]
    nu2 = v_rel_body[..., 3:6]
    top = (a11 @ nu1.unsqueeze(-1) + a12 @ nu2.unsqueeze(-1)).squeeze(-1)
    bot = (a21 @ nu1.unsqueeze(-1) + a22 @ nu2.unsqueeze(-1)).squeeze(-1)
    s_top = skew(top)
    s_bot = skew(bot)
    zero = torch.zeros_like(s_top)
    upper = torch.cat([zero, -s_top], dim=-1)
    lower = torch.cat([-s_top, -s_bot], dim=-1)
    c_a = torch.cat([upper, lower], dim=-2)  # [...,6,6]
    return -(c_a @ v_rel_body.unsqueeze(-1)).squeeze(-1)


def added_mass_residual(added_mass_offdiag: Tensor, accel_body: Tensor) -> Tensor:
    """-(M_A_offdiag @ a). Off-diagonal added-mass reaction (Plan B feeds filtered accel)."""
    return -(added_mass_offdiag @ accel_body.unsqueeze(-1)).squeeze(-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_forces_addedmass.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/lighthill/forces.py tests/test_forces_addedmass.py
git commit -m "feat(forces): added-mass Coriolis + off-diagonal residual kernels"
```

---

### Task 6: Current field + relative velocity (`current.py`)

**Files:**
- Create: `src/lighthill/current.py`
- Test: `tests/test_current.py`

**Interfaces:**
- Produces:
  - `@dataclass class CurrentField` — `max_speed: float = 0.5`, `noise_std: float = 0.0`
    - `sample(num_envs: int, generator: torch.Generator | None = None) -> Tensor [num_envs,3]` — world-frame current, magnitude uniform in `[0, max_speed]`, uniformly random direction (domain randomization at reset)
    - `perturb(current_world: Tensor, generator=None) -> Tensor` — add per-step Gaussian noise `noise_std` (no-op if 0)
  - `relative_velocity(v_body: Tensor, quat_wb: Tensor, current_world: Tensor) -> Tensor`
    — `v_body [...,6]`, returns body twist with linear part `v_lin − Rᵀ·v_current`; angular part unchanged. Current affects drag/Coriolis only (it enters via `v_rel`).
- Consumes: `frames.world_vec_to_body`, `constants.LIN/ANG`

- [ ] **Step 1: Write the failing test** (`tests/test_current.py`)

```python
import torch
from lighthill.current import CurrentField, relative_velocity

IDQUAT = torch.tensor([1.0, 0.0, 0.0, 0.0])


def test_zero_current_leaves_velocity_unchanged():
    v = torch.tensor([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    out = relative_velocity(v, IDQUAT, torch.zeros(3))
    assert torch.allclose(out, v, atol=1e-6)


def test_current_subtracts_from_linear_only():
    v = torch.tensor([1.0, 0.0, 0.0, 0.5, 0.0, 0.0])
    cur = torch.tensor([0.4, 0.0, 0.0])  # 0.4 m/s along world +x; upright body
    out = relative_velocity(v, IDQUAT, cur)
    assert torch.isclose(out[0], torch.tensor(0.6), atol=1e-6)  # 1.0 - 0.4
    assert torch.allclose(out[3:], v[3:], atol=1e-6)            # angular untouched


def test_sample_magnitude_within_bounds():
    g = torch.Generator().manual_seed(0)
    field = CurrentField(max_speed=0.5)
    cur = field.sample(1000, generator=g)
    speeds = cur.norm(dim=-1)
    assert cur.shape == (1000, 3)
    assert speeds.max().item() <= 0.5 + 1e-6
    assert speeds.min().item() >= 0.0


def test_perturb_is_noop_when_std_zero():
    field = CurrentField(max_speed=0.5, noise_std=0.0)
    cur = torch.ones(4, 3)
    assert torch.allclose(field.perturb(cur), cur)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_current.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lighthill.current'`.

- [ ] **Step 3: Implement `current.py`**

```python
"""Ocean-current model: uniform global flow per env + relative-velocity computation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .constants import ANG, LIN
from .frames import world_vec_to_body


@dataclass
class CurrentField:
    max_speed: float = 0.5
    noise_std: float = 0.0

    def sample(self, num_envs: int, generator: torch.Generator | None = None) -> Tensor:
        speed = torch.rand(num_envs, 1, generator=generator) * self.max_speed
        direction = torch.randn(num_envs, 3, generator=generator)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return direction * speed

    def perturb(self, current_world: Tensor, generator: torch.Generator | None = None) -> Tensor:
        if self.noise_std == 0.0:
            return current_world
        noise = torch.randn(current_world.shape, generator=generator) * self.noise_std
        return current_world + noise


def relative_velocity(v_body: Tensor, quat_wb: Tensor, current_world: Tensor) -> Tensor:
    """Body twist relative to the flow. Current enters the linear part only."""
    cur_body = world_vec_to_body(current_world, quat_wb)
    out = v_body.clone()
    out[..., LIN] = v_body[..., LIN] - cur_body
    out[..., ANG] = v_body[..., ANG]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_current.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/lighthill/current.py tests/test_current.py
git commit -m "feat(current): uniform current field + relative-velocity kernel"
```

---

### Task 7: Reference integrator + validation scenarios (`validation/reference.py`)

A minimal single-rigid-body Fossen integrator (CPU torch) that assembles the kernels into full dynamics, plus the closed-form scenario drivers. This is what proves the physics before Isaac exists.

**Files:**
- Create: `src/lighthill/validation/__init__.py`
- Create: `src/lighthill/validation/reference.py`
- Test: `tests/test_validation.py`

**Interfaces:**
- Produces:
  - `@dataclass class Body` — `mass: float`, `inertia: tuple[float,float,float]` (Ixx,Iyy,Izz about body origin), `volume: float`, `cob: tuple[float,float,float]`, `added_mass: Tensor [6,6]`, `linear_damping: Tensor [6,6]`, `quadratic_damping: Tensor [6,6]`, `density: float`
  - `simulate(body: Body, *, steps: int, dt: float, external_force_body: Tensor | None = None, quat0=None, omega0=None, vel0=None, gravity=GRAVITY) -> dict[str, Tensor]` — returns trajectory dict with keys `pos [steps,3]`, `quat [steps,4]`, `twist [steps,6]`. Semi-implicit Euler. Mass matrix `M = M_RB + M_A`; accelerations from buoyancy (kernel) + gravity (−Z world) + drag (kernel) + added-mass Coriolis (kernel) + applied external body force.
  - `terminal_velocity_quadratic(force: float, d_quad: float) -> float` — closed form `sqrt(force/d_quad)`
- Consumes: all of `forces`, `frames`, `constants`

- [ ] **Step 1: Write the failing test** (`tests/test_validation.py`)

```python
import math
import torch
from lighthill.validation.reference import Body, simulate, terminal_velocity_quadratic
from lighthill.constants import RHO_SEAWATER, GRAVITY


def _neutral_body(**kw):
    # Neutrally buoyant: weight == buoyancy, so vertical is balanced unless forced.
    mass = RHO_SEAWATER * 0.01  # volume 0.01 m^3 -> neutral
    base = dict(
        mass=mass, inertia=(0.1, 0.1, 0.1), volume=0.01, cob=(0.0, 0.0, 0.0),
        added_mass=torch.diag(torch.tensor([5.0, 5.0, 5.0, 0.1, 0.1, 0.1])),
        linear_damping=torch.zeros(6, 6),
        quadratic_damping=torch.diag(torch.tensor([40.0, 40.0, 40.0, 1.0, 1.0, 1.0])),
        density=RHO_SEAWATER,
    )
    base.update(kw)
    return Body(**base)


def test_drag_terminal_velocity_matches_closed_form():
    # Constant surge thrust against pure quadratic drag -> known terminal speed.
    force = 50.0
    d_quad = 40.0
    body = _neutral_body()
    f_ext = torch.zeros(6)
    f_ext[0] = force
    traj = simulate(body, steps=4000, dt=0.005, external_force_body=f_ext)
    u_final = traj["twist"][-1, 0].item()
    expected = terminal_velocity_quadratic(force, d_quad)
    assert math.isclose(u_final, expected, rel_tol=0.02)


def test_restoring_returns_tilted_body_toward_upright():
    # cob above CoM -> stable; release from a roll, it should settle near upright.
    body = _neutral_body(cob=(0.0, 0.0, 0.03))
    q0 = torch.tensor([math.cos(0.15), math.sin(0.15), 0.0, 0.0])  # ~17 deg roll
    traj = simulate(body, steps=6000, dt=0.005, quat0=q0)
    # final roll angle (about x) should be much smaller than initial
    qf = traj["quat"][-1]
    roll_final = 2 * math.atan2(qf[1].item(), qf[0].item())
    assert abs(roll_final) < 0.05  # settled toward upright (started at ~0.30 rad)


def test_neutral_body_at_rest_stays_at_rest():
    body = _neutral_body()
    traj = simulate(body, steps=500, dt=0.01)
    assert traj["twist"][-1].abs().max().item() < 1e-3


def test_terminal_velocity_closed_form():
    assert math.isclose(terminal_velocity_quadratic(40.0, 10.0), 2.0, rel_tol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_validation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lighthill.validation'`.

- [ ] **Step 3: Implement `validation/__init__.py`**

```python
"""Numerical validation harness for the lighthill force kernels."""
```

- [ ] **Step 4: Implement `validation/reference.py`**

```python
"""Single-rigid-body Fossen integrator that assembles the kernels into full dynamics.

This is the on-CPU reference used to validate the force law before the Isaac Lab
glue exists. It is intentionally minimal: one body, semi-implicit Euler, body-frame
twist. It is NOT the production sim (that is Isaac Lab, Plan B)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..constants import GRAVITY
from ..forces import added_mass_coriolis, buoyancy_wrench, drag_wrench
from ..frames import quat_to_rotation_matrix


@dataclass
class Body:
    mass: float
    inertia: tuple[float, float, float]
    volume: float
    cob: tuple[float, float, float]
    added_mass: Tensor
    linear_damping: Tensor
    quadratic_damping: Tensor
    density: float


def terminal_velocity_quadratic(force: float, d_quad: float) -> float:
    return math.sqrt(force / d_quad)


def _rigid_body_mass_matrix(body: Body) -> Tensor:
    m = body.mass
    ix, iy, iz = body.inertia
    return torch.diag(torch.tensor([m, m, m, ix, iy, iz], dtype=torch.float32))


def _quat_mul(q: Tensor, r: Tensor) -> Tensor:
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return torch.tensor([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _integrate_quat(quat: Tensor, omega_body: Tensor, dt: float) -> Tensor:
    omega_quat = torch.cat([torch.zeros(1), omega_body])
    dq = 0.5 * _quat_mul(quat, omega_quat)
    q = quat + dq * dt
    return q / q.norm()


def simulate(body: Body, *, steps: int, dt: float,
             external_force_body: Tensor | None = None,
             quat0: Tensor | None = None, omega0: Tensor | None = None,
             vel0: Tensor | None = None, gravity: float = GRAVITY) -> dict[str, Tensor]:
    mass_matrix = _rigid_body_mass_matrix(body) + body.added_mass
    minv = torch.linalg.inv(mass_matrix)
    cob = torch.tensor(body.cob, dtype=torch.float32)
    vol = torch.tensor(body.volume, dtype=torch.float32)
    not_neutral = torch.tensor(False)
    f_ext = external_force_body if external_force_body is not None else torch.zeros(6)

    quat = quat0.clone() if quat0 is not None else torch.tensor([1.0, 0.0, 0.0, 0.0])
    twist = torch.zeros(6)
    if vel0 is not None:
        twist[0:3] = vel0
    if omega0 is not None:
        twist[3:6] = omega0
    pos = torch.zeros(3)

    pos_hist, quat_hist, twist_hist = [], [], []
    for _ in range(steps):
        buoy = buoyancy_wrench(quat, vol, cob, not_neutral, body.density, gravity)
        # gravity (weight) acts at CoM (body origin), world -Z, no moment
        R = quat_to_rotation_matrix(quat)
        weight_world = torch.tensor([0.0, 0.0, -body.mass * gravity])
        weight_body = (R.transpose(-1, -2) @ weight_world.unsqueeze(-1)).squeeze(-1)
        grav = torch.cat([weight_body, torch.zeros(3)])
        drag = drag_wrench(twist, body.linear_damping, body.quadratic_damping)
        cor = added_mass_coriolis(body.added_mass, twist)
        total = buoy + grav + drag + cor + f_ext
        accel = (minv @ total.unsqueeze(-1)).squeeze(-1)
        twist = twist + accel * dt  # semi-implicit
        # advance pose: linear in world frame, angular via quaternion kinematics
        vel_world = (R @ twist[0:3].unsqueeze(-1)).squeeze(-1)
        pos = pos + vel_world * dt
        quat = _integrate_quat(quat, twist[3:6], dt)
        pos_hist.append(pos.clone())
        quat_hist.append(quat.clone())
        twist_hist.append(twist.clone())

    return {
        "pos": torch.stack(pos_hist),
        "quat": torch.stack(quat_hist),
        "twist": torch.stack(twist_hist),
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_validation.py -v`
Expected: PASS (4 passed). If `test_drag_terminal_velocity` is marginally outside tol, increase `steps` to 6000 — it must converge, not be re-toleranced loose.

- [ ] **Step 6: Commit**

```bash
git add src/lighthill/validation/__init__.py src/lighthill/validation/reference.py tests/test_validation.py
git commit -m "feat(validation): single-body Fossen integrator + drag/restoring scenarios"
```

---

### Task 8: Example configs as shipped assets (`configs/*.yaml`)

Ship the two example configs the spec names, and prove they load and resolve.

**Files:**
- Create: `src/lighthill/configs/bluerov2_auv.yaml`
- Create: `src/lighthill/configs/bluerov2_alpha_uvms.yaml`
- Modify: `pyproject.toml` (ensure non-`.py` files in `src/lighthill/configs/` are packaged — `uv_build` includes package data by default; add an explicit note/comment)
- Test: `tests/test_example_configs.py`

**Interfaces:**
- Consumes: `config.RobotHydroConfig.from_yaml`, `coefficients.resolve_coefficients`
- Produces: a helper `lighthill.example_config_path(name: str) -> Path` (added to `__init__.py` in Task 9; for this task reference the files by `importlib.resources`)

Coefficients: AUV uses BlueROV2 diagonal values (from MarineGym `BlueROV.yaml`, MIT — record provenance in a YAML comment). UVMS adds 5 cylinder-modeled Reach Alpha links.

- [ ] **Step 1: Write the failing test** (`tests/test_example_configs.py`)

```python
from importlib.resources import files
import torch
from lighthill.config import RobotHydroConfig
from lighthill.coefficients import resolve_coefficients


def _path(name):
    return files("lighthill.configs").joinpath(name)


def test_auv_config_loads_and_resolves_to_one_link():
    cfg = RobotHydroConfig.from_yaml(_path("bluerov2_auv.yaml"))
    rc = resolve_coefficients(cfg)
    assert len(cfg.links) == 1
    assert rc.added_mass.shape == (1, 6, 6)
    assert (rc.volume > 0).all()


def test_uvms_config_has_vehicle_plus_arm_links():
    cfg = RobotHydroConfig.from_yaml(_path("bluerov2_alpha_uvms.yaml"))
    rc = resolve_coefficients(cfg)
    assert len(cfg.links) >= 6  # vehicle + 5 arm links
    assert rc.added_mass.shape[0] == len(cfg.links)
    # arm links are cylinder-modeled -> finite transverse added mass
    assert torch.isfinite(rc.added_mass).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_example_configs.py -v`
Expected: FAIL — `FileNotFoundError`/`ModuleNotFoundError` for `lighthill.configs`.

- [ ] **Step 3: Create `src/lighthill/configs/bluerov2_auv.yaml`**

```yaml
# BlueROV2 (AUV-only) hydro config.
# Diagonal coefficients adapted from MarineGym's BlueROV.yaml (MIT, chu2025marinegym).
# Added mass / damping are vehicle-frame diagonals in NWU [x,y,z,roll,pitch,yaw].
density: 1025.0
links:
  - name: base_link
    volume: 0.0134            # ~ neutral against ~13.5 kg
    center_of_buoyancy: [0.0, 0.0, 0.01]
    neutrally_buoyant: false
    added_mass: {kind: matrix, matrix: [6.36, 7.12, 18.68, 0.189, 0.135, 0.222]}
    linear_damping: [13.7, 0.0, 33.0, 0.0, 0.8, 0.0]
    quadratic_damping: [141.0, 217.0, 190.0, 1.19, 0.47, 1.5]
```

- [ ] **Step 4: Create `src/lighthill/configs/bluerov2_alpha_uvms.yaml`**

```yaml
# BlueROV2 + Reach Alpha 5-DOF arm (UVMS) hydro config.
# Vehicle: explicit BlueROV2 diagonal (as in bluerov2_auv.yaml).
# Arm links: cylinder shape models (Reach Alpha slender links), axis=z, neutrally
# buoyant approximation (calibrate volumes during Plan B pilot — see design spec risks).
density: 1025.0
links:
  - name: base_link
    volume: 0.0134
    center_of_buoyancy: [0.0, 0.0, 0.01]
    neutrally_buoyant: false
    added_mass: {kind: matrix, matrix: [6.36, 7.12, 18.68, 0.189, 0.135, 0.222]}
    linear_damping: [13.7, 0.0, 33.0, 0.0, 0.8, 0.0]
    quadratic_damping: [141.0, 217.0, 190.0, 1.19, 0.47, 1.5]
  - name: alpha_link_1
    volume: 0.00012
    center_of_buoyancy: [0.0, 0.0, 0.0]
    neutrally_buoyant: true
    added_mass: {kind: cylinder, radius: 0.020, length: 0.10, axis: z}
    linear_damping: [3.0, 3.0, 1.0, 0.01, 0.01, 0.01]
    quadratic_damping: [12.0, 12.0, 4.0, 0.05, 0.05, 0.05]
  - name: alpha_link_2
    volume: 0.00010
    center_of_buoyancy: [0.0, 0.0, 0.0]
    neutrally_buoyant: true
    added_mass: {kind: cylinder, radius: 0.018, length: 0.12, axis: z}
    linear_damping: [3.0, 3.0, 1.0, 0.01, 0.01, 0.01]
    quadratic_damping: [12.0, 12.0, 4.0, 0.05, 0.05, 0.05]
  - name: alpha_link_3
    volume: 0.00008
    center_of_buoyancy: [0.0, 0.0, 0.0]
    neutrally_buoyant: true
    added_mass: {kind: cylinder, radius: 0.016, length: 0.12, axis: z}
    linear_damping: [2.5, 2.5, 0.8, 0.01, 0.01, 0.01]
    quadratic_damping: [10.0, 10.0, 3.0, 0.04, 0.04, 0.04]
  - name: alpha_link_4
    volume: 0.00006
    center_of_buoyancy: [0.0, 0.0, 0.0]
    neutrally_buoyant: true
    added_mass: {kind: cylinder, radius: 0.014, length: 0.10, axis: z}
    linear_damping: [2.0, 2.0, 0.6, 0.01, 0.01, 0.01]
    quadratic_damping: [8.0, 8.0, 2.5, 0.03, 0.03, 0.03]
  - name: alpha_link_5
    volume: 0.00004
    center_of_buoyancy: [0.0, 0.0, 0.0]
    neutrally_buoyant: true
    added_mass: {kind: cylinder, radius: 0.012, length: 0.08, axis: z}
    linear_damping: [1.5, 1.5, 0.5, 0.01, 0.01, 0.01]
    quadratic_damping: [6.0, 6.0, 2.0, 0.02, 0.02, 0.02]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_example_configs.py -v`
Expected: PASS (2 passed). If the configs dir is not importable, ensure `src/lighthill/configs/` has the YAMLs and that `uv_build` packages package data (it does by default for files under the package dir).

- [ ] **Step 6: Commit**

```bash
git add src/lighthill/configs/bluerov2_auv.yaml src/lighthill/configs/bluerov2_alpha_uvms.yaml tests/test_example_configs.py pyproject.toml
git commit -m "feat(configs): ship BlueROV2 AUV + BlueROV2+Alpha UVMS example configs"
```

---

### Task 9: Public API + full-gate green (`__init__.py`)

Expose a clean import surface and make the whole gate pass.

**Files:**
- Modify: `src/lighthill/__init__.py` (add exports + `example_config_path`)
- Modify: `tests/test_smoke.py` (replace the scaffold smoke test with a real public-API smoke test)
- Test: `tests/test_smoke.py`

**Interfaces:**
- Produces:
  - re-exports: `RobotHydroConfig`, `LinkConfig`, `AddedMassSpec`, `ConfigError`, `resolve_coefficients`, `ResolvedCoefficients`, `buoyancy_wrench`, `drag_wrench`, `added_mass_coriolis`, `added_mass_residual`, `CurrentField`, `relative_velocity`
  - `example_config_path(name: str) -> Path`

- [ ] **Step 1: Write the failing test** (replace `tests/test_smoke.py`)

```python
import torch
import lighthill


def test_public_api_is_importable():
    for sym in [
        "RobotHydroConfig", "resolve_coefficients", "buoyancy_wrench",
        "drag_wrench", "added_mass_coriolis", "CurrentField", "relative_velocity",
    ]:
        assert hasattr(lighthill, sym), sym


def test_end_to_end_auv_wrench_from_shipped_config():
    cfg = lighthill.RobotHydroConfig.from_yaml(lighthill.example_config_path("bluerov2_auv.yaml"))
    rc = lighthill.resolve_coefficients(cfg)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    w = lighthill.buoyancy_wrench(quat, rc.volume, rc.center_of_buoyancy,
                                  rc.neutrally_buoyant, rc.density)
    assert w.shape == (1, 6)
    assert w[0, 2] > 0  # buoyancy points world-up
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: FAIL — `AttributeError: module 'lighthill' has no attribute 'RobotHydroConfig'`.

- [ ] **Step 3: Implement exports in `__init__.py`** (append below the existing docstring + `__version__`)

```python
from pathlib import Path
from importlib.resources import files

from .config import AddedMassSpec, ConfigError, LinkConfig, RobotHydroConfig
from .coefficients import ResolvedCoefficients, resolve_coefficients
from .forces import (
    added_mass_coriolis,
    added_mass_residual,
    buoyancy_wrench,
    drag_wrench,
)
from .current import CurrentField, relative_velocity


def example_config_path(name: str) -> Path:
    """Absolute path to a shipped example config under lighthill/configs/."""
    return Path(str(files("lighthill.configs").joinpath(name)))


__all__ = [
    "__version__",
    "RobotHydroConfig", "LinkConfig", "AddedMassSpec", "ConfigError",
    "resolve_coefficients", "ResolvedCoefficients",
    "buoyancy_wrench", "drag_wrench", "added_mass_coriolis", "added_mass_residual",
    "CurrentField", "relative_velocity", "example_config_path",
]
```

- [ ] **Step 4: Run the focused test, then the full gate**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: PASS (2 passed).

Run: `just check`
Expected: ruff clean, mypy clean, all tests pass, branch coverage ≥78%. Fix any lint/type findings (e.g. add return types, `from __future__ import annotations` where needed) until green.

- [ ] **Step 5: Commit**

```bash
git add src/lighthill/__init__.py tests/test_smoke.py
git commit -m "feat(api): export public surface + example_config_path; full gate green"
```

---

## Self-Review

**Spec coverage** (against `docs/design/2026-06-28-hydrodynamics-design.md`):

| Spec element | Task |
|---|---|
| `config` module — per-link schema, validated at load | Task 2 |
| `coefficients` — explicit 6×6 or shape-based | Task 3 |
| Shape models cylinder/sphere/box | Task 3 |
| `forces` — buoyancy/restoring | Task 4 |
| `forces` — drag (linear+quadratic) | Task 4 |
| `forces` — added-mass Coriolis | Task 5 |
| `forces` — off-diagonal residual | Task 5 |
| `current` — flow field + relative velocity | Task 6 |
| NWU frame handling + explicit sign asserts | Tasks 1, 4 (tests) |
| Validation tests 1–3 (free-decay, drag terminal, restoring) | Task 7 |
| Example configs: AUV BlueROV2 + UVMS BlueROV2+Alpha | Task 8 |
| Borrow provenance (MarineGym coefficients) | Task 8 (YAML comments) |
| GPU-vectorized/batched, no CPU round-trips | Tasks 1,4,5,6 (batched torch, no `.numpy()`) |

**Deferred to Plan B (correctly out of scope here):** `apply` (Isaac glue), `set_body_inertias` inertia augmentation, in-sim validation test 4 (arm-swing reaction gate), per-link wrench application over an `Articulation`. The diagonal-vs-off-diagonal added-mass split is *implemented* here (`added_mass_residual` is a tested pure function) but its *use* (diagonal → inertia augmentation) is Plan B.

**Placeholder scan:** none — every code step has complete code; every test has real assertions.

**Type consistency:** `RobotHydroConfig`/`LinkConfig`/`AddedMassSpec` names match across Tasks 2/3/8/9. `ResolvedCoefficients` field names (`added_mass`, `linear_damping`, `quadratic_damping`, `volume`, `center_of_buoyancy`, `neutrally_buoyant`, `density`, `names`) are consumed identically in Tasks 4/7/8/9. Wrench `[...,6]` and twist `[...,6]` ordering is fixed in Global Constraints and used consistently. `buoyancy_wrench` signature is identical in Tasks 4 and 9.

**Known calibration caveats (not plan defects):** the BlueROV2 coefficients and Reach Alpha cylinder volumes are literature/shape-derived starting values; the spec explicitly defers experimental identification and per-arm-link buoyancy calibration to Plan B's pilot. The validation scenarios assert *physical behavior* (terminal velocity, settling), not specific coefficient values, so they remain valid regardless of calibration.
