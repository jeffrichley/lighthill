"""Per-link hydrodynamics config schema + validated YAML loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from .constants import RHO_SEAWATER


class ConfigError(ValueError):
    """Raised when a hydro config is structurally invalid."""


@dataclass(frozen=True)
class AddedMassSpec:
    """How a link's added mass is specified, before it is resolved to a matrix.

    ``kind`` selects one of four authoring shortcuts and dictates which optional
    fields are required: ``matrix`` needs an explicit 6- or 36-element ``matrix``;
    ``cylinder`` needs ``radius``/``length``/``axis`` (slender-body form);
    ``sphere`` needs ``radius``; ``box`` needs ``radius`` (half-extent) plus a
    form-drag coefficient ``cd``. Resolution to a 6x6 tensor happens later in
    :func:`~lighthill.coefficients.resolve_coefficients`.
    """

    kind: Literal["matrix", "cylinder", "sphere", "box"]
    matrix: tuple[float, ...] | None = None
    radius: float | None = None
    length: float | None = None
    axis: Literal["x", "y", "z"] | None = None
    cd: float | None = None


@dataclass(frozen=True)
class LiftSpec:
    """Ellipsoid Kutta + Magnus lift geometry for a link. Semi-axes are the equivalent
    ellipsoid half-lengths (all > 0); coefficients follow MuJoCo's fluidcoef defaults."""

    semi_axes: tuple[float, float, float]
    c_kutta: float = 1.0
    c_magnus: float = 1.0


@dataclass(frozen=True)
class LinkConfig:
    """Immutable hydrodynamic description of one rigid link of the robot.

    Carries the geometry and coefficients the force law needs for a single
    body: displaced ``volume`` and ``center_of_buoyancy`` (buoyancy), the
    ``added_mass`` spec (inertial reaction of the entrained fluid), and the
    ``linear_damping`` / ``quadratic_damping`` vectors (6 diagonal or 36 full
    entries) that drive the drag wrench.
    """

    name: str
    volume: float
    center_of_buoyancy: tuple[float, float, float]
    added_mass: AddedMassSpec
    linear_damping: tuple[float, ...]
    quadratic_damping: tuple[float, ...]
    lift: LiftSpec | None = None


@dataclass(frozen=True)
class RobotHydroConfig:
    """Whole-robot hydro config: the ordered links plus the ambient fluid density.

    ``density`` defaults to seawater (:data:`RHO_SEAWATER`) and applies to every
    link's buoyancy and geometry-derived added mass; ``links`` is the ordered
    tuple whose index becomes the per-link batch dimension downstream.
    """

    links: tuple[LinkConfig, ...]
    density: float = RHO_SEAWATER

    @staticmethod
    def from_yaml(path: str | Path) -> RobotHydroConfig:
        """Load and validate a robot hydro config from a YAML file.

        Expects a mapping with a non-empty ``links`` list and an optional
        top-level ``density`` (falls back to seawater). Raises
        :class:`ConfigError` on any structural problem — not a mapping, missing
        or empty ``links``, or a malformed link/added-mass/damping entry — so
        callers get a single, typed failure mode for bad configs.
        """
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict) or "links" not in data:
            raise ConfigError("config must be a mapping with a 'links' list")
        density = float(data.get("density", RHO_SEAWATER))
        links = tuple(_parse_link(raw) for raw in data["links"])
        if not links:
            raise ConfigError("config must declare at least one link")
        return RobotHydroConfig(links=links, density=density)


def _parse_link(raw: dict[str, Any]) -> LinkConfig:
    """Build a validated :class:`LinkConfig` from one raw YAML link mapping.

    Applies the per-field invariants the dataclass itself cannot enforce:
    non-negative volume, a 3-vector center of buoyancy, and 6-or-36-element
    damping vectors. Missing fields fall back to inert defaults (zero volume,
    origin CoB, zero damping) so a sparse config still loads.
    """
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
    lift = _parse_lift(name, raw["lift"]) if raw.get("lift") is not None else None
    return LinkConfig(
        name=name,
        volume=volume,
        center_of_buoyancy=tuple(float(c) for c in cob),  # type: ignore[arg-type]
        added_mass=am,
        linear_damping=tuple(float(v) for v in lin),
        quadratic_damping=tuple(float(v) for v in quad),
        lift=lift,
    )


def _parse_lift(name: str, raw: dict[str, Any]) -> LiftSpec:
    axes = raw.get("semi_axes")
    if axes is None or len(axes) != 3:
        raise ConfigError(f"link '{name}': lift semi_axes must have 3 elements")
    axes = tuple(float(a) for a in axes)
    if any(a <= 0.0 for a in axes):
        raise ConfigError(f"link '{name}': lift semi_axes must all be > 0, got {axes}")
    return LiftSpec(
        semi_axes=axes,
        c_kutta=float(raw.get("c_kutta", 1.0)),
        c_magnus=float(raw.get("c_magnus", 1.0)),
    )


def _validate_damping(name: str, key: str, vals: list[Any]) -> list[Any]:
    """Check a damping vector is a 6-element diagonal or a 36-element 6x6.

    Length is the only structural constraint here; the values themselves are
    coerced to ``float`` by the caller. Returns the list unchanged so it can be
    used directly in the enclosing :class:`LinkConfig` construction.
    """
    if len(vals) not in (6, 36):
        raise ConfigError(f"link '{name}': {key} must have 6 or 36 elements, got {len(vals)}")
    return vals


def _parse_added_mass(name: str, raw: dict[str, Any]) -> AddedMassSpec:
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


def _require_symmetric(name: str, m: list[Any]) -> None:
    """Reject a 36-element added-mass matrix that is not symmetric.

    A physical added-mass tensor must be symmetric (it derives from a quadratic
    kinetic-energy form); asymmetry indicates a hand-authored config error.
    Compares each ``M[i,j]`` against ``M[j,i]`` in row-major order with a small
    tolerance.
    """
    for i in range(6):
        for j in range(6):
            if abs(m[i * 6 + j] - m[j * 6 + i]) > 1e-9:
                raise ConfigError(f"link '{name}': added-mass matrix must be symmetric")
