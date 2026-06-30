"""Per-link hydrodynamics config schema + validated YAML loader."""

from __future__ import annotations

from dataclasses import dataclass
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
    added_mass: AddedMassSpec
    linear_damping: tuple[float, ...]
    quadratic_damping: tuple[float, ...]


@dataclass(frozen=True)
class RobotHydroConfig:
    links: tuple[LinkConfig, ...]
    density: float = RHO_SEAWATER

    @staticmethod
    def from_yaml(path: str | Path) -> RobotHydroConfig:
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
