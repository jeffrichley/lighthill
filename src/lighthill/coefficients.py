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
        assert spec.radius is not None and spec.length is not None and spec.axis is not None
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
    added = torch.stack([_resolve_added_mass(link.added_mass, config.density, dtype) for link in links])
    lin = torch.stack([_to_6x6(link.linear_damping, dtype) for link in links])
    quad = torch.stack([_to_6x6(link.quadratic_damping, dtype) for link in links])
    vol = torch.tensor([link.volume for link in links], dtype=dtype)
    cob = torch.tensor([link.center_of_buoyancy for link in links], dtype=dtype)
    neutral = torch.tensor([link.neutrally_buoyant for link in links], dtype=torch.bool)
    return ResolvedCoefficients(
        added_mass=added, linear_damping=lin, quadratic_damping=quad,
        volume=vol, center_of_buoyancy=cob, neutrally_buoyant=neutral,
        density=config.density, names=tuple(link.name for link in links),
    )
