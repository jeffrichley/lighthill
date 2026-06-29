import math

import torch

from lighthill.coefficients import (
    cylinder_added_mass,
    resolve_coefficients,
    sphere_added_mass,
)
from lighthill.config import AddedMassSpec, LinkConfig, RobotHydroConfig


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


def test_zero_length_cylinder_resolves_to_zero_added_mass():
    cfg = RobotHydroConfig(links=(_link(
        added_mass=AddedMassSpec(kind="cylinder", radius=0.02, length=0.0, axis="z")),))
    rc = resolve_coefficients(cfg)
    assert torch.allclose(rc.added_mass[0], torch.zeros(6, 6))
