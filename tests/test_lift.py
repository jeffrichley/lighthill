"""Closed-form oracles for the ellipsoid Kutta + Magnus lift kernel (forces.lift_wrench).

The formulas are transcribed verbatim from MuJoCo's ellipsoid fluid model:
  f_K = C_K rho A_proj (v_hat . n_hat) (n_hat x v) x v      (Kutta, circulatory lift)
  f_M = C_M rho V (omega x v)                                (Magnus, spin lift)
Lift is force-only (no torque). These tests pin the kernel against hand-computed
values and the structural invariants the physics guarantees (lift is perpendicular to
flow; Kutta vanishes for a sphere and at zero angle of attack; nothing blows up at v=0).
"""

from __future__ import annotations

import math

import torch

from lighthill.forces import lift_wrench

RHO = 1000.0


def _wrench(v, omega, semi_axes, *, c_kutta=1.0, c_magnus=1.0, density=RHO):
    v_rel = torch.tensor([*v, *omega], dtype=torch.float64)
    r = torch.tensor(semi_axes, dtype=torch.float64)
    ck = torch.tensor(c_kutta, dtype=torch.float64)
    cm = torch.tensor(c_magnus, dtype=torch.float64)
    return lift_wrench(v_rel, r, ck, cm, density)


def test_magnus_closed_form():
    # sphere -> Kutta identically zero, so the force is pure Magnus.
    # f_M = C_M rho V (omega x v),  V = 4/3 pi,  omega x v = (0,0,2)x(3,0,0) = (0,6,0)
    w = _wrench(v=(3.0, 0.0, 0.0), omega=(0.0, 0.0, 2.0), semi_axes=(1.0, 1.0, 1.0))
    volume = 4.0 / 3.0 * math.pi
    expected = torch.tensor([0.0, RHO * volume * 6.0, 0.0], dtype=torch.float64)
    assert torch.allclose(w[0:3], expected, rtol=1e-9, atol=1e-6)
    assert torch.allclose(w[3:6], torch.zeros(3, dtype=torch.float64))  # lift adds no torque


def test_kutta_vanishes_for_sphere():
    # equal semi-axes -> surface normal is parallel to v -> (n_hat x v) = 0.
    w = _wrench(v=(1.0, 2.0, -0.5), omega=(0.0, 0.0, 0.0), semi_axes=(1.0, 1.0, 1.0),
                c_magnus=0.0)
    assert torch.allclose(w, torch.zeros(6, dtype=torch.float64), atol=1e-9)


def test_kutta_vanishes_at_zero_angle_of_attack():
    # flow straight along the long axis -> n_s parallel to v -> no Kutta lift.
    w = _wrench(v=(5.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0), semi_axes=(2.0, 1.0, 1.0),
                c_magnus=0.0)
    assert torch.allclose(w, torch.zeros(6, dtype=torch.float64), atol=1e-9)


def test_lift_is_perpendicular_to_flow():
    v = (1.0, 1.0, 0.0)
    w = _wrench(v=v, omega=(0.0, 0.0, 0.0), semi_axes=(2.0, 1.0, 1.0), c_magnus=0.0)
    f = w[0:3]
    assert f.norm() > 1.0  # non-trivial lift at angle of attack
    assert abs(float(torch.dot(f, torch.tensor(v, dtype=torch.float64)))) < 1e-6


def test_kutta_closed_form():
    # prolate ellipsoid r=(2,1,1), flow (1,1,0) at 45deg, hand-computed:
    #   n_s=(0.5,2,0), v_hat.n_hat=0.857508, (n_hat x v)x v=(0.727607,-0.727607,0),
    #   A_proj=pi*sqrt(3.4)=5.79266  ->  f_K=(3614.15, -3614.15, 0)
    w = _wrench(v=(1.0, 1.0, 0.0), omega=(0.0, 0.0, 0.0), semi_axes=(2.0, 1.0, 1.0),
                c_magnus=0.0)
    expected = torch.tensor([3614.15, -3614.15, 0.0], dtype=torch.float64)
    assert torch.allclose(w[0:3], expected, rtol=2e-4, atol=1.0)
    assert torch.allclose(w[3:6], torch.zeros(3, dtype=torch.float64))


def test_zero_velocity_is_finite_and_zero():
    w = _wrench(v=(0.0, 0.0, 0.0), omega=(1.0, 2.0, 3.0), semi_axes=(2.0, 1.0, 1.0))
    assert torch.isfinite(w).all()
    assert torch.allclose(w, torch.zeros(6, dtype=torch.float64), atol=1e-9)


def test_batched_matches_looped():
    # the kernel must vectorize over leading [E,B] dims identically to per-element calls.
    torch.manual_seed(0)
    v_rel = torch.randn(4, 3, 6, dtype=torch.float64)
    r = torch.rand(4, 3, 3, dtype=torch.float64) + 0.5
    ck = torch.rand(4, 3, dtype=torch.float64)
    cm = torch.rand(4, 3, dtype=torch.float64)
    batched = lift_wrench(v_rel, r, ck, cm, RHO)
    for i in range(4):
        for j in range(3):
            one = lift_wrench(v_rel[i, j], r[i, j], ck[i, j], cm[i, j], RHO)
            assert torch.allclose(batched[i, j], one, rtol=1e-9, atol=1e-9)
