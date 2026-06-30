import torch

from lighthill.constants import GRAVITY, RHO_SEAWATER
from lighthill.forces import buoyancy_wrench, drag_wrench

IDQUAT = torch.tensor([1.0, 0.0, 0.0, 0.0])


def test_buoyancy_upright_pushes_world_up_in_body_frame():
    V = torch.tensor(0.01)
    w = buoyancy_wrench(IDQUAT, V, torch.zeros(3), RHO_SEAWATER)
    expected_fz = RHO_SEAWATER * GRAVITY * 0.01
    assert torch.allclose(w[:3], torch.tensor([0.0, 0.0, expected_fz]), atol=1e-4)
    assert torch.allclose(w[3:], torch.zeros(3), atol=1e-6)  # cob at origin -> no moment


def test_buoyancy_offset_cob_produces_exact_restoring_couple():
    # The case the deleted neutrally_buoyant flag silently dropped: an offset CoB
    # must yield the full couple cob x F_b regardless of any net-neutrality.
    # Upright, CoB offset +0.1 m in surge (+x). Buoyancy F = rho*g*V acts world-up
    # (== body +z when upright), so the couple is exactly cob x F = (0, -dx*F, 0).
    V = torch.tensor(0.01)
    dx = 0.1
    cob = torch.tensor([dx, 0.0, 0.0])
    w = buoyancy_wrench(IDQUAT, V, cob, RHO_SEAWATER)
    F = RHO_SEAWATER * GRAVITY * 0.01
    assert torch.allclose(w[:3], torch.tensor([0.0, 0.0, F]), atol=1e-3)
    assert torch.allclose(w[3:], torch.tensor([0.0, -dx * F, 0.0]), atol=1e-3)


def test_buoyancy_with_cob_offset_makes_restoring_moment():
    # cob 0.02 m above origin (+z); upright -> force +z through a point on +z axis -> zero moment
    V = torch.tensor(0.01)
    cob = torch.tensor([0.0, 0.0, 0.02])
    w_up = buoyancy_wrench(IDQUAT, V, cob, RHO_SEAWATER)
    assert torch.allclose(w_up[3:], torch.zeros(3), atol=1e-6)
    # roll 90 deg about x: world-up now along body -y; force x cob -> nonzero moment about... check magnitude
    import math
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    q_roll = torch.tensor([c, s, 0.0, 0.0])  # 90 deg about x
    w_tilt = buoyancy_wrench(q_roll, V, cob, RHO_SEAWATER)
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
    w = buoyancy_wrench(q, V, cob, RHO_SEAWATER)
    assert w.shape == (4, 6)
