import math

import torch

from lighthill.constants import RHO_SEAWATER
from lighthill.validation.reference import Body, simulate, terminal_velocity_quadratic


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
