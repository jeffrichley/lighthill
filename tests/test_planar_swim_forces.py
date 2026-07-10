"""Step-3 probe: single-segment planar-swim forces, checked against Fossen by hand.

Context (anguilla swim-direction investigation). The Newton sim swims the snake
TAIL-first (wrong) and drifts out of the yaw plane in z, while our independent RFT
oracle -- run on the sim's OWN realized joint angles -- swims HEAD-first. The
kinematics are therefore faithful (Step 2), so the discrepancy is in the FORCES.

This module isolates lighthill's force kernel from Isaac and from the CPG: one
segment, one hand-set velocity/orientation, the force read straight out of
``drag_wrench`` / ``compute_wrench`` / ``apply`` and asserted against Fossen. No
Isaac, no container, no oracle -- lighthill's REAL kernel, inputs we control.

Segment drag (matches anguilla's snake): axial c_t = 2, transverse c_n = 40
(anisotropy c_n/c_t = 20). Predictions:

  * lateral (+y) velocity -> force pure -y, |F| = 40*|v|*v, with x == z == 0
  * axial   (+x) velocity -> force pure -x, |F| =  2*|v|*v  (20x weaker)
  * planar (yaw-plane) motion NEVER produces a z-force  <- the sim's z-drift symptom
  * a yaw rotation keeps a planar body force planar in world (apply's R)

A PASS exonerates the physics kernel and points the swim-direction + z-drift bug at
the Newton adapter's state feed (apply_newton.py); a FAIL localizes it right here.
"""

from __future__ import annotations

import math

import torch

from lighthill.apply import UnderwaterHydrodynamics
from lighthill.articulation import FakeArticulation
from lighthill.coefficients import resolve_coefficients
from lighthill.config import AddedMassSpec, LinkConfig, RobotHydroConfig
from lighthill.forces import drag_wrench

SEG_LEN, SEG_W = 0.15, 0.08
C_AXIAL, C_TRANS = 2.0, 40.0  # quadratic drag: low along the body, high across it
QUAD = (C_AXIAL, C_TRANS, C_TRANS, 0.2, 0.2, 0.2)
V = 0.3  # a representative segment speed, m/s


def _segment(added_mass: bool):
    """One snake segment's resolved coefficients; added mass on/off to isolate drag."""
    am = (
        AddedMassSpec(kind="cylinder", radius=SEG_W / 2, length=SEG_LEN, axis="x")
        if added_mass
        else AddedMassSpec(kind="matrix", matrix=(0.0,) * 6)
    )
    link = LinkConfig(
        name="seg",
        volume=0.0,
        center_of_buoyancy=(0.0, 0.0, 0.0),
        added_mass=am,
        linear_damping=(0.0,) * 6,
        quadratic_damping=QUAD,
    )
    return resolve_coefficients(RobotHydroConfig(links=(link,)))


def _yaw_quat(theta: float) -> torch.Tensor:
    """Body->world quaternion for a yaw of theta about world +z (wxyz), shaped [1,1,4]."""
    return torch.tensor([[[math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)]]])


# --- Level 1: the drag kernel directly (exact Fossen numbers) ----------------


def test_lateral_drag_is_pure_lateral_and_opposing():
    """Moving +y: drag is pure -y at 40*v^2, with EXACTLY zero fore-aft and zero z."""
    d_zero = torch.zeros(6, 6)
    d_quad = torch.diag(torch.tensor(QUAD))
    v = torch.tensor([0.0, V, 0.0, 0.0, 0.0, 0.0])
    w = drag_wrench(v, d_zero, d_quad)
    assert torch.allclose(
        w, torch.tensor([0.0, -C_TRANS * V * V, 0.0, 0.0, 0.0, 0.0]), atol=1e-6
    )
    assert w[2].abs() < 1e-9  # the z-drift smoking gun: planar velocity -> zero z-force


def test_axial_drag_is_20x_weaker_than_lateral():
    """Same speed axially vs laterally -> axial drag is c_axial/c_trans = 1/20 as large."""
    d_zero = torch.zeros(6, 6)
    d_quad = torch.diag(torch.tensor(QUAD))
    w_ax = drag_wrench(torch.tensor([V, 0.0, 0.0, 0.0, 0.0, 0.0]), d_zero, d_quad)
    w_lat = drag_wrench(torch.tensor([0.0, V, 0.0, 0.0, 0.0, 0.0]), d_zero, d_quad)
    assert w_ax[0] < 0 and w_lat[1] < 0  # both oppose motion
    assert torch.isclose(
        w_ax[0].abs() / w_lat[1].abs(), torch.tensor(C_AXIAL / C_TRANS), atol=1e-5
    )


# --- Level 2: the full kernel on one segment (compute_wrench / apply) ---------


def test_segment_lateral_velocity_planar_and_opposing_identity():
    """Full compute_wrench, added mass off: a +y segment velocity gives pure -y drag,
    no fore-aft leak, and (the symptom under test) no vertical force."""
    coeffs = _segment(added_mass=False)
    art = FakeArticulation(1, 1)
    art.set_body_velocity(torch.tensor([[[0.0, V, 0.0, 0.0, 0.0, 0.0]]]))
    hydro = UnderwaterHydrodynamics(art, coeffs)
    hydro.reset(current_world=torch.zeros(1, 3))
    f = hydro.compute_wrench(dt=0.005)[0, 0, 0:3]
    assert f[1] < 0  # drag opposes +y
    assert f[0].abs() < 1e-6 and f[2].abs() < 1e-6  # no fore-aft, no vertical
    assert torch.isclose(f[1], torch.tensor(-C_TRANS * V * V), atol=1e-5)


def test_yaw_rotated_segment_keeps_force_in_world_plane():
    """A segment yawed 40 deg, moving laterally in its body frame: apply() rotates the
    body drag to WORLD via R. A yaw about z MUST keep the world force in the xy-plane --
    world z stays 0. A nonzero world z here would BE the sim's out-of-plane drift."""
    coeffs = _segment(added_mass=False)
    art = FakeArticulation(1, 1)
    art.set_body_quat(_yaw_quat(math.radians(40.0)))
    art.set_body_velocity(torch.tensor([[[0.0, V, 0.0, 0.0, 0.0, 0.0]]]))
    hydro = UnderwaterHydrodynamics(art, coeffs)
    hydro.reset(current_world=torch.zeros(1, 3))
    hydro.apply(dt=0.005)
    w = art.last_wrench[0, 0]
    assert w[2].abs() < 1e-6  # world z-force is zero for yaw-plane motion
    assert w[0:2].norm() > 1e-3  # ... and there IS a real in-plane drag force


def test_added_mass_path_no_z_force_for_planar_motion():
    """With added mass ON (as the sim runs it), a planar velocity + yaw rate must still
    give zero body-frame z-force: the coriolis/residual added-mass terms cannot leak out
    of the yaw plane. Second symptom probe for the sim's z-drift."""
    coeffs = _segment(added_mass=True)
    art = FakeArticulation(1, 1)
    art.set_body_velocity(torch.tensor([[[0.2, V, 0.0, 0.0, 0.0, 1.5]]]))  # surge+sway+yaw
    hydro = UnderwaterHydrodynamics(art, coeffs)
    hydro.reset(current_world=torch.zeros(1, 3))
    w = hydro.compute_wrench(dt=0.005)[0, 0]
    assert w[2].abs() < 1e-6  # no vertical force from purely in-plane motion
