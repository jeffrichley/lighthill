"""Tests for the independent planar momentum-reconstruction reference (Task 6 referee).

This is the *third* model in the arm-swing verification: a from-scratch, pure-NumPy
planar (x, z, pitch-about-Y) solver that reconstructs the free base motion from
**momentum conservation alone** -- it enforces total linear and angular momentum = 0
at every step and solves the base velocity algebraically, with NO composite-inertia
matrix and NO acceleration integration. It shares only the joint *kinematics*
convention with ``reference_coupled`` (that is not the thing under test); its dynamics
derivation is fully independent, so agreement between the two is strong corroboration
and disagreement localizes a bug.

These tests certify the physics (no Isaac, no torch): the single-body limit, a
closed-form reaction-wheel case (the pure rotational coupling in dispute), and
integration correctness via a fixed system CoM.
"""

from __future__ import annotations

import math

import numpy as np

from sim_validation.reference_planar_momentum import simulate_planar_momentum


def _cosine_swing(amp: float, omega: float):
    """q(t)=amp*(1-cos(omega t)); returns (q, qd)."""
    def traj(t: float) -> tuple[float, float]:
        return amp * (1.0 - math.cos(omega * t)), amp * omega * math.sin(omega * t)
    return traj


def test_massless_arm_produces_no_base_reaction():
    """As arm mass/inertia -> 0 the reaction transmitted to the base -> 0: commanding a
    swing must leave the free base at rest (translation and pitch)."""
    out = simulate_planar_momentum(
        m0=13.7, m1=1e-8, I0yy=0.30, I1yy=1e-10,
        anchor_base=(0.0, 0.0, -0.045), anchor_arm=(0.0, 0.0, 0.125),
        q_traj=_cosine_swing(0.6, 4.0), steps=1000, dt=0.002,
    )
    assert np.abs(out["base_pitch"]).max() < 1e-6
    assert np.abs(out["base_x"]).max() < 1e-6
    assert np.abs(out["base_z"]).max() < 1e-6


def test_reaction_wheel_closed_form():
    """Arm CoM pinned to the base CoM (both anchors zero) -> pure reaction wheel, no
    translation. Angular-momentum conservation gives the exact closed form
    theta0(t) = -I1yy/(I0yy+I1yy) * q(t). This directly certifies the rotational
    coupling (the term in dispute between the Featherstone reference and PhysX)."""
    m0, m1, I0yy, I1yy = 13.7, 0.6, 0.30, 0.05
    amp, omega = 0.4, 2.0
    out = simulate_planar_momentum(
        m0=m0, m1=m1, I0yy=I0yy, I1yy=I1yy,
        anchor_base=(0.0, 0.0, 0.0), anchor_arm=(0.0, 0.0, 0.0),
        q_traj=_cosine_swing(amp, omega), steps=8000, dt=0.0005,
    )
    # no translation for a reaction wheel
    assert np.abs(out["base_x"]).max() < 1e-9
    assert np.abs(out["base_z"]).max() < 1e-9
    # closed form: theta0 = -k q, k = I1yy/(I0yy+I1yy), checked over the whole trajectory
    k = I1yy / (I0yy + I1yy)
    theta_expected = -k * out["q"]
    assert np.abs(out["base_pitch"] - theta_expected).max() < 1e-3
    # and it actually pitched a meaningful amount (not a trivially-zero pass)
    assert np.abs(out["base_pitch"]).max() > 1e-3


def test_system_com_stays_fixed_under_offset_swing():
    """With an offset arm (real gate geometry) and no external force, the system CoM
    cannot move. This checks the coupled velocity solve + position integration are
    self-consistent (translation AND rotation), not just the enforced momentum."""
    m0, m1 = 13.7, 0.6
    out = simulate_planar_momentum(
        m0=m0, m1=m1, I0yy=0.388, I1yy=0.0128,
        anchor_base=(0.0, 0.0, -0.045), anchor_arm=(0.0, 0.0, 0.125),
        q_traj=_cosine_swing(0.4, 2.0), steps=8000, dt=0.0005,
    )
    x1 = out["arm_x"]
    z1 = out["arm_z"]
    com_x = (m0 * out["base_x"] + m1 * x1) / (m0 + m1)
    com_z = (m0 * out["base_z"] + m1 * z1) / (m0 + m1)
    assert np.abs(com_x - com_x[0]).max() < 1e-4
    assert np.abs(com_z - com_z[0]).max() < 1e-4
    # base must actually recoil opposite the arm in x (nonzero, high-SNR)
    assert np.abs(out["base_x"]).max() > 1e-3
    assert np.abs(out["base_pitch"]).max() > 1e-3
