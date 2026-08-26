"""Tests for the coupled floating-base 2-body Featherstone reference (Task 6).

The reference integrates a free-floating base + one revolute-jointed arm link with
a *commanded* joint trajectory, computing the analytical base reaction to an arm
swing. It is the correctness-critical artifact the in-sim arm-swing gate compares
against. These tests certify the physics (no Isaac): the arm kinematics, momentum
conservation under an arm swing (the coupling), the single-body limit, and static
equilibrium.
"""

from __future__ import annotations

import math

import torch

from lighthill.validation.reference import Body
from lighthill.validation.reference_coupled import (
    TwoBodyChain,
    arm_kinematics,
    simulate_coupled,
)


def _cosine_swing(amp: float, omega: float):
    """q(t)=amp*(1-cos(omega t)): smooth start (q,qd=0 at t=0)."""
    def traj(t: float) -> tuple[float, float, float]:
        q = amp * (1.0 - math.cos(omega * t))
        qd = amp * omega * math.sin(omega * t)
        qdd = amp * omega * omega * math.cos(omega * t)
        return q, qd, qdd
    return traj


def _diag6(vals):
    return torch.diag(torch.tensor(vals, dtype=torch.float32))


def _zero_hydro_link(mass, inertia):
    """A link with all hydro coefficients zeroed (pure rigid body)."""
    return Body(
        mass=mass, inertia=inertia, volume=0.0, cob=(0.0, 0.0, 0.0),
        added_mass=torch.zeros(6, 6), linear_damping=torch.zeros(6, 6),
        quadratic_damping=torch.zeros(6, 6), density=1025.0,
    )


def _simple_chain():
    """base + arm, revolute about base-frame Y, anchors chosen for hand-checked kinematics."""
    base = _zero_hydro_link(13.7, (0.3, 0.3, 0.3))
    arm = _zero_hydro_link(0.6, (0.01, 0.05, 0.01))
    return TwoBodyChain(
        base=base, arm=arm,
        joint_axis=(0.0, 1.0, 0.0),
        anchor_base=(0.0, 0.0, -0.15),
        anchor_arm=(0.0, 0.0, 0.25),
    )


def test_kinematics_at_rest_identity_base():
    chain = _simple_chain()
    quat0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    v0 = torch.zeros(3)
    w0 = torch.zeros(3)
    k = arm_kinematics(quat0, v0, w0, q=0.0, qd=0.0, qdd=0.0, chain=chain)
    # arm CoM offset from base CoM = anchor_base - anchor_arm = (0,0,-0.40)
    assert torch.allclose(k["r"], torch.tensor([0.0, 0.0, -0.40]), atol=1e-6)
    assert torch.allclose(k["n_hat"], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    assert torch.allclose(k["v1_w"], torch.zeros(3), atol=1e-6)
    assert torch.allclose(k["omega1_w"], torch.zeros(3), atol=1e-6)


def test_kinematics_joint_rate_gives_arm_velocity():
    chain = _simple_chain()
    quat0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    v0 = torch.zeros(3)
    w0 = torch.zeros(3)
    k = arm_kinematics(quat0, v0, w0, q=0.0, qd=2.0, qdd=0.0, chain=chain)
    # omega1 = axis * qd
    assert torch.allclose(k["omega1_w"], torch.tensor([0.0, 2.0, 0.0]), atol=1e-6)
    # v1 = R0 * ddot_d/dq * qd ; with axis=Y, anchor_arm=(0,0,0.25):
    #   d'(q)= -skew(axis) R_rel(q) anchor_arm ; at q=0 -> -skew(Y)@(0,0,0.25)=-(0.25,0,0)
    #   v1 = d'(0)*qd = (-0.25,0,0)*2 = (-0.5,0,0)
    assert torch.allclose(k["v1_w"], torch.tensor([-0.5, 0.0, 0.0]), atol=1e-6)


def test_kinematics_joint_accel_bias_terms():
    chain = _simple_chain()
    quat0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    v0 = torch.zeros(3)
    w0 = torch.zeros(3)
    k = arm_kinematics(quat0, v0, w0, q=0.0, qd=0.0, qdd=1.0, chain=chain)
    # k2 (arm angular accel bias) = axis * qdd = (0,1,0)
    assert torch.allclose(k["k2"], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    # k1 (arm linear accel bias) = R0 d''... at rest reduces to d'(0)*qdd = (-0.25,0,0)
    assert torch.allclose(k["k1"], torch.tensor([-0.25, 0.0, 0.0]), atol=1e-6)


def test_free_chain_conserves_momentum_under_arm_swing():
    """No external forces: an arm swing must leave total linear & angular momentum
    unchanged (==0 from rest) while the base recoils. This is the coupling claim."""
    chain = _simple_chain()
    out = simulate_coupled(
        chain, steps=4000, dt=0.001, q_traj=_cosine_swing(amp=0.6, omega=4.0),
        use_gravity=False, use_buoyancy=False,
    )
    p = out["lin_momentum"]       # [steps,3]
    ell = out["ang_momentum"]     # [steps,3]
    # peak arm momentum sets the scale; drift must be tiny against it
    arm_mom_scale = (chain.arm.mass * out["arm_vel_w"].norm(dim=-1)).max()
    assert arm_mom_scale > 0.05, "arm should actually be moving"
    assert p.norm(dim=-1).max() < 0.02 * arm_mom_scale
    assert ell.norm(dim=-1).max() < 0.02 * arm_mom_scale
    # base must recoil (translation and/or rotation)
    assert out["base_vel_w"].norm(dim=-1).max() > 1e-3
    assert out["base_omega_w"].norm(dim=-1).max() > 1e-2


def test_momentum_drift_is_first_order_in_dt():
    """The momentum residual is semi-implicit-Euler truncation, not a modeling error:
    it must vanish ~linearly with dt. A sign/coupling bug would leave a dt-independent
    residual. Halving dt must roughly halve the drift."""
    chain = _simple_chain()
    swing = _cosine_swing(amp=0.6, omega=4.0)

    def drift(dt: float) -> tuple[float, float]:
        out = simulate_coupled(chain, steps=int(2.0 / dt), dt=dt, q_traj=swing,
                               use_gravity=False, use_buoyancy=False)
        return (float(out["lin_momentum"].norm(dim=-1).max()),
                float(out["ang_momentum"].norm(dim=-1).max()))

    p_coarse, l_coarse = drift(0.002)
    p_fine, l_fine = drift(0.001)
    assert 1.7 < p_coarse / p_fine < 2.3
    assert 1.7 < l_coarse / l_fine < 2.3


def test_base_recoils_keeping_system_com_fixed():
    """Sanity on direction: with no external force the system CoM cannot move, so the
    free base CoM must translate opposite the arm. CoM displacement stays ~0."""
    chain = _simple_chain()
    out = simulate_coupled(
        chain, steps=600, dt=0.002, q_traj=_cosine_swing(amp=0.5, omega=4.0),
        use_gravity=False, use_buoyancy=False,
    )
    m0, m1 = chain.base.mass, chain.arm.mass
    com = (m0 * out["base_pos"] + m1 * out["arm_pos"]) / (m0 + m1)
    com_disp = (com - com[0]).norm(dim=-1).max()
    assert com_disp < 1e-3
    # and the base really did move opposite the arm in x (nonzero recoil)
    assert out["base_pos"][:, 0].abs().max() > 1e-3


def test_massless_arm_produces_no_base_reaction():
    """Single-body limit: as the arm mass/inertia -> 0 the reaction it transmits to
    the base -> 0, so commanding a swing leaves the base at rest."""
    base = _zero_hydro_link(13.7, (0.3, 0.3, 0.3))
    arm = _zero_hydro_link(1e-7, (1e-9, 1e-9, 1e-9))
    chain = TwoBodyChain(base, arm, (0.0, 1.0, 0.0), (0.0, 0.0, -0.15), (0.0, 0.0, 0.25))
    out = simulate_coupled(chain, steps=1000, dt=0.002, q_traj=_cosine_swing(0.6, 4.0),
                           use_gravity=False, use_buoyancy=False)
    assert out["base_vel_w"].norm(dim=-1).max() < 1e-5
    assert out["base_omega_w"].norm(dim=-1).max() < 1e-5


def _neutral_link(mass, inertia, rho=1025.0):
    """Trimmed link: buoyancy (V=m/rho) cancels weight, CoB at CoM (no couple)."""
    return Body(
        mass=mass, inertia=inertia, volume=mass / rho, cob=(0.0, 0.0, 0.0),
        added_mass=torch.zeros(6, 6), linear_damping=torch.zeros(6, 6),
        quadratic_damping=torch.zeros(6, 6), density=rho,
    )


def test_trimmed_static_equilibrium_base_stays_put():
    """Frozen arm, gravity + buoyancy both on but trimmed per link (weight==buoyancy,
    CoB==CoM): every link's net wrench is zero, so the free base must not drift.
    Exercises the gravity + buoyancy wiring."""
    chain = TwoBodyChain(
        _neutral_link(13.7, (0.3, 0.3, 0.3)), _neutral_link(0.6, (0.01, 0.05, 0.01)),
        (0.0, 1.0, 0.0), (0.0, 0.0, -0.15), (0.0, 0.0, 0.25))
    out = simulate_coupled(chain, steps=1500, dt=0.002,
                           q_traj=lambda _t: (0.0, 0.0, 0.0),
                           use_gravity=True, use_buoyancy=True)
    disp = (out["base_pos"] - out["base_pos"][0]).norm(dim=-1).max()
    assert disp < 1e-3
    assert out["base_vel_w"].norm(dim=-1).max() < 1e-3


def test_gravity_without_buoyancy_makes_base_sink():
    """Control for the equilibrium test: remove buoyancy and the (now untrimmed)
    base must fall under gravity -- confirms gravity is actually applied."""
    chain = TwoBodyChain(
        _neutral_link(13.7, (0.3, 0.3, 0.3)), _neutral_link(0.6, (0.01, 0.05, 0.01)),
        (0.0, 1.0, 0.0), (0.0, 0.0, -0.15), (0.0, 0.0, 0.25))
    out = simulate_coupled(chain, steps=500, dt=0.002,
                           q_traj=lambda _t: (0.0, 0.0, 0.0),
                           use_gravity=True, use_buoyancy=False)
    assert out["base_pos"][-1, 2] < -0.05  # sank in -Z


def test_hydro_drag_breaks_conservation_and_pushes_system():
    """With per-link drag + anisotropic added-mass residual active, the swinging arm
    sheds momentum to the fluid: total body momentum is no longer conserved and the
    system CoM is pushed. Exercises the drag / Coriolis / residual kernels in the
    coupled path (the residual is the anisotropic-linear added-mass remainder)."""
    base = Body(
        mass=13.7, inertia=(0.3, 0.3, 0.3), volume=0.0, cob=(0.0, 0.0, 0.0),
        added_mass=_diag6([6.36, 7.12, 18.68, 0.189, 0.135, 0.222]),
        linear_damping=_diag6([13.7, 20.0, 33.0, 1.0, 1.0, 1.0]),
        quadratic_damping=_diag6([141.0, 217.0, 190.0, 1.2, 0.5, 1.5]), density=1025.0)
    arm = Body(
        mass=0.6, inertia=(0.01, 0.05, 0.01), volume=0.0, cob=(0.0, 0.0, 0.0),
        added_mass=_diag6([2.0, 2.0, 0.5, 0.0, 0.0, 0.0]),
        linear_damping=_diag6([3.0, 3.0, 1.0, 0.01, 0.01, 0.01]),
        quadratic_damping=_diag6([12.0, 12.0, 4.0, 0.05, 0.05, 0.05]), density=1025.0)
    chain = TwoBodyChain(base, arm, (0.0, 1.0, 0.0), (0.0, 0.0, -0.15), (0.0, 0.0, 0.25))
    out = simulate_coupled(chain, steps=3000, dt=0.001, q_traj=_cosine_swing(0.6, 4.0),
                           use_gravity=False, use_buoyancy=False)
    m0, m1 = 13.7 + 6.36, 0.6 + 0.5  # effective masses (rigid + isotropic added-mass bump)
    com = (m0 * out["base_pos"] + m1 * out["arm_pos"]) / (m0 + m1)
    # drag pushes the whole system far beyond the conserved-case truncation floor (~4e-4)
    assert (com - com[0]).norm(dim=-1).max() > 5e-3
    # momentum no longer conserved (well above the ~1e-3 no-drag floor)
    assert out["lin_momentum"].norm(dim=-1).max() > 0.05
