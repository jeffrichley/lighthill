"""Coupled floating-base 2-body Featherstone reference (Plan B, Task 6).

The single-body integrator in ``reference.py`` validates the per-link force law on
one rigid body. This module extends it to a **floating-base 2-body chain** -- a free
vehicle (base) plus one revolute-jointed arm link -- with a *commanded* joint
trajectory, and computes the analytical reaction of the free base to an arm swing.
That base reaction is the headline UVMS coupling claim; this is the analytical
reference the in-sim Isaac gate is held against.

Modeling (kept identical to the in-sim path so the gate isolates the *coupling*,
not the force law -- the force law itself is already certified by the single-body
scenarios):

* Per-link hydro wrench is computed by the **same** lighthill kernels the Isaac
  adapter drives (buoyancy + drag + added-mass Coriolis + off-diagonal/anisotropic
  residual via the same EMA acceleration filter), and the diagonal added mass is
  folded into each link's effective mass/inertia exactly as
  ``set_body_inertias`` does in sim.
* The only *new* physics here is the floating-base coupling: the joint angle is
  prescribed, the base is free, and the 6 base-DOF accelerations are solved from
  the system's composite spatial inertia about the base CoM. The vehicle<->arm
  reaction emerges from that solve -- exactly what the gate tests.

NWU frame, scalar-first (w,x,y,z) quaternions, body-frame twists -- same as Plan A.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from ..accel import AccelerationFilter
from ..constants import GRAVITY
from ..forces import (
    added_mass_coriolis,
    added_mass_residual,
    buoyancy_wrench,
    drag_wrench,
)
from ..frames import quat_to_rotation_matrix, skew
from ..inertia import effective_inertia, split_added_mass
from .reference import Body, _integrate_quat


@dataclass
class TwoBodyChain:
    """A free base + one revolute-jointed arm link.

    ``joint_axis`` is the revolute axis in the **base** body frame; ``anchor_base``
    and ``anchor_arm`` are the joint anchor point expressed in the base and arm body
    frames respectively. At joint angle ``q`` the arm body frame is ``R_rel(q)``
    (rotation about ``joint_axis`` by ``q``) relative to the base, positioned so the
    two anchors coincide.
    """

    base: Body
    arm: Body
    joint_axis: tuple[float, float, float]
    anchor_base: tuple[float, float, float]
    anchor_arm: tuple[float, float, float]


def _axis_rotation(axis: Tensor, q: float) -> Tensor:
    """Rodrigues rotation about a unit ``axis`` by angle ``q`` -> [3,3]."""
    k = skew(axis)
    return torch.eye(3) + torch.sin(torch.tensor(q)) * k + (1.0 - torch.cos(torch.tensor(q))) * (k @ k)


def arm_kinematics(quat0: Tensor, v0_w: Tensor, omega0_w: Tensor,
                   q: float, qd: float, qdd: float, chain: TwoBodyChain) -> dict[str, Tensor]:
    """Arm-link world kinematics given base state + the commanded joint motion.

    Returns a dict with the world-frame arm rotation ``R1``, the base->arm CoM offset
    ``r`` (world), the world joint axis ``n_hat``, the arm CoM velocity ``v1_w`` and
    angular velocity ``omega1_w``, and the joint-driven acceleration biases ``k1``
    (linear) and ``k2`` (angular) -- the parts of the arm acceleration that do *not*
    depend on the (still unknown) base acceleration.
    """
    R0 = quat_to_rotation_matrix(quat0)
    axis = torch.tensor(chain.joint_axis, dtype=torch.float32)
    aa = torch.tensor(chain.anchor_arm, dtype=torch.float32)
    ab = torch.tensor(chain.anchor_base, dtype=torch.float32)
    k_ax = skew(axis)

    r_rel = _axis_rotation(axis, q)
    d = ab - r_rel @ aa                       # arm CoM offset in base frame
    d_p = -(k_ax @ r_rel @ aa)                # d'(q)
    d_pp = -(k_ax @ k_ax @ r_rel @ aa)        # d''(q)
    d_dot = d_p * qd                          # base frame
    d_ddot = d_p * qdd + d_pp * qd * qd       # base frame

    r = R0 @ d
    n_hat = R0 @ axis
    R0_ddot = R0 @ d_dot
    R0_dddot = R0 @ d_ddot

    v1_w = v0_w + torch.cross(omega0_w, r, dim=-1) + R0_ddot
    omega1_w = omega0_w + n_hat * qd
    k1 = (torch.cross(omega0_w, torch.cross(omega0_w, r, dim=-1), dim=-1)
          + 2.0 * torch.cross(omega0_w, R0_ddot, dim=-1) + R0_dddot)
    k2 = torch.cross(omega0_w, n_hat, dim=-1) * qd + n_hat * qdd

    return {
        "R0": R0, "R1": R0 @ r_rel, "r": r, "n_hat": n_hat,
        "v1_w": v1_w, "omega1_w": omega1_w, "k1": k1, "k2": k2,
    }


@dataclass
class _LinkModel:
    """Per-link effective inertia + residual matrix, precomputed once (matches sim init)."""

    mass_eff: float          # rigid mass + isotropic added-mass bump (the PhysX scalar mass)
    inertia_eff: Tensor      # [3] principal effective inertia (body frame)
    residual: Tensor         # [6,6] anisotropic-linear + off-diagonal added-mass remainder
    body: Body


def _build_link_model(body: Body) -> _LinkModel:
    routing = split_added_mass(body.added_mass.unsqueeze(0))
    m_eff, i_eff = effective_inertia(
        torch.tensor([body.mass]), torch.tensor([body.inertia]), routing)
    return _LinkModel(mass_eff=float(m_eff[0]), inertia_eff=i_eff[0],
                      residual=routing.residual[0], body=body)


def _link_wrench_world(model: _LinkModel, quat_wb: Tensor, twist_body: Tensor,
                       accel_body: Tensor, *, use_buoyancy: bool) -> tuple[Tensor, Tensor]:
    """External hydro wrench on a link (no gravity), returned as world (F, M) at the CoM.

    Uses the *same* lighthill kernels the Isaac adapter drives: buoyancy + drag +
    added-mass Coriolis + off-diagonal/anisotropic residual (filtered accel).
    """
    body = model.body
    w = torch.zeros(6)
    if use_buoyancy:
        w = w + buoyancy_wrench(
            quat_wb, torch.tensor(body.volume), torch.tensor(body.cob),
            body.density)
    w = w + drag_wrench(twist_body, body.linear_damping, body.quadratic_damping)
    w = w + added_mass_coriolis(body.added_mass, twist_body)
    w = w + added_mass_residual(model.residual, accel_body)
    R = quat_to_rotation_matrix(quat_wb)
    f_world = R @ w[0:3]
    m_world = R @ w[3:6]
    return f_world, m_world


def simulate_coupled(
    chain: TwoBodyChain,
    *,
    steps: int,
    dt: float,
    q_traj: Callable[[float], tuple[float, float, float]],
    gravity: float = GRAVITY,
    use_gravity: bool = True,
    use_buoyancy: bool = True,
    alpha: float = 0.08,
    pos0: Tensor | None = None,
    quat0: Tensor | None = None,
) -> dict[str, Tensor]:
    """Integrate the free base + commanded arm; return base/arm trajectories + momenta.

    The joint angle ``q(t)`` is prescribed by ``q_traj`` (returns ``(q, qd, qdd)``);
    the base is free. Each step solves the 6 base-DOF world accelerations from the
    system composite spatial inertia about the base CoM (the only point where the
    vehicle<->arm reaction is computed), then semi-implicit-Euler integrates the base.

    Gravity (when enabled) acts on the **effective** mass, matching PhysX, which
    applies gravity on the augmented mass written by ``set_body_inertias``.
    """
    base_m = _build_link_model(chain.base)
    arm_m = _build_link_model(chain.arm)
    m0, m1 = base_m.mass_eff, arm_m.mass_eff
    eye3 = torch.eye(3)

    p0 = torch.zeros(3) if pos0 is None else pos0.clone()
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0]) if quat0 is None else quat0.clone()
    v0_w = torch.zeros(3)
    w0_w = torch.zeros(3)
    filt = AccelerationFilter(shape=(1, 2), alpha=alpha)

    rec: dict[str, list[Tensor]] = {
        "base_pos": [], "base_quat": [], "base_vel_w": [], "base_omega_w": [],
        "arm_pos": [], "arm_vel_w": [], "lin_momentum": [], "ang_momentum": [],
        "q": [],
    }
    for step in range(steps):
        t = step * dt
        q, qd, qdd = q_traj(t)
        kin = arm_kinematics(quat, v0_w, w0_w, q, qd, qdd, chain)
        R0, R1 = kin["R0"], kin["R1"]
        r = kin["r"]
        v1_w, w1_w, k1, k2 = kin["v1_w"], kin["omega1_w"], kin["k1"], kin["k2"]
        p1 = p0 + r

        # body-frame twists for the hydro kernels
        nu0 = torch.cat([R0.T @ v0_w, R0.T @ w0_w])
        nu1 = torch.cat([R1.T @ v1_w, R1.T @ w1_w])
        a_filt = filt.update(torch.stack([nu0, nu1]).unsqueeze(0), dt)[0]  # [2,6]

        quat_arm = _quat_of(R1)
        f0, t0 = _link_wrench_world(base_m, quat, nu0, a_filt[0], use_buoyancy=use_buoyancy)
        f1, t1 = _link_wrench_world(arm_m, quat_arm, nu1, a_filt[1], use_buoyancy=use_buoyancy)
        if use_gravity:
            f0 = f0 + torch.tensor([0.0, 0.0, -m0 * gravity])
            f1 = f1 + torch.tensor([0.0, 0.0, -m1 * gravity])

        # world inertia tensors (effective)
        I0_w = R0 @ torch.diag(base_m.inertia_eff) @ R0.T
        I1_w = R1 @ torch.diag(arm_m.inertia_eff) @ R1.T
        sr = skew(r)

        # composite spatial inertia about the base CoM O
        A = torch.zeros(6, 6)
        A[0:3, 0:3] = (m0 + m1) * eye3
        A[0:3, 3:6] = -m1 * sr
        A[3:6, 0:3] = m1 * sr
        A[3:6, 3:6] = I0_w + I1_w - m1 * (sr @ sr)

        rhs_lin = (f0 + f1) - m1 * k1
        t_o = t0 + t1 + torch.linalg.cross(r, f1)
        rdot = v1_w - v0_w
        bias_ang = (torch.linalg.cross(w0_w, I0_w @ w0_w) + I1_w @ k2
                    + torch.linalg.cross(w1_w, I1_w @ w1_w)
                    + m1 * (torch.linalg.cross(rdot, v1_w) + torch.linalg.cross(r, k1)))
        rhs_ang = t_o - bias_ang

        sol = torch.linalg.solve(A, torch.cat([rhs_lin, rhs_ang]))
        a0, alpha0 = sol[0:3], sol[3:6]

        # record the pre-update snapshot (consistent sampling)
        rec["base_pos"].append(p0.clone())
        rec["base_quat"].append(quat.clone())
        rec["base_vel_w"].append(v0_w.clone())
        rec["base_omega_w"].append(w0_w.clone())
        rec["arm_pos"].append(p1.clone())
        rec["arm_vel_w"].append(v1_w.clone())
        rec["lin_momentum"].append(m0 * v0_w + m1 * v1_w)
        ell = (I0_w @ w0_w + I1_w @ w1_w
               + m0 * torch.linalg.cross(p0, v0_w) + m1 * torch.linalg.cross(p1, v1_w))
        rec["ang_momentum"].append(ell)
        rec["q"].append(torch.tensor(q))

        # semi-implicit Euler integrate the base
        v0_w = v0_w + a0 * dt
        w0_w = w0_w + alpha0 * dt
        p0 = p0 + v0_w * dt
        quat = _integrate_quat(quat, R0.T @ w0_w, dt)

    return {k: torch.stack(v) for k, v in rec.items()}


def _quat_of(rot: Tensor) -> Tensor:
    """Rotation matrix [3,3] -> scalar-first (w,x,y,z) quaternion [4]."""
    m = rot
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = torch.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = torch.stack([w, x, y, z])
    return q / q.norm()
