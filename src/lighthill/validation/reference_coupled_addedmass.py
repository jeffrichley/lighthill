"""Independent coupled reference WITH anisotropic added mass (momentum reconstruction).

Motivation: ``reference_coupled`` and the in-sim gate share lighthill's added-mass ROUTING
(min-axis fold + EMA-filtered explicit residual), so comparing them cannot referee whether the
coupled added-mass response is right -- the approximation cancels. This module is a genuinely
independent oracle: it reconstructs the free base's planar motion from **conservation of total
(rigid + added-mass) momentum**, carrying the FULL anisotropic per-link added mass with no
fold, no EMA, and no acceleration-level solve. It shares only the joint kinematics convention
with ``reference_coupled`` (not the thing under test), exactly like
``reference_planar_momentum`` -- which this generalizes (set all added mass to 0 and this
reduces algebraically to that rigid reference).

Method (planar: x-z translation, rotation about world +Y; gate is gravity/buoyancy/drag-free,
so total linear & angular momentum stay exactly zero from rest):

  Each body i carries added momentum M_A,i @ nu_i in its body frame. In the plane the relevant
  added masses are surge ``Xu`` (body-x), heave ``Zw`` (body-z) and pitch ``Mq`` (about body-y).
  The world-frame linear added-mass 2x2 is ``A_i(theta_i) = R(theta_i) diag(Xu,Zw) R(theta_i)^T``,
  so body i's total linear momentum is ``(m_i I2 + A_i) v_i`` and its spin momentum is
  ``(I_i + Mq_i) w_i``. Enforcing P_x = P_z = L_y = 0 is a 3x3 linear system in the base
  velocity (v0x, v0z, omega0) once the arm velocity is written via the joint kinematics; solving
  and integrating reconstructs the base pose.

Torch float64 (lighthill's stack; numpy is not a dependency here). Imports NO lighthill
added-mass code -> independent of the routing under test.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

_F64 = torch.float64


def _r2(theta: float) -> torch.Tensor:
    """World<-body 2x2 rotation for (x, z) under a +Y (NWU) rotation by ``theta``."""
    c, s = torch.cos(torch.tensor(theta, dtype=_F64)), torch.sin(torch.tensor(theta, dtype=_F64))
    return torch.tensor([[c, s], [-s, c]], dtype=_F64)


def _added_lin(theta: float, xu: float, zw: float) -> torch.Tensor:
    """World-frame 2x2 linear added-mass A = R diag(Xu, Zw) R^T at orientation ``theta``."""
    R = _r2(theta)
    return R @ torch.diag(torch.tensor([xu, zw], dtype=_F64)) @ R.T


def simulate_planar_added_mass(
    *,
    m0: float, m1: float,
    I0yy: float, I1yy: float,
    added0: tuple[float, float, float],   # base (Xu, Zw, Mq)
    added1: tuple[float, float, float],   # arm  (Xu, Zw, Mq)
    anchor_base: tuple[float, float, float],
    anchor_arm: tuple[float, float, float],
    q_traj: Callable[[float], tuple[float, float]],
    steps: int,
    dt: float,
    pos0: tuple[float, float] | None = None,
    theta0: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Reconstruct the free base motion from rigid+added-mass momentum conservation.

    ``q_traj(t) -> (q, qd)`` prescribes the joint angle and rate. Returns per-step arrays
    (pre-update snapshots): ``base_x``, ``base_z``, ``base_pitch``, ``omega0``, ``q``, and the
    recomputed total ``lin_mom`` [steps,2] / ``ang_mom`` [steps] (~0 by construction).
    """
    ab = torch.tensor(anchor_base, dtype=_F64)
    aa = torch.tensor(anchor_arm, dtype=_F64)
    xu0, zw0, mq0 = added0
    xu1, zw1, mq1 = added1
    x0, z0 = (0.0, 0.0) if pos0 is None else pos0
    theta = theta0

    eye2 = torch.eye(2, dtype=_F64)
    P0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=_F64)
    e2 = torch.tensor([0.0, 0.0, 1.0], dtype=_F64)

    rec: dict[str, list] = {k: [] for k in
                            ("base_x", "base_z", "base_pitch", "omega0", "q", "px", "pz", "ly")}
    for step in range(steps):
        t = step * dt
        q, qd = q_traj(t)

        # arm CoM offset (base frame) and joint-driven arm-CoM velocity (world) -- 3D like the
        # rigid reference, then take the (x, z) planar components.
        c, s = torch.cos(torch.tensor(q, dtype=_F64)), torch.sin(torch.tensor(q, dtype=_F64))
        Ry = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=_F64)
        dRy = torch.tensor([[-s, 0.0, c], [0.0, 0.0, 0.0], [-c, 0.0, -s]], dtype=_F64)
        d3 = ab - Ry @ aa
        ddq3 = -dRy @ aa * qd
        c0, s0 = torch.cos(torch.tensor(theta, dtype=_F64)), torch.sin(torch.tensor(theta, dtype=_F64))
        R0 = torch.tensor([[c0, 0.0, s0], [0.0, 1.0, 0.0], [-s0, 0.0, c0]], dtype=_F64)
        rel3 = R0 @ d3
        svec3 = R0 @ ddq3
        rx, rz = float(rel3[0]), float(rel3[2])
        sx, sz = float(svec3[0]), float(svec3[2])
        x1, z1 = x0 + rx, z0 + rz

        # per-body world linear generalized mass B_i = m_i I2 + A_i(theta_i); theta1 = theta + q
        B0 = m0 * eye2 + _added_lin(theta, xu0, zw0)
        B1 = m1 * eye2 + _added_lin(theta + q, xu1, zw1)

        # v0_lin = P0 u ; v1_lin = J1 u + c1 ; omega0 = e2 u ; omega1 = e2 u + qd
        J1 = torch.tensor([[1.0, 0.0, rz], [0.0, 1.0, -rx]], dtype=_F64)
        c1 = torch.tensor([sx, sz], dtype=_F64)

        Mlin = B0 @ P0 + B1 @ J1                                  # [2,3]
        blin = -(B1 @ c1)                                         # [2]

        w0row = torch.tensor([z0, -x0], dtype=_F64)               # picks (z*px - x*pz)
        w1row = torch.tensor([z1, -x1], dtype=_F64)
        Mang = ((I0yy + mq0 + I1yy + mq1) * e2
                + w0row @ B0 @ P0 + w1row @ B1 @ J1)              # [3]
        bang = -((I1yy + mq1) * qd + float(w1row @ B1 @ c1))      # scalar

        M = torch.cat([Mlin, Mang.reshape(1, 3)], dim=0)          # [3,3]
        b = torch.cat([blin, torch.tensor([bang], dtype=_F64)])   # [3]
        v0x, v0z, w0 = torch.linalg.solve(M, b).tolist()

        # total momentum from the solved state (should be ~0 -- bookkeeping check)
        v0 = torch.tensor([v0x, v0z], dtype=_F64)
        v1 = J1 @ torch.tensor([v0x, v0z, w0], dtype=_F64) + c1
        p = B0 @ v0 + B1 @ v1
        ly = ((I0yy + mq0) * w0 + (I1yy + mq1) * (w0 + qd)
              + float(w0row @ (B0 @ v0)) + float(w1row @ (B1 @ v1)))

        rec["base_x"].append(x0)
        rec["base_z"].append(z0)
        rec["base_pitch"].append(theta)
        rec["omega0"].append(w0)
        rec["q"].append(q)
        rec["px"].append(float(p[0]))
        rec["pz"].append(float(p[1]))
        rec["ly"].append(ly)

        x0 += v0x * dt
        z0 += v0z * dt
        theta += w0 * dt

    def t64(k: str) -> torch.Tensor:
        return torch.tensor(rec[k], dtype=_F64)

    return {
        "base_x": t64("base_x"), "base_z": t64("base_z"), "base_pitch": t64("base_pitch"),
        "omega0": t64("omega0"), "q": t64("q"),
        "lin_mom": torch.stack([t64("px"), t64("pz")], dim=-1), "ang_mom": t64("ly"),
    }
