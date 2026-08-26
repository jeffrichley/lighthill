"""Independent planar momentum-reconstruction reference (Task 6 verification referee).

The Featherstone reference (``reference_coupled``) and the in-sim PhysX gate disagree
on the free base's *pitch* reaction to an arm swing (a clean ~0.80 amplitude ratio at
fine dt). Momentum arguments alone cannot referee that -- both models approximately
conserve momentum -- so this module is a **third, fully independent** analytical model
to break the tie.

Method (planar: motion in x-z, rotation about the world Y axis; the arm-swing gate is
run gravity/buoyancy/drag-free, so the only external wrench is zero):

  With zero net external force and torque and starting from rest, total linear and
  angular momentum stay **exactly zero** for all time. That is three scalar
  constraints -- P_x = P_z = L_y = 0 -- linear in the base velocity (v0x, v0z, omega0)
  once the arm's velocity is written via the joint kinematics. Solving that 3x3 system
  at each instant *reconstructs* the base velocity directly (the classic
  "robot-in-space" reconstruction equation); integrating it forward advances the base
  pose. No composite spatial-inertia matrix and no acceleration-level solve are used --
  the derivation is independent of ``reference_coupled`` and shares only the joint
  kinematics convention (which is not the thing under test).

Pure NumPy, no torch, to keep the implementation maximally independent.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _ry(a: float) -> np.ndarray:
    """Rotation about +Y by ``a`` (NWU), [3,3]."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _dry(a: float) -> np.ndarray:
    """d/da of :func:`_ry`, [3,3]."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[-s, 0.0, c], [0.0, 0.0, 0.0], [-c, 0.0, -s]])


def simulate_planar_momentum(
    *,
    m0: float,
    m1: float,
    I0yy: float,
    I1yy: float,
    anchor_base: tuple[float, float, float],
    anchor_arm: tuple[float, float, float],
    q_traj: Callable[[float], tuple[float, float]],
    steps: int,
    dt: float,
    pos0: tuple[float, float] | None = None,
    theta0: float = 0.0,
) -> dict[str, np.ndarray]:
    """Reconstruct the free base motion from momentum conservation.

    ``q_traj(t) -> (q, qd)`` prescribes the joint angle and rate. Anchors follow the
    ``reference_coupled`` convention: the arm CoM offset in the base frame is
    ``d(q) = anchor_base - Ry(q) @ anchor_arm``. Returns per-step arrays (pre-update
    snapshots, matching the Featherstone reference's sampling): ``base_x``, ``base_z``,
    ``base_pitch`` (theta about Y), ``arm_x``, ``arm_z``, ``omega0``, ``q``, and the
    recomputed ``lin_mom`` [steps,2] / ``ang_mom`` [steps] (both ~0 by construction --
    a bookkeeping check).
    """
    ab = np.asarray(anchor_base, dtype=float)
    aa = np.asarray(anchor_arm, dtype=float)
    x0, z0 = (0.0, 0.0) if pos0 is None else pos0
    theta = theta0

    rec: dict[str, list[float]] = {
        "base_x": [], "base_z": [], "base_pitch": [], "arm_x": [], "arm_z": [],
        "omega0": [], "q": [], "px": [], "pz": [], "ly": [],
    }
    for step in range(steps):
        t = step * dt
        q, qd = q_traj(t)

        d = ab - _ry(q) @ aa                 # arm CoM offset in base frame
        d_dq = -_dry(q) @ aa                 # d/dq of d
        R0 = _ry(theta)
        rel = R0 @ d                         # base->arm CoM offset, world
        svec = R0 @ d_dq * qd                # joint-driven part of arm CoM velocity, world
        rx, rz = rel[0], rel[2]
        sx, sz = svec[0], svec[2]
        x1, z1 = x0 + rx, z0 + rz

        # arm CoM velocity (planar): v1 = v0 + omega0 x rel + svec
        #   v1x = v0x + omega0*rz + sx ;  v1z = v0z - omega0*rx + sz
        # Enforce P_x = P_z = L_y(origin) = 0 -> 3x3 linear system in (v0x, v0z, omega0).
        M = np.array([
            [m0 + m1, 0.0, m1 * rz],
            [0.0, m0 + m1, -m1 * rx],
            [m0 * z0 + m1 * z1, -(m0 * x0 + m1 * x1),
             I0yy + I1yy + m1 * (z1 * rz + x1 * rx)],
        ])
        b = np.array([
            -m1 * sx,
            -m1 * sz,
            -(I1yy * qd + m1 * (z1 * sx - x1 * sz)),
        ])
        v0x, v0z, w0 = np.linalg.solve(M, b)

        # recompute momenta from the solved state (should be ~0 -- bookkeeping check)
        v1x = v0x + w0 * rz + sx
        v1z = v0z - w0 * rx + sz
        px = m0 * v0x + m1 * v1x
        pz = m0 * v0z + m1 * v1z
        ly = (I0yy * w0 + I1yy * (w0 + qd)
              + m0 * (z0 * v0x - x0 * v0z) + m1 * (z1 * v1x - x1 * v1z))

        rec["base_x"].append(x0)
        rec["base_z"].append(z0)
        rec["base_pitch"].append(theta)
        rec["arm_x"].append(x1)
        rec["arm_z"].append(z1)
        rec["omega0"].append(w0)
        rec["q"].append(q)
        rec["px"].append(px)
        rec["pz"].append(pz)
        rec["ly"].append(ly)

        # integrate the base pose forward with the reconstructed velocity
        x0 += v0x * dt
        z0 += v0z * dt
        theta += w0 * dt

    return {
        "base_x": np.array(rec["base_x"]),
        "base_z": np.array(rec["base_z"]),
        "base_pitch": np.array(rec["base_pitch"]),
        "arm_x": np.array(rec["arm_x"]),
        "arm_z": np.array(rec["arm_z"]),
        "omega0": np.array(rec["omega0"]),
        "q": np.array(rec["q"]),
        "lin_mom": np.stack([rec["px"], rec["pz"]], axis=-1),
        "ang_mom": np.array(rec["ly"]),
    }
