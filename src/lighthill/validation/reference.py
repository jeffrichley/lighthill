"""Single-rigid-body Fossen integrator that assembles the kernels into full dynamics.

This is the on-CPU reference used to validate the force law before the Isaac Lab
glue exists. It is intentionally minimal: one body, semi-implicit Euler, body-frame
twist. It is NOT the production sim (that is Isaac Lab, Plan B)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..constants import GRAVITY
from ..forces import added_mass_coriolis, buoyancy_wrench, drag_wrench
from ..frames import quat_to_rotation_matrix


@dataclass
class Body:
    mass: float
    inertia: tuple[float, float, float]
    volume: float
    cob: tuple[float, float, float]
    added_mass: Tensor
    linear_damping: Tensor
    quadratic_damping: Tensor
    density: float


def terminal_velocity_quadratic(force: float, d_quad: float) -> float:
    return math.sqrt(force / d_quad)


def _rigid_body_mass_matrix(body: Body) -> Tensor:
    m = body.mass
    ix, iy, iz = body.inertia
    return torch.diag(torch.tensor([m, m, m, ix, iy, iz], dtype=torch.float32))


def _quat_mul(q: Tensor, r: Tensor) -> Tensor:
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return torch.tensor([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _integrate_quat(quat: Tensor, omega_body: Tensor, dt: float) -> Tensor:
    omega_quat = torch.cat([torch.zeros(1), omega_body])
    dq = 0.5 * _quat_mul(quat, omega_quat)
    q = quat + dq * dt
    return q / q.norm()


def simulate(
    body: Body,
    *,
    steps: int,
    dt: float,
    external_force_body: Tensor | None = None,
    quat0: Tensor | None = None,
    omega0: Tensor | None = None,
    vel0: Tensor | None = None,
    gravity: float = GRAVITY,
) -> dict[str, Tensor]:
    mass_matrix = _rigid_body_mass_matrix(body) + body.added_mass
    minv = torch.linalg.inv(mass_matrix)
    cob = torch.tensor(body.cob, dtype=torch.float32)
    vol = torch.tensor(body.volume, dtype=torch.float32)
    not_neutral = torch.tensor(False)
    f_ext = external_force_body if external_force_body is not None else torch.zeros(6)

    quat = quat0.clone() if quat0 is not None else torch.tensor([1.0, 0.0, 0.0, 0.0])
    twist = torch.zeros(6)
    if vel0 is not None:
        twist[0:3] = vel0
    if omega0 is not None:
        twist[3:6] = omega0
    pos = torch.zeros(3)

    pos_hist, quat_hist, twist_hist = [], [], []
    for _ in range(steps):
        buoy = buoyancy_wrench(quat, vol, cob, not_neutral, body.density, gravity)
        # gravity (weight) acts at CoM (body origin), world -Z, no moment
        R = quat_to_rotation_matrix(quat)
        weight_world = torch.tensor([0.0, 0.0, -body.mass * gravity])
        weight_body = (R.transpose(-1, -2) @ weight_world.unsqueeze(-1)).squeeze(-1)
        grav = torch.cat([weight_body, torch.zeros(3)])
        drag = drag_wrench(twist, body.linear_damping, body.quadratic_damping)
        cor = added_mass_coriolis(body.added_mass, twist)
        total = buoy + grav + drag + cor + f_ext
        accel = (minv @ total.unsqueeze(-1)).squeeze(-1)
        twist = twist + accel * dt  # semi-implicit
        # advance pose: linear in world frame, angular via quaternion kinematics
        vel_world = (R @ twist[0:3].unsqueeze(-1)).squeeze(-1)
        pos = pos + vel_world * dt
        quat = _integrate_quat(quat, twist[3:6], dt)
        pos_hist.append(pos.clone())
        quat_hist.append(quat.clone())
        twist_hist.append(twist.clone())

    return {
        "pos": torch.stack(pos_hist),
        "quat": torch.stack(quat_hist),
        "twist": torch.stack(twist_hist),
    }
