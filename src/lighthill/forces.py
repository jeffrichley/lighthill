"""Per-link Fossen force kernels. Pure torch; body-frame wrench [...,6] = [F(3), M(3)]."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .constants import GRAVITY
from .frames import skew, world_vec_to_body


def buoyancy_wrench(quat_wb: Tensor, volume: Tensor, cob_body: Tensor,
                    density: float, gravity: float = GRAVITY) -> Tensor:
    """Buoyancy F = rho*g*V at the center of buoyancy, as a body-frame wrench.

    Always applied: force AND moment = cob_body x F_body. Neutrality is expressed
    by tuning `volume` (V = m/rho) so buoyancy cancels weight in net while the
    CoB-CoM offset still produces the restoring couple -- never by skipping the
    force. See DECISIONS.md (D1)."""
    mag = density * gravity * volume  # [...]
    f_world = torch.zeros(*volume.shape, 3, dtype=volume.dtype, device=volume.device)
    f_world[..., 2] = mag  # +Z world (NWU up)
    f_body = world_vec_to_body(f_world, quat_wb)  # [...,3]
    moment = torch.cross(cob_body, f_body, dim=-1)  # r x F
    return torch.cat([f_body, moment], dim=-1)


def drag_wrench(v_rel_body: Tensor, linear_damping: Tensor,
                quadratic_damping: Tensor) -> Tensor:
    """-(D_lin @ v + D_quad @ (|v| * v)), body frame."""
    v = v_rel_body.unsqueeze(-1)  # [...,6,1]
    quad_term = (v_rel_body.abs() * v_rel_body).unsqueeze(-1)
    drag = linear_damping @ v + quadratic_damping @ quad_term
    return -drag.squeeze(-1)


def added_mass_coriolis(added_mass: Tensor, v_rel_body: Tensor) -> Tensor:
    """-C_A(nu) @ nu, with C_A from Fossen's skew construction. Body frame."""
    a11 = added_mass[..., 0:3, 0:3]
    a12 = added_mass[..., 0:3, 3:6]
    a21 = added_mass[..., 3:6, 0:3]
    a22 = added_mass[..., 3:6, 3:6]
    nu1 = v_rel_body[..., 0:3]
    nu2 = v_rel_body[..., 3:6]
    top = (a11 @ nu1.unsqueeze(-1) + a12 @ nu2.unsqueeze(-1)).squeeze(-1)
    bot = (a21 @ nu1.unsqueeze(-1) + a22 @ nu2.unsqueeze(-1)).squeeze(-1)
    s_top = skew(top)
    s_bot = skew(bot)
    zero = torch.zeros_like(s_top)
    upper = torch.cat([zero, -s_top], dim=-1)
    lower = torch.cat([-s_top, -s_bot], dim=-1)
    c_a = torch.cat([upper, lower], dim=-2)  # [...,6,6]
    return -(c_a @ v_rel_body.unsqueeze(-1)).squeeze(-1)


def added_mass_residual(added_mass_offdiag: Tensor, accel_body: Tensor) -> Tensor:
    """-(M_A_offdiag @ a). Off-diagonal added-mass reaction (Plan B feeds filtered accel)."""
    return -(added_mass_offdiag @ accel_body.unsqueeze(-1)).squeeze(-1)


def _ellipsoid_projected_area(u_hat: Tensor, semi_axes: Tensor) -> Tensor:
    """Area of the ellipsoid's shadow on the plane normal to unit vector u_hat [...,1].

    MuJoCo's ellipsoid lemma: A = pi*sqrt(sum r_j^4 r_k^4 u_i^2 / sum r_j^2 r_k^2 u_i^2)."""
    rx2, ry2, rz2 = (semi_axes[..., i] ** 2 for i in range(3))
    ux2, uy2, uz2 = (u_hat[..., i] ** 2 for i in range(3))
    num = ry2 * rz2 * (ry2 * rz2) * ux2 + rz2 * rx2 * (rz2 * rx2) * uy2 + rx2 * ry2 * (rx2 * ry2) * uz2
    den = ry2 * rz2 * ux2 + rz2 * rx2 * uy2 + rx2 * ry2 * uz2
    return math.pi * torch.sqrt(num / den.clamp_min(1e-30))


def lift_wrench(v_rel_body: Tensor, semi_axes: Tensor, c_kutta: Tensor, c_magnus: Tensor,
                density: float) -> Tensor:
    """Ellipsoid Kutta + Magnus lift as a body-frame wrench [...,6] (force only, zero moment).

    Transcribed from MuJoCo's ellipsoid fluid model:
      f_K = C_K rho A_proj (v_hat . n_hat) (n_hat x v) x v      (circulatory lift at angle of attack)
      f_M = C_M rho V (omega x v)                                (spin/Magnus lift)
    with surface normal n_s = ((ry rz/rx) vx, (rz rx/ry) vy, (rx ry/rz) vz), volume
    V = 4/3 pi rx ry rz, and A_proj the projected area normal to v. Kutta vanishes for a
    sphere (n_hat == v_hat) and at zero angle of attack; both vanish at zero speed. Links
    with zero coefficients contribute nothing, so lift is opt-in per link."""
    v = v_rel_body[..., 0:3]      # linear velocity relative to the fluid, body frame
    omega = v_rel_body[..., 3:6]  # angular velocity, body frame
    rx, ry, rz = semi_axes[..., 0], semi_axes[..., 1], semi_axes[..., 2]

    # Magnus: safe at any velocity (no normalisation).
    volume = (4.0 / 3.0) * math.pi * rx * ry * rz
    f_magnus = (c_magnus * density * volume).unsqueeze(-1) * torch.cross(omega, v, dim=-1)

    # Kutta: needs v_hat and the (normalised) surface normal, so guard the zero-speed axis.
    speed = v.norm(dim=-1, keepdim=True)
    safe_speed = speed.clamp_min(1e-12)
    v_hat = v / safe_speed
    n_s = torch.stack([ry * rz / rx * v[..., 0],
                       rz * rx / ry * v[..., 1],
                       rx * ry / rz * v[..., 2]], dim=-1)
    n_hat = n_s / n_s.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dot = (v_hat * n_hat).sum(dim=-1, keepdim=True)
    a_proj = _ellipsoid_projected_area(v_hat, semi_axes).unsqueeze(-1)
    lift_dir = torch.cross(torch.cross(n_hat, v, dim=-1), v, dim=-1)
    f_kutta = c_kutta.unsqueeze(-1) * density * a_proj * dot * lift_dir
    f_kutta = torch.where(speed > 1e-9, f_kutta, torch.zeros_like(f_kutta))

    force = f_magnus + f_kutta
    return torch.cat([force, torch.zeros_like(force)], dim=-1)
