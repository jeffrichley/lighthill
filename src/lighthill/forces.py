"""Per-link Fossen force kernels. Pure torch; body-frame wrench [...,6] = [F(3), M(3)]."""

from __future__ import annotations

import torch
from torch import Tensor

from .constants import GRAVITY
from .frames import skew, world_vec_to_body


def buoyancy_wrench(quat_wb: Tensor, volume: Tensor, cob_body: Tensor,
                    neutrally_buoyant: Tensor, density: float,
                    gravity: float = GRAVITY) -> Tensor:
    """Buoyancy at the center of buoyancy, expressed as a body-frame wrench."""
    mag = density * gravity * volume  # [...]
    f_world = torch.zeros(*volume.shape, 3, dtype=volume.dtype, device=volume.device)
    f_world[..., 2] = mag  # +Z world (NWU up)
    f_body = world_vec_to_body(f_world, quat_wb)  # [...,3]
    f_body = torch.where(neutrally_buoyant.unsqueeze(-1), torch.zeros_like(f_body), f_body)
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
