"""Per-link Fossen force kernels. Pure torch; body-frame wrench [...,6] = [F(3), M(3)]."""

from __future__ import annotations

import torch
from torch import Tensor

from .constants import GRAVITY
from .frames import world_vec_to_body


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
