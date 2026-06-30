"""Rotation/frame utilities (NWU, scalar-first quaternions). Pure torch."""

import torch
from torch import Tensor


def quat_to_rotation_matrix(quat: Tensor) -> Tensor:
    """(w,x,y,z) body->world quaternion ``[...,4]`` -> rotation matrix ``[...,3,3]``."""
    quat = quat / quat.norm(dim=-1, keepdim=True)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return R.reshape(*quat.shape[:-1], 3, 3)


def world_vec_to_body(vec_world: Tensor, quat_wb: Tensor) -> Tensor:
    """Express a world-frame vector in the body frame: ``R^T @ v``."""
    R = quat_to_rotation_matrix(quat_wb)
    return (R.transpose(-1, -2) @ vec_world.unsqueeze(-1)).squeeze(-1)


def skew(v: Tensor) -> Tensor:
    """Skew-symmetric matrix ``[...,3,3]`` such that ``skew(a) @ b == a x b``."""
    zero = torch.zeros_like(v[..., 0])
    row0 = torch.stack([zero, -v[..., 2], v[..., 1]], dim=-1)
    row1 = torch.stack([v[..., 2], zero, -v[..., 0]], dim=-1)
    row2 = torch.stack([-v[..., 1], v[..., 0], zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)
