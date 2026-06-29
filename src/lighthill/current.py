"""Ocean-current model: uniform global flow per env + relative-velocity computation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .constants import ANG, LIN
from .frames import world_vec_to_body


@dataclass
class CurrentField:
    max_speed: float = 0.5
    noise_std: float = 0.0

    def sample(self, num_envs: int, generator: torch.Generator | None = None) -> Tensor:
        speed = torch.rand(num_envs, 1, generator=generator) * self.max_speed
        direction = torch.randn(num_envs, 3, generator=generator)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return direction * speed

    def perturb(self, current_world: Tensor, generator: torch.Generator | None = None) -> Tensor:
        if self.noise_std == 0.0:
            return current_world
        noise = torch.randn(current_world.shape, generator=generator) * self.noise_std
        return current_world + noise


def relative_velocity(v_body: Tensor, quat_wb: Tensor, current_world: Tensor) -> Tensor:
    """Body twist relative to the flow. Current enters the linear part only."""
    cur_body = world_vec_to_body(current_world, quat_wb)
    out = v_body.clone()
    out[..., LIN] = v_body[..., LIN] - cur_body
    out[..., ANG] = v_body[..., ANG]
    return out
