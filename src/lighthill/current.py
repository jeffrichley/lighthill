"""Ocean-current model: uniform global flow per env + relative-velocity computation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .constants import ANG, LIN
from .frames import world_vec_to_body


@dataclass
class CurrentField:
    """Uniform ocean-current model: one constant flow vector per environment.

    Each environment gets a single spatially-uniform current, drawn as a random
    direction scaled by a speed in ``[0, max_speed]`` (m/s). ``noise_std``
    optionally jitters that vector per step to emulate turbulence. Simple by
    design: the current enters the dynamics only through the flow-relative
    linear velocity of each body.
    """

    max_speed: float = 0.5
    noise_std: float = 0.0

    def sample(self, num_envs: int, generator: torch.Generator | None = None) -> Tensor:
        """Draw one current vector [num_envs, 3] per environment.

        Direction is uniform on the unit sphere (normalized Gaussian) and speed
        is uniform in ``[0, max_speed]``. ``generator`` seeds the draw for
        reproducible episodes.
        """
        speed = torch.rand(num_envs, 1, generator=generator) * self.max_speed
        direction = torch.randn(num_envs, 3, generator=generator)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        current: Tensor = direction * speed
        return current

    def perturb(self, current_world: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Add zero-mean Gaussian turbulence (std ``noise_std``) to a current.

        Returns ``current_world`` unchanged when ``noise_std`` is zero, so a
        deterministic run pays no RNG cost.
        """
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
