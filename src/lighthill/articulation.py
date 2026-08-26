"""The articulation interface apply.py depends on, plus a CPU fake for tests.

The real Isaac Lab adapter (apply_isaac.py) implements this Protocol; the fake
lets all assembly logic be tested without Isaac installed."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor


class ArticulationView(Protocol):
    """Per-body state read + per-body wrench/inertia write. Shapes: (num_envs, num_bodies, ...)."""

    num_envs: int
    num_bodies: int
    mass: Tensor          # [E,B] per-body rigid mass (read attribute)
    inertia_diag: Tensor  # [E,B,3] per-body principal rigid inertia (read attribute)

    def body_states(self) -> tuple[Tensor, Tensor, Tensor]:
        """(pos [E,B,3] world, quat [E,B,4] wxyz body->world, vel [E,B,6] body twist)."""
        ...

    def set_external_wrench(self, wrench_world: Tensor) -> None:
        """Apply per-body external wrench [E,B,6] = [F(3), M(3)] (frame per adapter)."""
        ...

    def set_body_inertias(self, mass: Tensor, inertia_diag: Tensor) -> None:
        """Set per-body scalar mass [E,B] and principal inertia [E,B,3] (init-time)."""
        ...


class FakeArticulation:
    """In-memory CPU stand-in for tests. Records wrenches; lets tests set state."""

    def __init__(self, num_envs: int, num_bodies: int) -> None:
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self._pos = torch.zeros(num_envs, num_bodies, 3)
        self._quat = torch.zeros(num_envs, num_bodies, 4)
        self._quat[..., 0] = 1.0
        self._vel = torch.zeros(num_envs, num_bodies, 6)
        self.last_wrench = torch.zeros(num_envs, num_bodies, 6)
        self.mass = torch.ones(num_envs, num_bodies)
        self.inertia_diag = torch.ones(num_envs, num_bodies, 3)

    def body_states(self) -> tuple[Tensor, Tensor, Tensor]:
        return self._pos, self._quat, self._vel

    def set_external_wrench(self, wrench_world: Tensor) -> None:
        self.last_wrench = wrench_world.clone()

    def set_body_inertias(self, mass: Tensor, inertia_diag: Tensor) -> None:
        self.mass = mass.clone()
        self.inertia_diag = inertia_diag.clone()

    # test helpers
    def set_body_velocity(self, vel: Tensor) -> None:
        self._vel = vel.clone()

    def set_body_quat(self, quat: Tensor) -> None:
        self._quat = quat.clone()
