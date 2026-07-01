"""Newton (Isaac Lab 3.0) adapter: implements the `ArticulationView` Protocol over an
``isaaclab_newton`` Articulation / RigidObject. Newton-only (needs Isaac Lab 3.0 + Newton).

This is the sibling of ``apply_isaac.py`` (PhysX). The hydro core is unchanged; only this
adapter differs, and only in three spots:

* ``.data.*`` reads are warp-backed ``ProxyArray`` objects, not torch tensors, so each is
  converted with ``wp.to_torch`` to the ``[E, B, *]`` torch shape the kernels expect.
* the inertia write goes through the articulation's own ``set_masses`` / ``set_inertias``
  (Newton has no ``root_physx_view``).
* the wrench write -- ``set_external_force_and_torque(is_global=True)`` + ``write_data_to_sim``
  -- is **identical** to PhysX; Isaac Lab 3.0's multi-backend Articulation maps it onto
  Newton's ``xfrc`` internally.

Only ``set_body_inertias`` folds isotropic/angular added mass into the rigid mass/inertia the
way the PhysX path does. On MuJoCo/Newton this could instead be pushed entirely through the
residual xfrc wrench; that is a deliberate follow-up choice, not required for correctness.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .frames import world_vec_to_body


def _to_torch(a: object) -> Tensor:
    """Isaac Lab Newton ``.data.*`` reads are warp-backed ProxyArrays; convert to torch."""
    if isinstance(a, torch.Tensor):
        return a
    import warp as wp

    return wp.to_torch(a)


class NewtonArticulationView:
    """Wrap an ``isaaclab_newton`` Articulation/RigidObject as a lighthill ``ArticulationView``."""

    def __init__(self, asset: object) -> None:
        self._asset = asset
        d = asset.data  # type: ignore[attr-defined]
        self.num_envs = int(asset.num_instances)  # type: ignore[attr-defined]
        self.num_bodies = int(asset.num_bodies)  # type: ignore[attr-defined]
        # Protocol read attributes: per-body rigid mass [E,B] and principal inertia [E,B,3].
        self.mass = _to_torch(d.default_mass).reshape(self.num_envs, self.num_bodies).clone()
        inertia_flat = _to_torch(d.default_inertia).reshape(self.num_envs, self.num_bodies, 9)
        self.inertia_diag = inertia_flat[..., [0, 4, 8]].clone()

    def body_states(self) -> tuple[Tensor, Tensor, Tensor]:
        """(pos [E,B,3] world, quat [E,B,4] wxyz, vel [E,B,6] BODY-frame twist).

        Newton reports velocities in the world frame like PhysX; rotate into the body frame.
        """
        d = self._asset.data  # type: ignore[attr-defined]
        pos = _to_torch(d.body_pos_w)
        quat = _to_torch(d.body_quat_w)
        lin_w = _to_torch(d.body_lin_vel_w)
        ang_w = _to_torch(d.body_ang_vel_w)
        lin_b = world_vec_to_body(lin_w, quat)
        ang_b = world_vec_to_body(ang_w, quat)
        return pos, quat, torch.cat([lin_b, ang_b], dim=-1)

    def set_external_wrench(self, wrench_world: Tensor) -> None:
        """Apply a per-body WORLD-frame wrench [E,B,6]=[F(3),M(3)] (mapped to Newton xfrc)."""
        forces = wrench_world[..., 0:3].contiguous()
        torques = wrench_world[..., 3:6].contiguous()
        self._asset.set_external_force_and_torque(forces, torques, is_global=True)  # type: ignore[attr-defined]
        self._asset.write_data_to_sim()  # type: ignore[attr-defined]

    def set_body_inertias(self, mass: Tensor, inertia_diag: Tensor) -> None:
        """Set per-body scalar mass [E,B] and principal inertia [E,B,3] via Newton's own setters."""
        self._asset.set_masses(  # type: ignore[attr-defined]
            masses=mass.reshape(self.num_envs, self.num_bodies).contiguous())
        flat = torch.zeros(self.num_envs, self.num_bodies, 9,
                           device=inertia_diag.device, dtype=inertia_diag.dtype)
        flat[..., 0] = inertia_diag[..., 0]
        flat[..., 4] = inertia_diag[..., 1]
        flat[..., 8] = inertia_diag[..., 2]
        self._asset.set_inertias(inertias=flat.contiguous())  # type: ignore[attr-defined]
