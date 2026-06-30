"""Isaac Lab adapter: the ONLY core module that imports isaaclab.

`IsaacArticulationView` implements the `ArticulationView` Protocol over an Isaac
`RigidObject` or `Articulation`, using the calls pinned in
``docs/isaac-api-findings.md``. It does no physics — only state read, wrench
write, inertia write, and the world->body velocity-frame conversion the kernels
need. Anything importing this module requires a live Isaac Sim.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .frames import world_vec_to_body


class IsaacArticulationView:
    """Wrap an Isaac `RigidObject`/`Articulation` as a lighthill `ArticulationView`.

    Both Isaac asset types expose the same surface the adapter needs:
    ``.data.body_{pos,quat,lin_vel,ang_vel}_w``, ``.set_external_force_and_torque``,
    ``.write_data_to_sim``, and ``.root_physx_view`` for mass/inertia writes.
    """

    def __init__(self, asset: object) -> None:
        self._asset = asset
        data = asset.data  # type: ignore[attr-defined]
        # body_pos_w is (num_envs, num_bodies, 3)
        self.num_envs = int(data.body_pos_w.shape[0])
        self.num_bodies = int(data.body_pos_w.shape[1])
        # Protocol read attributes: per-body rigid mass [E,B] and principal inertia [E,B,3].
        # Normalize shapes: a single-body RigidObject reports default_mass [E,1] and
        # default_inertia [E,9] (no body dim), while an Articulation reports [E,B] and
        # [E,B,9]. Reshape both to the [E,B,...] the Protocol promises.
        self.mass = data.default_mass.reshape(self.num_envs, self.num_bodies).clone()
        inertia_flat = data.default_inertia.reshape(self.num_envs, self.num_bodies, 9)
        # default_inertia is a flattened 3x3 (9); the principal diagonal is indices 0,4,8.
        self.inertia_diag = inertia_flat[..., [0, 4, 8]].clone()

    def body_states(self) -> tuple[Tensor, Tensor, Tensor]:
        """(pos [E,B,3] world, quat [E,B,4] wxyz, vel [E,B,6] BODY-frame twist).

        Isaac reports velocities in the world frame; the kernels want body-frame
        twist, so rotate both linear and angular parts into the body frame.
        """
        data = self._asset.data  # type: ignore[attr-defined]
        pos = data.body_pos_w
        quat = data.body_quat_w
        lin_w = data.body_lin_vel_w  # [E,B,3] world
        ang_w = data.body_ang_vel_w  # [E,B,3] world
        lin_b = world_vec_to_body(lin_w, quat)
        ang_b = world_vec_to_body(ang_w, quat)
        vel = torch.cat([lin_b, ang_b], dim=-1)  # [E,B,6]
        return pos, quat, vel

    def set_external_wrench(self, wrench_world: Tensor) -> None:
        """Apply a per-body WORLD-frame wrench [E,B,6]=[F(3),M(3)].

        `apply()` already converted body->world, so we hand Isaac global-frame
        vectors (is_global=True). Must write_data_to_sim before the next step.
        """
        forces = wrench_world[..., 0:3].contiguous()
        torques = wrench_world[..., 3:6].contiguous()
        self._asset.set_external_force_and_torque(forces, torques, is_global=True)  # type: ignore[attr-defined]
        self._asset.write_data_to_sim()  # type: ignore[attr-defined]

    def set_body_inertias(self, mass: Tensor, inertia_diag: Tensor) -> None:
        """Set per-body scalar mass [E,B] and principal inertia [E,B,3] (init-time).

        Writes through the PhysX tensor view. Inertia is set as a flattened
        diagonal 3x3 (off-diagonal zero), matching the view's get_inertias shape.
        """
        view = self._asset.root_physx_view  # type: ignore[attr-defined]
        masses = view.get_masses()
        view.set_masses(mass.reshape(masses.shape).to(masses.device), torch.arange(self.num_envs))
        inertias = view.get_inertias()  # [..., 9] flattened 3x3
        flat = torch.zeros_like(inertias).reshape(self.num_envs, self.num_bodies, 9)
        flat[..., 0] = inertia_diag[..., 0]
        flat[..., 4] = inertia_diag[..., 1]
        flat[..., 8] = inertia_diag[..., 2]
        view.set_inertias(flat.reshape(inertias.shape).to(inertias.device), torch.arange(self.num_envs))
