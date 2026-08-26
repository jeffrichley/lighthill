"""Route a 6x6 added-mass matrix to mass bump / inertia bump / residual wrench."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class AddedMassRouting:
    mass_bump: Tensor      # [N] isotropic scalar mass addition
    inertia_bump: Tensor   # [N,3] principal inertia addition
    residual: Tensor       # [N,6,6] anisotropic linear remainder + off-diagonal


def split_added_mass(added_mass: Tensor) -> AddedMassRouting:
    diag = torch.diagonal(added_mass, dim1=-2, dim2=-1)  # [N,6]
    lin_diag = diag[:, 0:3]
    ang_diag = diag[:, 3:6]
    mass_bump = lin_diag.min(dim=-1).values  # isotropic safe part
    inertia_bump = ang_diag.clone()
    residual = added_mass.clone()
    idx = torch.arange(6, device=added_mass.device)
    # zero the angular diagonal (moved to inertia) and subtract the isotropic mass on linear diagonal
    residual[:, idx, idx] = 0.0
    # restore the anisotropic linear remainder on the linear diagonal
    remainder = lin_diag - mass_bump.unsqueeze(-1)
    for k in range(3):
        residual[:, k, k] = remainder[:, k]
    return AddedMassRouting(mass_bump=mass_bump, inertia_bump=inertia_bump, residual=residual)


def effective_inertia(rigid_mass: Tensor, rigid_inertia: Tensor,
                      routing: AddedMassRouting) -> tuple[Tensor, Tensor]:
    return rigid_mass + routing.mass_bump, rigid_inertia + routing.inertia_bump
