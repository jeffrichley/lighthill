"""Route a 6x6 added-mass matrix to mass bump / inertia bump / residual wrench."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class AddedMassRouting:
    """The three destinations a 6x6 added-mass matrix is split into.

    Added mass has an isotropic translational part that a rigid-body integrator
    can absorb directly (``mass_bump``), a rotational part that maps onto the
    principal inertia (``inertia_bump``), and an anisotropic / off-diagonal
    remainder that has no rigid-body home and must be applied as an explicit
    per-step wrench (``residual``). This routing lets the bulk of added mass ride
    the sim's own integrator while only the leftover is computed each step.
    """

    mass_bump: Tensor      # [N] isotropic scalar mass addition
    inertia_bump: Tensor   # [N,3] principal inertia addition
    residual: Tensor       # [N,6,6] anisotropic linear remainder + off-diagonal


def split_added_mass(added_mass: Tensor) -> AddedMassRouting:
    """Route each link's 6x6 added mass into mass bump, inertia bump, and residual.

    Takes the largest isotropic translational mass that is safe to fold into the
    rigid body (the per-link min of the three linear-diagonal entries) as
    ``mass_bump``, moves the angular diagonal wholesale into ``inertia_bump``,
    and leaves everything else — the anisotropic linear remainder and all
    off-diagonal coupling — in ``residual``, which is applied as an explicit
    wrench so no added mass is double-counted or dropped.
    """
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
    """Fold the isotropic added-mass bumps into the rigid mass and inertia.

    Returns ``(rigid_mass + mass_bump, rigid_inertia + inertia_bump)`` so the
    absorbed part of added mass is baked into the body properties once at setup;
    only ``routing.residual`` remains as a runtime wrench.
    """
    return rigid_mass + routing.mass_bump, rigid_inertia + routing.inertia_bump
