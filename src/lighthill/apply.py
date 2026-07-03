"""UnderwaterHydrodynamics: per-link Fossen wrench over an articulation. Isaac-free."""

from __future__ import annotations

import torch
from torch import Tensor

from .accel import AccelerationFilter
from .articulation import ArticulationView
from .coefficients import ResolvedCoefficients
from .current import CurrentField, relative_velocity
from .forces import added_mass_coriolis, added_mass_residual, buoyancy_wrench, drag_wrench
from .frames import quat_to_rotation_matrix
from .inertia import AddedMassRouting, effective_inertia, split_added_mass


class UnderwaterHydrodynamics:
    """Per-link Fossen hydrodynamic wrench applied over an articulation each step.

    Assembles the four force contributions of the UVMS model — buoyancy, linear
    plus quadratic drag, added-mass Coriolis, and the added-mass residual — for
    every body across every environment, and writes the result back through an
    :class:`ArticulationView`. The isotropic part of each link's added mass is
    folded into the rigid mass/inertia once at construction (via
    :func:`split_added_mass` / :func:`effective_inertia`) so only the
    anisotropic remainder is carried as an explicit residual wrench at runtime.
    Deliberately Isaac-free: it depends only on the ArticulationView Protocol,
    so the whole force law is testable on CPU without Isaac Lab installed.
    """

    def __init__(self, view: ArticulationView, coeffs: ResolvedCoefficients, *,
                 current: CurrentField | None = None, alpha: float = 0.08) -> None:
        self.view = view
        self.coeffs = coeffs
        self.current_field = current or CurrentField()
        self.routing = split_added_mass(coeffs.added_mass)  # [B,...]

        # Derive the device from the view's own state tensors.  The
        # ArticulationView Protocol guarantees body_states() returns real tensors
        # on the view's device, making this the canonical, Protocol-safe source.
        _dev = view.body_states()[0].device
        self._device = _dev

        self._filter = AccelerationFilter(shape=(view.num_envs, view.num_bodies), alpha=alpha)
        self._current_world = torch.zeros(view.num_envs, 3, device=_dev)

        # Pre-expand per-body coefficients to [E,B,...] so torch.cross and other
        # ops that require matching number of dims work without reshaping call sites.
        E, B = view.num_envs, view.num_bodies
        self._volume = coeffs.volume.unsqueeze(0).expand(E, B)                    # [E,B]
        self._cob = coeffs.center_of_buoyancy.unsqueeze(0).expand(E, B, 3)        # [E,B,3]
        self._lin_damp = coeffs.linear_damping.unsqueeze(0).expand(E, B, 6, 6)    # [E,B,6,6]
        self._quad_damp = coeffs.quadratic_damping.unsqueeze(0).expand(E, B, 6, 6) # [E,B,6,6]
        self._added_mass = coeffs.added_mass.unsqueeze(0).expand(E, B, 6, 6)      # [E,B,6,6]
        self._residual = self.routing.residual.unsqueeze(0).expand(E, B, 6, 6)    # [E,B,6,6]

        # augment inertias once (broadcast per-body routing across envs)
        mass0 = view.mass
        inertia0 = view.inertia_diag
        m_eff, i_eff = effective_inertia(
            mass0, inertia0,
            _broadcast_routing(self.routing, E),
        )
        view.set_body_inertias(m_eff, i_eff)

    def reset(self, current_world: Tensor | None = None) -> None:
        """Re-seed the ambient current and clear the acceleration filter.

        Call at episode boundaries. When ``current_world`` ([E,3], world frame)
        is given it becomes the fixed per-environment flow; otherwise a fresh
        flow is drawn from the configured :class:`CurrentField`. The
        acceleration EMA is fully reset so the first post-reset step contributes
        no spurious residual wrench.
        """
        if current_world is not None:
            self._current_world = current_world.to(self._device)
        else:
            self._current_world = self.current_field.sample(self.view.num_envs).to(self._device)
        self._filter.reset()

    def compute_wrench(self, dt: float) -> Tensor:
        """Sum the four hydrodynamic wrench terms in the body frame for one step.

        Reads current body pose and twist, forms the flow-relative velocity, and
        returns ``buoyancy + drag + added-mass Coriolis + added-mass residual``
        as a per-body [E,B,6] wrench (force then moment). ``dt`` is the sim step,
        needed only to finite-difference twist into the acceleration that drives
        the residual term. The wrench is expressed in each body's own frame;
        :meth:`apply` rotates it to world before handing it to the view.
        """
        _pos, quat, twist = self.view.body_states()  # [E,B,*]
        cur = self._current_world.unsqueeze(1).expand(-1, self.view.num_bodies, -1)
        v_rel = relative_velocity(twist, quat, cur)
        buoy = buoyancy_wrench(quat, self._volume, self._cob, self.coeffs.density)
        drag = drag_wrench(v_rel, self._lin_damp, self._quad_damp)
        cor = added_mass_coriolis(self._added_mass, v_rel)
        a_filt = self._filter.update(twist, dt)
        resid = added_mass_residual(self._residual, a_filt)
        return buoy + drag + cor + resid

    def apply(self, dt: float) -> None:
        """Compute this step's wrench and push it to the sim in world frame.

        Rotates the body-frame force and moment from :meth:`compute_wrench` into
        the world frame with each body's orientation and writes the [E,B,6]
        result via ``view.set_external_wrench``. This is the one call the sim
        integration loop needs per step.
        """
        w_body = self.compute_wrench(dt)
        quat = self.view.body_states()[1]
        R = quat_to_rotation_matrix(quat)  # [E,B,3,3]
        f_world = (R @ w_body[..., 0:3].unsqueeze(-1)).squeeze(-1)
        m_world = (R @ w_body[..., 3:6].unsqueeze(-1)).squeeze(-1)
        self.view.set_external_wrench(torch.cat([f_world, m_world], dim=-1))


def _broadcast_routing(routing: AddedMassRouting, num_envs: int) -> AddedMassRouting:
    return AddedMassRouting(
        mass_bump=routing.mass_bump.unsqueeze(0).expand(num_envs, -1),
        inertia_bump=routing.inertia_bump.unsqueeze(0).expand(num_envs, -1, -1),
        residual=routing.residual,
    )
