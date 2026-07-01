"""Independent verification: lighthill's added-mass ROUTING vs the closed-form single-body
response. No simulator -- pure CPU integration of the real routing code, so no engine/harness
artifacts. Truth is exact for one body in pure translation (Fossen):

    (m + M_A[axis]) * vdot = F(t)  ->  v(t) = (A/((m+M_A[axis]) w)) (1 - cos w t).

This is the guard that finally caught what internal, self-referential checks could not: it
integrates the SAME body with lighthill's split_added_mass + effective_inertia +
added_mass_residual + AccelerationFilter (folded mass implicit, residual explicit, exactly as
the engine applies it) and requires it to track the closed form -- on the folded axis AND the
anisotropic explicit-residual axis.
"""

from __future__ import annotations

import math

import torch

from lighthill.accel import AccelerationFilter
from lighthill.forces import added_mass_coriolis, added_mass_residual
from lighthill.inertia import effective_inertia, split_added_mass

_DT = 0.00125
_STEPS = 4800          # 6 s
_AMP = 40.0
_OMEGA = 2.0
_M_RIGID = 13.7
# anisotropic added-mass diagonal (bluerov base): surge/sway/heave, roll/pitch/yaw
_AM_DIAG = [6.36, 7.12, 18.68, 0.189, 0.135, 0.222]


def _routing_rel_err(axis: int) -> float:
    added = torch.diag(torch.tensor(_AM_DIAG, dtype=torch.float64)).reshape(1, 6, 6)
    routing = split_added_mass(added)
    m_eff, _ = effective_inertia(
        torch.tensor([_M_RIGID], dtype=torch.float64),
        torch.ones(1, 3, dtype=torch.float64),
        routing,
    )
    solver_mass = float(m_eff[0])
    resid_mat = routing.residual
    filt = AccelerationFilter(shape=(1,), alpha=0.08)

    m_true = _M_RIGID + _AM_DIAG[axis]
    nu = torch.zeros(1, 1, 6, dtype=torch.float64)
    v_sim, v_true = [], []
    for k in range(_STEPS):
        t = k * _DT
        f = _AMP * math.sin(_OMEGA * t)
        a_filt = filt.update(nu.reshape(1, 6), _DT).reshape(1, 1, 6)
        resid = added_mass_residual(resid_mat.reshape(1, 1, 6, 6), a_filt).reshape(6)
        cor = added_mass_coriolis(added.reshape(1, 1, 6, 6), nu).reshape(6)
        total = torch.zeros(6, dtype=torch.float64)
        total[axis] = f
        total = total + resid + cor
        nu = nu + _DT * (total / solver_mass).reshape(1, 1, 6)   # semi-implicit Euler
        v_sim.append(float(nu[0, 0, axis]))
        v_true.append((_AMP / (m_true * _OMEGA)) * (1.0 - math.cos(_OMEGA * (t + _DT))))

    vs, vt = torch.tensor(v_sim), torch.tensor(v_true)
    transient = int(0.2 / _DT)
    return float((vs[transient:] - vt[transient:]).abs().max() / vt.abs().max().clamp_min(1e-9))


def test_routing_surge_folded_axis_is_exact():
    # min-axis added mass is folded into the solver mass -> no explicit residual -> exact.
    assert _routing_rel_err(0) < 5e-3


def test_routing_heave_explicit_residual_tracks_closed_form():
    # the large heave remainder rides the explicit EMA-filtered residual; it must still track
    # the closed form. (This is the axis a broken sim harness once made look catastrophic.)
    assert _routing_rel_err(2) < 2e-2
