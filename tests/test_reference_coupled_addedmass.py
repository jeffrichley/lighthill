"""Tests for the independent coupled added-mass momentum reference."""

from __future__ import annotations

import math

import torch

from lighthill.validation.reference_coupled_addedmass import simulate_planar_added_mass

# gate scenario (arm_swing_reaction): free base + one revolute arm, gravity/buoyancy/drag off.
_M0, _M1 = 13.7, 0.6
_I0, _I1 = 0.388167, 0.012820
_ANCHOR_BASE = (0.0, 0.0, -0.045)
_ANCHOR_ARM = (0.0, 0.0, 0.125)
_AMP, _OMEGA = 0.4, 2.0
_DT, _STEPS = 0.005, 800
# anisotropic added mass (bluerov): base (Xu, Zw, Mq), arm (Xu, Zw, Mq)
_ADDED0 = (6.36, 18.68, 0.135)
_ADDED1 = (0.1288, 0.0, 0.0)


def _q_traj(t: float) -> tuple[float, float]:
    return _AMP * (1.0 - math.cos(_OMEGA * t)), _AMP * _OMEGA * math.sin(_OMEGA * t)


def _run(added0, added1):
    return simulate_planar_added_mass(
        m0=_M0, m1=_M1, I0yy=_I0, I1yy=_I1, added0=added0, added1=added1,
        anchor_base=_ANCHOR_BASE, anchor_arm=_ANCHOR_ARM, q_traj=_q_traj, steps=_STEPS, dt=_DT)


def test_momentum_is_conserved_at_zero():
    # total (rigid+added) momentum must stay ~0 by construction, added mass on.
    out = _run(_ADDED0, _ADDED1)
    assert float(out["lin_mom"].abs().max()) < 1e-9
    assert float(out["ang_mom"].abs().max()) < 1e-9


def test_rigid_limit_matches_known_analytics():
    # added mass -> 0 must reproduce the rigid arm-swing base-pitch peak (~2.709 deg), the value
    # the gate's rigid coupling and the two rigid analytics agree on.
    out = _run((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    peak_deg = math.degrees(float(out["base_pitch"].abs().max()))
    assert abs(peak_deg - 2.709) < 0.05, f"rigid-limit pitch {peak_deg:.4f} deg != 2.709"


def test_added_mass_changes_the_reaction():
    # a sanity floor: the anisotropic added mass must measurably change the base reaction vs rigid
    # (it lowers the peak), and stay finite/conserving.
    rigid = math.degrees(float(_run((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))["base_pitch"].abs().max()))
    withA = math.degrees(float(_run(_ADDED0, _ADDED1)["base_pitch"].abs().max()))
    assert math.isfinite(withA) and withA > 0.0
    assert abs(withA - rigid) > 0.05  # added mass shifts the reaction by more than 0.05 deg


def _pitch_of(quat) -> float:
    w, x, y, z = (float(v) for v in quat)
    return math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))


def _fold_ema_peak_deg(am0_6, am1_6) -> float:
    """Base-pitch peak from lighthill's fold+EMA reference_coupled, same gate scenario."""
    from lighthill.validation.reference import Body
    from lighthill.validation.reference_coupled import TwoBodyChain, simulate_coupled

    i0 = (0.285417, _I0, 0.468083)
    i1 = (0.012820, _I1, 0.000640)

    def body(mass, inertia, amd):
        return Body(mass=mass, inertia=inertia, volume=0.0, cob=(0.0, 0.0, 0.0),
                    added_mass=torch.diag(torch.tensor(amd, dtype=torch.float32)),
                    linear_damping=torch.zeros(6, 6), quadratic_damping=torch.zeros(6, 6),
                    density=1000.0)

    chain = TwoBodyChain(body(_M0, i0, am0_6), body(_M1, i1, am1_6),
                         (0.0, 1.0, 0.0), _ANCHOR_BASE, _ANCHOR_ARM)
    out = simulate_coupled(
        chain, steps=_STEPS, dt=_DT,
        q_traj=lambda t: (_AMP * (1 - math.cos(_OMEGA * t)), _AMP * _OMEGA * math.sin(_OMEGA * t),
                          _AMP * _OMEGA * _OMEGA * math.cos(_OMEGA * t)),
        use_gravity=False, use_buoyancy=False)
    return math.degrees(max(abs(_pitch_of(q)) for q in out["base_quat"]))


def test_agrees_with_fold_ema_reference_on_added_mass():
    """Cross-validation: the fold+EMA reference_coupled (which the in-sim gate mirrors) must match
    this INDEPENDENT momentum reference on the anisotropic added-mass coupling. Agreement here is
    what certifies the coupled added-mass response is correct -- previously everything shared the
    fold+EMA kernels, so no internal check could referee it."""
    am0_6 = [_ADDED0[0], 7.12, _ADDED0[1], 0.189, _ADDED0[2], 0.222]  # 6x6 diag: x,y,z,roll,pitch,yaw
    am1_6 = [_ADDED1[0], _ADDED1[0], _ADDED1[1], 0.0, _ADDED1[2], 0.0]
    fold = _fold_ema_peak_deg(am0_6, am1_6)
    mom = math.degrees(float(_run(_ADDED0, _ADDED1)["base_pitch"].abs().max()))
    assert abs(fold - mom) / max(mom, 1e-9) < 0.03, f"fold+EMA {fold:.4f} vs momentum {mom:.4f} deg"
