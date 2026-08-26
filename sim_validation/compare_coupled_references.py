"""Referee the coupled added-mass case: does lighthill's fold+EMA `reference_coupled` agree
with the INDEPENDENT momentum reference (reference_coupled_addedmass)? Same gate scenario, same
anisotropic added mass, same prescribed arm swing. If they agree, the fold+EMA routing is
validated for the coupled case (so the gate's ~21% sim-vs-reference gap is the sim's explicit
discretization, not a model error). Pure CPU torch, no simulator.

Run:  PYTHONPATH=src python sim_validation/compare_coupled_references.py   (via uv run)
"""

from __future__ import annotations

import math

import torch

from lighthill.validation.reference import Body
from lighthill.validation.reference_coupled import TwoBodyChain, simulate_coupled
from lighthill.validation.reference_coupled_addedmass import simulate_planar_added_mass

M0, M1 = 13.7, 0.6
I0 = (0.285417, 0.388167, 0.468083)
I1 = (0.012820, 0.012820, 0.000640)
ANCHOR_BASE = (0.0, 0.0, -0.045)
ANCHOR_ARM = (0.0, 0.0, 0.125)
AMP, OMEGA, DT, STEPS = 0.4, 2.0, 0.005, 800
AM0 = [6.36, 7.12, 18.68, 0.189, 0.135, 0.222]   # base 6x6 diagonal
AM1 = [0.1288, 0.1288, 0.0, 0.0, 0.0, 0.0]        # arm


def _pitch_of(quat) -> float:
    w, x, y, z = (float(v) for v in quat)
    return math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))


def _fold_ema_peak(am0, am1) -> float:
    def body(mass, inertia, amd):
        return Body(mass=mass, inertia=inertia, volume=0.0, cob=(0.0, 0.0, 0.0),
                    added_mass=torch.diag(torch.tensor(amd, dtype=torch.float32)),
                    linear_damping=torch.zeros(6, 6), quadratic_damping=torch.zeros(6, 6),
                    density=1000.0)
    chain = TwoBodyChain(body(M0, I0, am0), body(M1, I1, am1),
                         (0.0, 1.0, 0.0), ANCHOR_BASE, ANCHOR_ARM)
    out = simulate_coupled(chain, steps=STEPS, dt=DT,
                           q_traj=lambda t: (AMP * (1 - math.cos(OMEGA * t)),
                                             AMP * OMEGA * math.sin(OMEGA * t),
                                             AMP * OMEGA * OMEGA * math.cos(OMEGA * t)),
                           use_gravity=False, use_buoyancy=False)
    return math.degrees(max(abs(_pitch_of(q)) for q in out["base_quat"]))


def _momentum_peak(added0, added1) -> float:
    out = simulate_planar_added_mass(
        m0=M0, m1=M1, I0yy=I0[1], I1yy=I1[1], added0=added0, added1=added1,
        anchor_base=ANCHOR_BASE, anchor_arm=ANCHOR_ARM,
        q_traj=lambda t: (AMP * (1 - math.cos(OMEGA * t)), AMP * OMEGA * math.sin(OMEGA * t)),
        steps=STEPS, dt=DT)
    return math.degrees(float(out["base_pitch"].abs().max()))


def main() -> None:
    # planar (Xu, Zw, Mq) picked from the 6x6 diagonals: x-index 0, z-index 2, pitch-index 4
    p0 = (AM0[0], AM0[2], AM0[4])
    p1 = (AM1[0], AM1[2], AM1[4])

    rigid_fold = _fold_ema_peak([0.0] * 6, [0.0] * 6)
    rigid_mom = _momentum_peak((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    print(f"COMPARE:: RIGID   fold+EMA={rigid_fold:.4f}deg  momentum={rigid_mom:.4f}deg  "
          f"diff={abs(rigid_fold - rigid_mom):.4f}  [both should be ~2.709]", flush=True)

    am_fold = _fold_ema_peak(AM0, AM1)
    am_mom = _momentum_peak(p0, p1)
    rel = abs(am_fold - am_mom) / max(am_mom, 1e-9)
    print(f"COMPARE:: ADDEDM  fold+EMA={am_fold:.4f}deg  momentum(indep)={am_mom:.4f}deg  "
          f"rel_diff={rel:.4f}", flush=True)
    print(f"COMPARE:: VERDICT {'fold+EMA MATCHES independent momentum ref (routing OK)' if rel < 0.05 else 'fold+EMA DIVERGES from independent ref'}",
          flush=True)


if __name__ == "__main__":
    main()
