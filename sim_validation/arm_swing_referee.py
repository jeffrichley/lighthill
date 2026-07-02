"""Three-way referee for the arm-swing coupling (paper verification, offline).

The arm-swing gate compares the free base's pitch reaction between PhysX (sim) and the
Featherstone reference (``reference_coupled``). At fine dt those two disagree on the
*pitch* amplitude (a clean ~0.81 ratio), and momentum arguments alone cannot say which
is right -- so this script brings in a **third, independent** analytical model, the
pure-NumPy planar momentum-reconstruction solver (``reference_planar_momentum``), which
enforces total linear + angular momentum = 0 and reconstructs the base velocity
directly (no composite-inertia matrix, no acceleration integration).

It reads the persisted gate run (``_armswing_run.json``; write it by running
``arm_swing_reaction.py`` -- ideally with ``LIGHTHILL_GATE_NO_ADDEDMASS=1`` so the
rigid coupling is isolated) and prints the three base-pitch peaks side by side. If the
two independent analytics agree and the sim is the outlier, the coupling reference is
vindicated and the residual is a PhysX solver/drive artifact -- not a lighthill error.

Run (any Python with numpy + lighthill on the path; no Isaac needed):
  python sim_validation/arm_swing_referee.py
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

try:  # package import (pytest / repo root on path) vs standalone `python sim_validation/...`
    from sim_validation.reference_planar_momentum import simulate_planar_momentum
except ModuleNotFoundError:
    from reference_planar_momentum import simulate_planar_momentum

# effective (scaled) joint anchors -- must match arm_swing_reaction.ANCHOR_*_EFF
ANCHOR_BASE_EFF = (0.0, 0.0, -0.045)
ANCHOR_ARM_EFF = (0.0, 0.0, 0.125)


def main() -> None:
    path = os.path.join(os.path.dirname(__file__), os.pardir, "_armswing_run.json")
    with open(path) as f:
        d = json.load(f)

    dt = d["dt"]
    q = np.array(d["q_sim"], dtype=float)
    n = len(q)
    m0, m1 = d["diag"]["masses_rigid"]
    I0yy = d["diag"]["inertias_rigid"][0][1]
    I1yy = d["diag"]["inertias_rigid"][1][1]

    # realized joint rate by central difference (same convention as the reference path)
    qd = np.zeros_like(q)
    qd[1:-1] = (q[2:] - q[:-2]) / (2 * dt)
    qd[0] = (q[1] - q[0]) / dt
    qd[-1] = (q[-1] - q[-2]) / dt

    def traj(t: float) -> tuple[float, float]:
        i = min(int(round(t / dt)), n - 1)
        return float(q[i]), float(qd[i])

    out = simulate_planar_momentum(
        m0=m0, m1=m1, I0yy=I0yy, I1yy=I1yy,
        anchor_base=ANCHOR_BASE_EFF, anchor_arm=ANCHOR_ARM_EFF,
        q_traj=traj, steps=n, dt=dt)

    pk_mom = math.degrees(float(np.abs(out["base_pitch"]).max()))
    pk_sim = math.degrees(max(abs(x) for x in d["pitch_sim"]))
    pk_ref = math.degrees(max(abs(x) for x in d["pitch_ref"]))
    cons = float(np.abs(out["ang_mom"]).max())

    print(f"REFEREE:: dt={dt * 1e3:.2f}ms  n={n}  (momentum model conserves Ly to {cons:.1e})")
    print("  base pitch peak (deg):")
    print(f"    PhysX sim            = {pk_sim:.4f}")
    print(f"    Featherstone ref     = {pk_ref:.4f}")
    print(f"    Momentum reconstr.   = {pk_mom:.4f}   [independent referee]")
    print(f"  ratios:  momentum/featherstone={pk_mom / pk_ref:.4f}   "
          f"sim/featherstone={pk_sim / pk_ref:.4f}")
    verdict = ("ANALYTICS AGREE -> reference correct, PhysX is the outlier"
               if abs(pk_mom / pk_ref - 1.0) < 0.02 else
               "ANALYTICS DISAGREE -> reference bug (momentum model matches sim)")
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
