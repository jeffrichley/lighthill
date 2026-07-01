"""Timestep-convergence study for the arm-swing coupling gate (paper verification evidence).

Runs the gate at a single physics timestep `dt` (from LIGHTHILL_CONV_DT), holding the
total simulated time and the commanded swing FIXED so the same physical motion is
compared at every resolution. Sweep dt across runs: if the sim-vs-reference peak pitch
error shrinks as dt -> 0, the residual is integration (discretization) error and both
independent models converge to the same coupled dynamics -- that is what "verification"
means. A plateau at finite error would instead flag a real modeling residual to hunt
down.

One SimulationApp per process, so run one dt per invocation, e.g.:

  for dt in 0.010 0.005 0.0025 0.00125; do
    LIGHTHILL_CONV_DT=$dt OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_PY" sim_validation/convergence_study.py
  done

Each invocation prints one CONV:: line; collect them into the convergence table.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

TOTAL_TIME = 4.0  # s, fixed across dt so the same swing is resolved at every timestep

if __name__ == "__main__":
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from sim_validation.arm_swing_reaction import run

        dt = float(os.environ.get("LIGHTHILL_CONV_DT", "0.005"))
        steps = round(TOTAL_TIME / dt)
        result = run(steps=steps, dt=dt)
        pk_ref = max(abs(x) for x in result["pitch_ref_deg"])
        pk_sim = max(abs(x) for x in result["pitch_sim_deg"])
        print(f"CONV:: dt={dt:.5f} steps={steps} "
              f"peak_rel_error={result['peak_rel_error']:.4f} "
              f"rel_pitch={result['rel_pitch']:.4f} rel_trans={result['rel_trans']:.4f} "
              f"pitch_ref_peak_deg={pk_ref:.3f} pitch_sim_peak_deg={pk_sim:.3f}", flush=True)
    except Exception as e:
        import traceback
        print("RUN_ERROR:", repr(e), flush=True)
        traceback.print_exc()
    finally:
        os._exit(0)
