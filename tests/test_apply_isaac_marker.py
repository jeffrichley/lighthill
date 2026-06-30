"""Real-Isaac in-sim validation gate (marked ``real_sim``).

Skipped unless ``LIGHTHILL_REAL_SIM_OK=1`` (needs a real Isaac Sim + GPU), so the CPU
gate and CI never run it. Each scenario in ``sim_validation/`` is a standalone Isaac
script: exactly one ``SimulationApp`` may exist per process and Kit hangs on teardown
(the scripts force-exit), so we invoke each as a subprocess and assert its ``PASS`` line
rather than launching Isaac in-process.

On the Isaac host:  LIGHTHILL_REAL_SIM_OK=1 uv run pytest tests/test_apply_isaac_marker.py -p no:xdist
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

real_sim = pytest.mark.skipif(
    os.environ.get("LIGHTHILL_REAL_SIM_OK") != "1",
    reason="needs a real Isaac Sim + GPU (set LIGHTHILL_REAL_SIM_OK=1)",
)

_ROOT = Path(__file__).resolve().parents[1]


def _run_scenario(script: str) -> str:
    proc = subprocess.run(
        [sys.executable, str(_ROOT / "sim_validation" / script)],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"},
    )
    return proc.stdout


@real_sim
@pytest.mark.parametrize("script", ["drag_terminal.py", "free_decay.py", "restoring.py"])
def test_in_sim_scenario_matches_cpu_reference(script: str) -> None:
    out = _run_scenario(script)
    assert "PASS" in out and "FAIL" not in out, f"{script} did not PASS:\n{out[-2000:]}"
