"""MuJoCo cross-check of the UVMS vehicle<->arm coupling (no Isaac, no PhysX).

Rebuilds the exact two-body rig -- a free-floating base + one revolute-jointed arm,
matched masses/inertias/geometry -- in MuJoCo, and runs the same two experiments that
PhysX failed:

  1. free-joint momentum exchange: arm given an initial hinge velocity, system coasts;
     check the base pose tracks its velocity (PhysX gave ratio 1.13-1.16 here).
  2. driven gate: hinge PD-driven through q(t)=AMP(1-cos(omega t)); compare the base
     pitch peak to the independent momentum-reconstruction reference (analytics = ~2.71
     deg; PhysX gave 2.23).

MuJoCo integrates qpos directly from qvel, so a correct engine has pose == integral of
velocity by construction; the real questions are whether it CONSERVES momentum and
MATCHES the analytics. If yes, the deficit is PhysX-specific and MuJoCo is a usable
backend.

Run:  PYTHONPATH=src python sim_validation/mujoco_coupling_test.py
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

AMP = 0.4
OMEGA = 2.0
DT = 0.00125

# uniform-box inertias (identical to the PhysX get_inertias readouts)
BASE_DIAG = "0.285417 0.388167 0.468083"
ARM_DIAG = "0.012820 0.012820 0.000640"


def build_xml(*, driven: bool) -> str:
    damping = 'damping="200"' if driven else 'damping="0"'
    actuator = '<actuator><position joint="hinge" kp="4000"/></actuator>' if driven else ""
    # implicitfast handles the stiff PD drive/damping (RK4 goes unstable); free joint fine on RK4
    integrator = "implicitfast" if driven else "RK4"
    return f"""
<mujoco>
  <option timestep="{DT}" gravity="0 0 0" integrator="{integrator}"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <freejoint/>
      <inertial pos="0 0 0" mass="13.7" diaginertia="{BASE_DIAG}"/>
      <geom type="box" size="0.25 0.2 0.15" density="0"/>
      <body name="arm" pos="0 0 -0.045">
        <joint name="hinge" type="hinge" axis="0 1 0" pos="0 0 0" {damping}/>
        <inertial pos="0 0 -0.125" mass="0.6" diaginertia="{ARM_DIAG}"/>
        <geom type="box" size="0.04 0.04 0.25" pos="0 0 -0.125" density="0"/>
      </body>
    </body>
  </worldbody>
  {actuator}
</mujoco>
"""


def _pitch(qpos_quat: np.ndarray) -> float:
    # MuJoCo quat = (w,x,y,z); base rotates about Y -> angle = 2*atan2(qy, qw)
    return 2.0 * math.atan2(qpos_quat[2], qpos_quat[0])


def run_freejoint(steps: int = 2000) -> None:
    model = mujoco.MjModel.from_xml_string(build_xml(driven=False))
    data = mujoco.MjData(model)
    data.qvel[6] = 2.0  # hinge initial rate (freejoint occupies qvel[0:6])
    mujoco.mj_forward(model, data)

    pose, wy = [], []
    qx_max = qz_max = 0.0
    for _k in range(steps):
        mujoco.mj_step(model, data)
        quat = data.qpos[3:7].copy()
        pose.append(_pitch(quat))
        wy.append(float(data.qvel[4]))          # base angular vel about (local=world) Y
        qx_max = max(qx_max, abs(float(quat[1])))
        qz_max = max(qz_max, abs(float(quat[3])))
    pose = np.array(pose)
    rate = np.gradient(pose, DT)
    m = np.abs(rate) > 0.005
    ratio = float(np.median(np.array(wy)[m] / rate[m]))
    print("FREEJOINT (MuJoCo):")
    print(f"  base wy peak={max(abs(x) for x in wy):.5f}  pose-rate peak={np.abs(rate).max():.5f}")
    print(f"  off-Y quat peaks |x|={qx_max:.2e} |z|={qz_max:.2e}")
    print(f"  ratio reported/pose (median) = {ratio:.4f}   "
          f"[PhysX articulation 1.16, maximal 1.13; ~1.0 => correct]")


def run_driven(steps: int = 3200) -> None:
    model = mujoco.MjModel.from_xml_string(build_xml(driven=True))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    q_sim, pitch_sim, wy = [], [], []
    for k in range(steps):
        t = k * DT
        data.ctrl[0] = AMP * (1.0 - math.cos(OMEGA * t))
        mujoco.mj_step(model, data)
        q_sim.append(float(data.qpos[7]))       # hinge angle (after freejoint's 7 qpos)
        pitch_sim.append(_pitch(data.qpos[3:7].copy()))
        wy.append(float(data.qvel[4]))
    pitch_sim = np.array(pitch_sim)
    q_sim = np.array(q_sim)

    # feed MuJoCo's realized q(t) to the independent momentum-reconstruction reference
    try:  # package import (repo root on path) vs standalone `python sim_validation/...`
        from sim_validation.reference_planar_momentum import simulate_planar_momentum
    except ModuleNotFoundError:
        from reference_planar_momentum import simulate_planar_momentum
    qd = np.gradient(q_sim, DT)

    def traj(t):
        i = min(int(round(t / DT)), len(q_sim) - 1)
        return float(q_sim[i]), float(qd[i])

    ref = simulate_planar_momentum(
        m0=13.7, m1=0.6, I0yy=0.388167, I1yy=0.012820,
        anchor_base=(0.0, 0.0, -0.045), anchor_arm=(0.0, 0.0, 0.125),
        q_traj=traj, steps=len(q_sim), dt=DT)
    pk_ref = math.degrees(float(np.abs(ref["base_pitch"]).max()))
    pk_sim = math.degrees(float(np.abs(pitch_sim).max()))
    # pose-vs-velocity consistency
    rate = np.gradient(pitch_sim, DT)
    m = np.abs(rate) > 0.002
    ratio = float(np.median(np.array(wy)[m] / rate[m]))
    print("\nDRIVEN GATE (MuJoCo):")
    print(f"  base pitch peak: MuJoCo={pk_sim:.4f} deg   analytics(reconstruction)={pk_ref:.4f} deg")
    print(f"  rel error = {abs(pk_sim - pk_ref) / pk_ref * 100:.2f}%   "
          f"[PhysX was 18% low at this dt]")
    print(f"  pose/velocity ratio = {ratio:.4f}")


if __name__ == "__main__":
    run_freejoint()
    run_driven()
