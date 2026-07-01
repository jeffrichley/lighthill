"""MuJoCo-Warp (MJWarp) cross-check — the GPU float32 solver that Newton / Isaac Lab 3.0 use.

Our float64 CPU MuJoCo run nailed the coupling to 0.00%. Newton's `SolverMuJoCo` is MJWarp
(a Warp GPU re-implementation of MuJoCo's algorithms), which runs in **float32**. The open
question before betting on Newton: does the coupling still conserve momentum / match the
analytics under MJWarp's float32 GPU path? This rebuilds the identical two-body rig and runs
it through MJWarp on the GPU.

  * free-joint: base pose/velocity ratio (PhysX 1.13-1.16; float64 MuJoCo 1.0000) + the
    momentum-conserving base reaction magnitude (~0.0335).
  * driven gate: base pitch peak vs the analytics (2.709 deg; PhysX gave 2.23, MuJoCo 2.7091).

Run (in a CUDA Docker container with mujoco + mujoco-warp):
  docker run --rm --gpus all -v <repo>:/work -w /work lighthill-mjwarp \
      python sim_validation/mjwarp_coupling_test.py
"""

from __future__ import annotations

import math

import mujoco
import mujoco_warp as mjw
import numpy as np

AMP = 0.4
OMEGA = 2.0
DT = 0.00125
BASE_DIAG = "0.285417 0.388167 0.468083"
ARM_DIAG = "0.012820 0.012820 0.000640"


def build_xml(*, driven: bool) -> str:
    damping = 'damping="200"' if driven else 'damping="0"'
    actuator = '<actuator><position joint="hinge" kp="4000"/></actuator>' if driven else ""
    integrator = "implicit" if driven else "RK4"
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


def _pitch(quat) -> float:
    return 2.0 * math.atan2(float(quat[2]), float(quat[0]))


def run_freejoint(steps: int = 2000) -> None:
    mjm = mujoco.MjModel.from_xml_string(build_xml(driven=False))
    mjd = mujoco.MjData(mjm)
    mjd.qvel[6] = 2.0
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd)

    pose, wy = [], []
    qx_max = qz_max = 0.0
    for _k in range(steps):
        mjw.step(m, d)
        qpos = d.qpos.numpy()[0]
        qvel = d.qvel.numpy()[0]
        pose.append(_pitch(qpos[3:7]))
        wy.append(float(qvel[4]))
        qx_max = max(qx_max, abs(float(qpos[4])))
        qz_max = max(qz_max, abs(float(qpos[6])))
    pose = np.array(pose)
    rate = np.gradient(pose, DT)
    mask = np.abs(rate) > 0.005
    ratio = float(np.median(np.array(wy)[mask] / rate[mask]))
    print("FREEJOINT (MJWarp float32, GPU):")
    print(f"  base wy peak={max(abs(x) for x in wy):.5f}  pose-rate peak={np.abs(rate).max():.5f}")
    print(f"  off-Y quat peaks |x|={qx_max:.2e} |z|={qz_max:.2e}")
    print(f"  ratio reported/pose = {ratio:.4f}   [PhysX 1.13-1.16; float64 MuJoCo 1.0000]")


def run_driven(steps: int = 3200) -> None:
    mjm = mujoco.MjModel.from_xml_string(build_xml(driven=True))
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd)

    pitch = []
    for k in range(steps):
        t = k * DT
        d.ctrl.assign(np.array([[AMP * (1.0 - math.cos(OMEGA * t))]], dtype=np.float32))
        mjw.step(m, d)
        pitch.append(_pitch(d.qpos.numpy()[0][3:7]))
    pk = math.degrees(max(abs(x) for x in pitch))
    print("\nDRIVEN GATE (MJWarp float32, GPU):")
    print(f"  base pitch peak = {pk:.4f} deg   [analytics 2.709; PhysX 2.23; float64 MuJoCo 2.7091]")
    print(f"  rel error vs analytics 2.709 = {abs(pk - 2.709) / 2.709 * 100:.2f}%")


if __name__ == "__main__":
    run_freejoint()
    try:
        run_driven()
    except Exception as e:
        import traceback
        print("DRIVEN failed (MJWarp ctrl/integrator API):", repr(e))
        traceback.print_exc()
