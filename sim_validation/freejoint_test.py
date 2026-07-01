"""Decisive fork: is the base pose under-rotation caused by the JOINT DRIVE, or by the
floating articulation ROOT itself?

The arm-swing gate showed the free base's reported orientation under-integrates its own
(momentum-conserving) angular velocity by ~18% when the position drive is active. This
test removes the drive entirely: the revolute joint is FREE (zero stiffness/damping), the
arm is given an initial joint velocity, and the whole 2-body articulation coasts with no
gravity, no hydro, no damping -- pure internal momentum exchange. With zero external
torque the base pose MUST conserve angular momentum.

  * If base pose rate == base velocity buffer (ratio ~1.0) and momentum is conserved
    -> the DRIVE was the culprit; the sim is salvageable with different actuation.
  * If base pose still under-rotates vs its velocity buffer (ratio ~1.2)
    -> the reduced-coordinate floating root itself is broken under coupling.

Run (Isaac env):  OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_PY" sim_validation/freejoint_test.py
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ARM_JOINT_VEL = 2.0   # rad/s initial arm rate relative to base (drives the swing, no motor)
DT = 0.00125
STEPS = 2000

BASE_SCALE = (0.5, 0.4, 0.3)
ARM_SCALE = (0.08, 0.08, 0.5)
BASE_MASS = 13.7
ARM_MASS = 0.6
ANCHOR_BASE = (0.0, 0.0, -0.15)
ANCHOR_ARM = (0.0, 0.0, 0.25)
ARM_POS0 = (0.0, 0.0, -0.40)


def run(dt: float = DT, steps: int = STEPS) -> None:
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    _app = app_launcher.app

    import isaaclab.sim as sim_utils
    import omni.usd
    import torch
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sim import SimulationContext
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    dev = "cuda:0"
    sim_cfg = sim_utils.SimulationCfg(
        dt=dt, device=dev, gravity=(0.0, 0.0, 0.0),
        physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True),
    )
    sim = SimulationContext(sim_cfg)
    stage = omni.usd.get_context().get_stage()
    root = "/World/Robot"
    UsdGeom.Xform.Define(stage, root)
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(root))

    def make_link(path, scale, mass, pos):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)
        x = UsdGeom.Xformable(cube)
        x.AddTranslateOp().Set(Gf.Vec3d(*pos))
        x.AddScaleOp().Set(Gf.Vec3f(*scale))
        p = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(p)
        UsdPhysics.RigidBodyAPI.Apply(p)
        UsdPhysics.MassAPI.Apply(p).GetMassAttr().Set(mass)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(p)   # zero PhysX default 0.05 damping
        rb.CreateLinearDampingAttr().Set(0.0)
        rb.CreateAngularDampingAttr().Set(0.0)
        return p

    base = make_link(root + "/base", BASE_SCALE, BASE_MASS, (0.0, 0.0, 0.0))
    arm = make_link(root + "/arm", ARM_SCALE, ARM_MASS, ARM_POS0)
    j = UsdPhysics.RevoluteJoint.Define(stage, root + "/joint")
    j.CreateBody0Rel().SetTargets([base.GetPath()])
    j.CreateBody1Rel().SetTargets([arm.GetPath()])
    j.CreateAxisAttr().Set("Y")
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(*ANCHOR_BASE))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(*ANCHOR_ARM))
    # FREE joint: drive present but zero gains -> no motor, no PD, pure passive hinge
    dr = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    dr.CreateTypeAttr().Set("force")
    dr.CreateStiffnessAttr().Set(0.0)
    dr.CreateDampingAttr().Set(0.0)

    robot = Articulation(ArticulationCfg(
        prim_path=root, spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={"joint": ImplicitActuatorCfg(
            joint_names_expr=["joint"], stiffness=0.0, damping=0.0)}))
    sim.reset()

    # give the ARM an initial joint velocity; base starts at rest
    q0 = torch.zeros(1, 1, device=dev)
    qd0 = torch.full((1, 1), ARM_JOINT_VEL, device=dev)
    robot.write_joint_state_to_sim(q0, qd0)

    qy, qw, w_body, w1_body, jp, jv = [], [], [], [], [], []
    for _k in range(steps):
        sim.step()
        robot.update(dt)
        dq = robot.data.body_quat_w[0, 0]
        qw.append(float(dq[0]))
        qy.append(float(dq[2]))
        w_body.append(float(robot.data.body_ang_vel_w[0, 0, 1]))
        w1_body.append(float(robot.data.body_ang_vel_w[0, 1, 1]))
        jp.append(float(robot.data.joint_pos[0, 0]))
        jv.append(float(robot.data.joint_vel[0, 0]))

    pose = [2.0 * math.atan2(y, w) for y, w in zip(qy, qw, strict=True)]
    pose_rate = [(pose[k + 1] - pose[k - 1]) / (2 * dt) for k in range(1, len(pose) - 1)]
    wb = w_body[1:-1]
    # compare reported base velocity to actual pose rate (active region: base spinning)
    ratios = [wb[i] / pose_rate[i] for i in range(len(pose_rate)) if abs(pose_rate[i]) > 0.005]
    ratios.sort()
    med = ratios[len(ratios) // 2] if ratios else float("nan")
    print(f"\nFREEJOINT:: dt={dt} steps={steps}  joint spun {math.degrees(jp[-1]):.0f} deg "
          f"(vel {jv[0]:.3f}->{jv[-1]:.3f})", flush=True)
    print(f"  base reported wy peak = {max(abs(x) for x in w_body):.5f}", flush=True)
    print(f"  base pose rate   peak = {max(abs(x) for x in pose_rate):.5f}", flush=True)
    print(f"  ratio reported/pose (median) = {med:.4f}   "
          f"[~1.0 => DRIVE was the cause; ~1.2 => floating root itself]", flush=True)
    # momentum: base spin from pose rate vs momentum-conserving expectation -I1*(w0+jv)/...
    print(f"  arm reported wy peak = {max(abs(x) for x in w1_body):.5f}", flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        import traceback
        print("RUN_ERROR:", repr(e), flush=True)
        traceback.print_exc()
    finally:
        os._exit(0)
