"""Maximal-coordinate coupling test: does dropping ArticulationRootAPI fix the free
base's pose under-integration?

Same two rigid bodies + revolute joint as the free-joint articulation test, but with
NO ArticulationRootAPI -- so PhysX simulates them as two independent maximal-coordinate
rigid bodies whose joint is an iteratively-solved CONSTRAINT, not a reduced-coordinate
articulation. The arm is given an initial angular velocity and the system coasts (no
gravity, no hydro, no damping, free joint). We compare the base's reported (momentum-
conserving) angular velocity to its actual pose rotation rate.

  * ratio reported/pose ~1.0  -> maximal coordinates integrate the base pose faithfully;
    the articulation reduced-coordinate root was the problem, and this is a usable config.
  * ratio ~1.16 (same as the articulation) -> maximal coordinates ALSO leak; the fault is
    deeper (constraint solver) and PhysX cannot do this coupling regardless.

Reference points: articulation free-joint gave 1.16; a lone RigidObject spin gave 1.003.

Run (Isaac env):  OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_PY" sim_validation/maximal_coupling_test.py
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ARM_JOINT_VEL = 2.0
DT = 0.00125
STEPS = 2000

BASE_SCALE = (0.5, 0.4, 0.3)
ARM_SCALE = (0.08, 0.08, 0.5)
BASE_MASS = 13.7
ARM_MASS = 0.6
ANCHOR_BASE = (0.0, 0.0, -0.15)   # authored; scaled -> (0,0,-0.045)
ANCHOR_ARM = (0.0, 0.0, 0.25)     # authored; scaled -> (0,0, 0.125)
# place the arm CoM where the (scaled) joint anchors coincide, so the maximal-coordinate
# constraint is already satisfied at reset (no violent initial correction impulse)
ARM_POS0 = (0.0, 0.0, -0.045 - 0.125)  # = (0,0,-0.170)


def run(dt: float = DT, steps: int = STEPS) -> None:
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    _app = app_launcher.app

    import isaaclab.sim as sim_utils
    import omni.usd
    import torch
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationContext
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    dev = "cuda:0"
    sim_cfg = sim_utils.SimulationCfg(
        dt=dt, device=dev, gravity=(0.0, 0.0, 0.0),
        physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True),
    )
    sim = SimulationContext(sim_cfg)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World/Rig")  # plain Xform -- NO ArticulationRootAPI

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

    base = make_link("/World/Rig/base", BASE_SCALE, BASE_MASS, (0.0, 0.0, 0.0))
    arm = make_link("/World/Rig/arm", ARM_SCALE, ARM_MASS, ARM_POS0)
    j = UsdPhysics.RevoluteJoint.Define(stage, "/World/Rig/joint")
    j.CreateBody0Rel().SetTargets([base.GetPath()])
    j.CreateBody1Rel().SetTargets([arm.GetPath()])
    j.CreateAxisAttr().Set("Y")
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(*ANCHOR_BASE))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(*ANCHOR_ARM))
    # free joint: zero-gain drive so it's a pure passive hinge (no motor)
    dr = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    dr.CreateTypeAttr().Set("force")
    dr.CreateStiffnessAttr().Set(0.0)
    dr.CreateDampingAttr().Set(0.0)

    base_obj = RigidObject(RigidObjectCfg(
        prim_path="/World/Rig/base", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))))
    arm_obj = RigidObject(RigidObjectCfg(
        prim_path="/World/Rig/arm", spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=ARM_POS0)))
    sim.reset()
    # verified uniform-box values (matches the articulation runs' get_inertias readouts)
    m0, m1 = BASE_MASS, ARM_MASS
    I0yy = BASE_MASS / 12.0 * (BASE_SCALE[0] ** 2 + BASE_SCALE[2] ** 2)
    I1yy = ARM_MASS / 12.0 * (ARM_SCALE[0] ** 2 + ARM_SCALE[2] ** 2)

    # consistent IC: arm rotates about the joint at ARM_JOINT_VEL, base at rest ->
    # arm CoM (0.125 below the joint) gets linear vel omega x r = (-0.125*wrel, 0, 0)
    av = torch.zeros(1, 6, device=dev)
    av[0, 0] = -0.125 * ARM_JOINT_VEL
    av[0, 4] = ARM_JOINT_VEL
    arm_obj.write_root_velocity_to_sim(av.clone())

    qy, qw, w_base, w_arm, Ly = [], [], [], [], []
    qx_max, qz_max = 0.0, 0.0
    for _k in range(steps):
        sim.step()
        base_obj.update(dt)
        arm_obj.update(dt)
        bq = base_obj.data.root_quat_w[0]  # (w,x,y,z)
        qw.append(float(bq[0]))
        qy.append(float(bq[2]))
        qx_max = max(qx_max, abs(float(bq[1])))
        qz_max = max(qz_max, abs(float(bq[3])))
        p0 = base_obj.data.root_pos_w[0]
        p1 = arm_obj.data.root_pos_w[0]
        v0 = base_obj.data.root_lin_vel_w[0]
        v1 = arm_obj.data.root_lin_vel_w[0]
        w0y = float(base_obj.data.root_ang_vel_w[0, 1])
        w1y = float(arm_obj.data.root_ang_vel_w[0, 1])
        w_base.append(w0y)
        w_arm.append(w1y)
        # total angular momentum about origin, Y (must be CONSTANT -> conserved)
        ly = (I0yy * w0y + I1yy * w1y
              + m0 * (float(p0[2]) * float(v0[0]) - float(p0[0]) * float(v0[2]))
              + m1 * (float(p1[2]) * float(v1[0]) - float(p1[0]) * float(v1[2])))
        Ly.append(ly)

    pose = [2.0 * math.atan2(y, w) for y, w in zip(qy, qw, strict=True)]
    pose_rate = [(pose[k + 1] - pose[k - 1]) / (2 * dt) for k in range(1, len(pose) - 1)]
    wb = w_base[1:-1]
    ratios = sorted(wb[i] / pose_rate[i] for i in range(len(pose_rate)) if abs(pose_rate[i]) > 0.005)
    med = ratios[len(ratios) // 2] if ratios else float("nan")
    # momentum-conservation quality: spread of L_y relative to its scale
    ly_mean = sum(Ly) / len(Ly)
    ly_spread = max(Ly) - min(Ly)
    ly_scale = max(abs(x) for x in Ly)
    print(f"\nMAXIMAL:: dt={dt} steps={steps}", flush=True)
    print(f"  base reported wy peak = {max(abs(x) for x in w_base):.5f}", flush=True)
    print(f"  base pose rate   peak = {max(abs(x) for x in pose_rate):.5f}", flush=True)
    print(f"  arm  reported wy peak = {max(abs(x) for x in w_arm):.5f}", flush=True)
    print(f"  base quat off-Y peaks: |x|={qx_max:.2e} |z|={qz_max:.2e}  "
          f"[~0 => pure-Y rotation, pose metric valid]", flush=True)
    print(f"  ratio reported/pose (median) = {med:.4f}   "
          f"[~1.0 => pose tracks velocity; articulation was 1.16]", flush=True)
    print(f"  L_y: mean={ly_mean:.5e} spread={ly_spread:.2e} ({ly_spread/ly_scale*100:.2f}% of scale)"
          f"  [small spread => momentum CONSERVED -> base reaction is physical]", flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        import traceback
        print("RUN_ERROR:", repr(e), flush=True)
        traceback.print_exc()
    finally:
        os._exit(0)
