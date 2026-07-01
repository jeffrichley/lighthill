"""Phase 0 coast test (+ Phase 1 property audit) for the arm-swing pitch discrepancy.

Question this isolates: does the FREE base actually conserve angular momentum in our
sim configuration, decoupled from the arm swing? Spin two free-floating articulations
about Y at t=0 and let them COAST (gravity 0, no hydro, joint drive merely holding
q=0) for a few seconds:

  * SOLO -- a lone base body (single-link free articulation).
  * DUO  -- the full base + revolute-jointed arm gate articulation, arm held at q=0.

With zero external torque, omega_y must stay constant. If it DECAYS we have an
unmodeled external torque / damping on the base -- i.e. WE are misusing PhysX, and the
arm-swing pitch deficit is that same bleed, not a PhysX pitch error. If SOLO holds but
DUO decays, the loss is articulation/drive-specific. The per-body damping / solver /
inertia audit is printed alongside so a decay can be attributed immediately.

Run (Isaac env):  OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_PY" sim_validation/coast_test.py
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

W0Y = 0.5          # rad/s, initial base spin about Y (small: well under any max-vel clamp)
COAST_TIME = 3.0   # s
DT = 0.00125       # s -- the finest gate dt, where the coupling deficit was worst (~19%)

# geometry copied verbatim from arm_swing_reaction.py
BASE_SCALE = (0.5, 0.4, 0.3)
ARM_SCALE = (0.08, 0.08, 0.5)
BASE_MASS = 13.7
ARM_MASS = 0.6
ANCHOR_BASE = (0.0, 0.0, -0.15)
ANCHOR_ARM = (0.0, 0.0, 0.25)
ARM_POS0 = (0.0, 0.0, -0.40)
DRIVE_STIFFNESS = 4000.0
DRIVE_DAMPING = 200.0


def run(dt: float = DT, steps: int | None = None) -> None:
    if steps is None:
        steps = int(COAST_TIME / dt)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    _app = app_launcher.app

    import isaaclab.sim as sim_utils
    import omni.usd
    import torch
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationContext
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    keep_damping = os.environ.get("LIGHTHILL_GATE_KEEP_DAMPING") == "1"

    dev = "cuda:0"
    sim_cfg = sim_utils.SimulationCfg(
        dt=dt, device=dev, gravity=(0.0, 0.0, 0.0),
        physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True),
    )
    sim = SimulationContext(sim_cfg)
    stage = omni.usd.get_context().get_stage()

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
        if not keep_damping:  # default: zero PhysX's 0.05 damping (Exp 2: pure-spin conservation)
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(p)
            rb.CreateLinearDampingAttr().Set(0.0)
            rb.CreateAngularDampingAttr().Set(0.0)
        return p

    # SOLO: single free rigid body (a RigidObject, NOT an articulation -- PhysX rejects a
    # jointless single-body articulation). Cleanest baseline for free-body momentum.
    solo_path = "/World/Solo"
    make_link(solo_path, BASE_SCALE, BASE_MASS, (0.0, 0.0, 0.0))

    # DUO: base + arm gate articulation, offset in X so the two don't overlap
    OFF = 5.0
    duo_root = "/World/Duo"
    UsdGeom.Xform.Define(stage, duo_root)
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(duo_root))
    base = make_link(duo_root + "/base", BASE_SCALE, BASE_MASS, (OFF, 0.0, 0.0))
    arm = make_link(duo_root + "/arm", ARM_SCALE, ARM_MASS,
                    (OFF + ARM_POS0[0], ARM_POS0[1], ARM_POS0[2]))
    j = UsdPhysics.RevoluteJoint.Define(stage, duo_root + "/joint")
    j.CreateBody0Rel().SetTargets([base.GetPath()])
    j.CreateBody1Rel().SetTargets([arm.GetPath()])
    j.CreateAxisAttr().Set("Y")
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(*ANCHOR_BASE))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(*ANCHOR_ARM))
    dr = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    dr.CreateTypeAttr().Set("force")
    dr.CreateStiffnessAttr().Set(DRIVE_STIFFNESS)
    dr.CreateDampingAttr().Set(DRIVE_DAMPING)
    dr.CreateTargetPositionAttr().Set(0.0)

    solo = RigidObject(RigidObjectCfg(
        prim_path=solo_path, spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))))
    duo = Articulation(ArticulationCfg(
        prim_path=duo_root, spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(pos=(OFF, 0.0, 0.0)),
        actuators={"joint": ImplicitActuatorCfg(
            joint_names_expr=["joint"], stiffness=DRIVE_STIFFNESS, damping=DRIVE_DAMPING)}))
    sim.reset()

    # ---- Phase 1 property audit (read what PhysX actually got) ----
    def audit_prim(prim_path, attrs):
        p = stage.GetPrimAtPath(prim_path)
        out = {}
        for a in attrs:
            at = p.GetAttribute(a)
            out[a.split(":")[-1]] = (at.Get() if at and at.IsValid() else "UNSET(default)")
        return out

    body_attrs = ["physxRigidBody:linearDamping", "physxRigidBody:angularDamping",
                  "physxRigidBody:sleepThreshold", "physxRigidBody:stabilizationThreshold",
                  "physxRigidBody:maxAngularVelocity", "physxRigidBody:disableGravity"]
    art_attrs = ["physxArticulation:solverPositionIterationCount",
                 "physxArticulation:solverVelocityIterationCount",
                 "physxArticulation:sleepThreshold", "physxArticulation:stabilizationThreshold",
                 "physxArticulation:enabledSelfCollisions"]
    print("AUDIT:: solo (rigid)", audit_prim(solo_path, body_attrs), flush=True)
    print("AUDIT:: duo/base   ", audit_prim(duo_root + "/base", body_attrs), flush=True)
    print("AUDIT:: duo/arm    ", audit_prim(duo_root + "/arm", body_attrs), flush=True)
    print("AUDIT:: duo  artic ", audit_prim(duo_root, art_attrs), flush=True)
    for nm, art in [("solo", solo), ("duo", duo)]:
        pv = art.root_physx_view
        try:
            print(f"AUDIT:: {nm} masses={pv.get_masses()[0].tolist()} "
                  f"inertias={pv.get_inertias()[0].tolist()}", flush=True)
        except Exception as e:
            print(f"AUDIT:: {nm} mass/inertia read err: {e!r}", flush=True)

    # ---- set the initial base spin about +Y (world) ----
    rv = torch.zeros(1, 6, device=dev)
    rv[0, 4] = W0Y  # [lin_x,lin_y,lin_z, ang_x,ang_y,ang_z]
    solo.write_root_velocity_to_sim(rv.clone())
    duo.write_root_velocity_to_sim(rv.clone())

    ts, w_solo, w_duo, w_duo_arm = [], [], [], []
    qy_solo, qw_solo, qy_duo, qw_duo = [], [], [], []  # pose (Y-angle) tracking
    for k in range(steps):
        duo.set_joint_position_target(torch.zeros(1, 1, device=dev))
        sim.step()
        solo.update(dt)
        duo.update(dt)
        ts.append(k * dt)
        w_solo.append(float(solo.data.root_ang_vel_w[0, 1]))  # RigidObject: [N,3]
        w_duo.append(float(duo.data.body_ang_vel_w[0, 0, 1]))
        w_duo_arm.append(float(duo.data.body_ang_vel_w[0, 1, 1]))
        sq = solo.data.root_quat_w[0]  # (w,x,y,z)
        dq = duo.data.body_quat_w[0, 0]
        qw_solo.append(float(sq[0]))
        qy_solo.append(float(sq[2]))
        qw_duo.append(float(dq[0]))
        qy_duo.append(float(dq[2]))

    def decay_pct(w):
        return (w[0] - w[-1]) / w[0] * 100.0 if w[0] else float("nan")

    def implied_damping(w, dt, steps):
        # PhysX angular damping applies omega *= 1/(1+d*dt) per step -> omega_end/omega_0=(1/(1+d dt))^steps
        r = w[-1] / w[0] if w[0] else float("nan")
        if r <= 0 or r >= 1:
            return float("nan")
        return (math.exp(-math.log(r) / steps) - 1.0) / dt

    print(f"\nCOAST:: w0y={W0Y} dt={dt} steps={steps} total_time={dt*steps:.2f}s", flush=True)
    print(f"  SOLO base wy: {w_solo[0]:.6f} -> {w_solo[-1]:.6f}  decay={decay_pct(w_solo):+.2f}%  "
          f"implied_ang_damping={implied_damping(w_solo, dt, steps):.4f}", flush=True)
    print(f"  DUO  base wy: {w_duo[0]:.6f} -> {w_duo[-1]:.6f}  decay={decay_pct(w_duo):+.2f}%  "
          f"implied_ang_damping={implied_damping(w_duo, dt, steps):.4f}", flush=True)
    print(f"  DUO  arm  wy: {w_duo_arm[0]:.6f} -> {w_duo_arm[-1]:.6f}  "
          f"(should track base if arm is rigid with it)", flush=True)
    # Exp 2: does the reported angular velocity match the actual pose rotation rate for a
    # PURE spin? Pose Y-angle = 2*atan2(qy, qw); its mean rate vs the mean reported wy.
    def pose_rate(qy, qw):
        ang = [2.0 * math.atan2(y, w) for y, w in zip(qy, qw, strict=True)]
        return (ang[-1] - ang[0]) / (dt * (len(ang) - 1))
    solo_pose_rate = pose_rate(qy_solo, qw_solo)
    duo_pose_rate = pose_rate(qy_duo, qw_duo)
    solo_wy_mean = sum(w_solo) / len(w_solo)
    duo_wy_mean = sum(w_duo) / len(w_duo)
    print("  VEL-vs-POSE (pure spin):", flush=True)
    print(f"    SOLO: mean reported wy={solo_wy_mean:.6f}  pose rate={solo_pose_rate:.6f}  "
          f"ratio rep/pose={solo_wy_mean/solo_pose_rate:.4f}", flush=True)
    print(f"    DUO : mean reported wy={duo_wy_mean:.6f}  pose rate={duo_pose_rate:.6f}  "
          f"ratio rep/pose={duo_wy_mean/duo_pose_rate:.4f}", flush=True)
    print("  curve (t, solo_wy, duo_wy):", flush=True)
    stride = max(1, steps // 12)
    for k in range(0, steps, stride):
        print(f"    t={ts[k]:.3f}  solo={w_solo[k]:.6f}  duo={w_duo[k]:.6f}", flush=True)

    import json
    out = {"dt": dt, "w0y": W0Y, "steps": steps, "t": ts,
           "solo_wy": w_solo, "duo_wy": w_duo, "duo_arm_wy": w_duo_arm}
    path = os.path.join(os.path.dirname(__file__), os.pardir, "_coast_run.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"PERSISTED:: {os.path.abspath(path)}", flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        import traceback
        print("RUN_ERROR:", repr(e), flush=True)
        traceback.print_exc()
    finally:
        os._exit(0)
