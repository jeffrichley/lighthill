"""In-sim validation 4 (the gate): UVMS vehicle<->arm coupling.

A free-floating 2-body articulation (vehicle base + one revolute-jointed arm link)
is driven through a commanded arm swing while lighthill applies per-link hydro and
the articulation solver propagates the coupling. The measured base reaction must
match the analytical floating-base Featherstone reference
(`lighthill.validation.reference_coupled`). This is the headline UVMS claim and the
hard gate: a wrong coupling (frame sign, wrench frame, inertia augmentation,
residual routing) fails it. **Do not loosen the tolerance to pass.**

Isolation discipline (see `sim_validation/reference_featherstone.md`):
  * Gravity OFF and buoyancy OFF (volumes zeroed) -- isolate the momentum coupling,
    which is present with or without gravity, and avoid the augmented-mass-gravity
    sink (PhysX applies gravity on the added-mass-augmented mass).
  * Feed the sim's *actual* realized joint angle q(t) to the reference (a stiff PD
    drive does not track perfectly) -- same parity discipline as restoring.py reading
    the actual inertia.
  * Exclude the startup transient before scoring; normalize error by the signal scale.

Run (in the Isaac env):  OMNI_KIT_ACCEPT_EULA=YES python sim_validation/arm_swing_reaction.py
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

# --- authored articulation geometry (self-consistent: arm CoM at anchor_base-anchor_arm) ---
BASE_SCALE = (0.5, 0.4, 0.3)
ARM_SCALE = (0.08, 0.08, 0.5)
BASE_MASS = 13.7
ARM_MASS = 0.6
JOINT_AXIS = (0.0, 1.0, 0.0)         # base-frame revolute axis (Y)
ANCHOR_BASE = (0.0, 0.0, -0.15)      # joint anchor in base frame
ANCHOR_ARM = (0.0, 0.0, 0.25)        # joint anchor in arm frame
ARM_POS0 = (0.0, 0.0, -0.40)         # = anchor_base - anchor_arm (q=0 placement)
DRIVE_STIFFNESS = 4000.0             # stiff enough to track the slow swing cleanly
DRIVE_DAMPING = 200.0

# --- commanded arm swing: smooth start q(t)=AMP*(1-cos(omega t)) ---
AMP = 0.4          # rad (~23 deg), moderate
OMEGA = 2.0        # rad/s -> ~3 s period, slow enough for clean tracking
SAMPLE_EVERY = 100


def _q_cmd(t: float) -> float:
    return AMP * (1.0 - math.cos(OMEGA * t))


def _finite_diff_traj(q_list, dt):
    """Realized q(t) array -> a q_traj(t)->(q,qd,qdd) closure (central differences)."""
    import torch

    q = torch.tensor(q_list, dtype=torch.float32)
    n = q.shape[0]
    qd = torch.zeros_like(q)
    qd[1:-1] = (q[2:] - q[:-2]) / (2 * dt)
    qd[0] = (q[1] - q[0]) / dt
    qd[-1] = (q[-1] - q[-2]) / dt
    qdd = torch.zeros_like(q)
    qdd[1:-1] = (qd[2:] - qd[:-2]) / (2 * dt)
    qdd[0] = (qd[1] - qd[0]) / dt
    qdd[-1] = (qd[-1] - qd[-2]) / dt

    def traj(t: float) -> tuple[float, float, float]:
        i = min(int(round(t / dt)), n - 1)
        return float(q[i]), float(qd[i]), float(qdd[i])

    return traj


def _reference(q_sim, steps, dt, masses, inertias):
    """Floating-base Featherstone reference fed the sim's realized q(t) + actual inertias."""
    import torch

    from lighthill import RobotHydroConfig, example_config_path, resolve_coefficients
    from lighthill.validation.reference import Body
    from lighthill.validation.reference_coupled import TwoBodyChain, simulate_coupled

    cfg = RobotHydroConfig.from_yaml(example_config_path("bluerov2_alpha_uvms.yaml"))
    cfg2 = RobotHydroConfig(links=cfg.links[:2], density=cfg.density)  # base + 1 arm link
    coeffs = resolve_coefficients(cfg2)

    def _body(i):
        return Body(
            mass=masses[i], inertia=inertias[i], volume=0.0, cob=(0.0, 0.0, 0.0),
            added_mass=coeffs.added_mass[i], linear_damping=coeffs.linear_damping[i],
            quadratic_damping=coeffs.quadratic_damping[i], density=coeffs.density,
        )

    chain = TwoBodyChain(_body(0), _body(1), JOINT_AXIS, ANCHOR_BASE, ANCHOR_ARM)
    out = simulate_coupled(
        chain, steps=steps, dt=dt, q_traj=_finite_diff_traj(q_sim, dt),
        use_gravity=False, use_buoyancy=False)
    return out, torch


def run(steps: int = 800, dt: float = 0.005) -> dict:
    """Run the in-sim arm-swing scenario; return the base reaction vs reference + error."""
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app  # noqa: F841 — held to keep Isaac alive

    import isaaclab.sim as sim_utils
    import omni.usd
    import torch
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sim import SimulationContext
    from pxr import Gf, UsdGeom, UsdPhysics

    from lighthill import RobotHydroConfig, example_config_path, resolve_coefficients
    from lighthill.apply import UnderwaterHydrodynamics
    from lighthill.apply_isaac import IsaacArticulationView

    dev = "cuda:0"
    # gravity OFF: isolate the momentum coupling, avoid the augmented-mass-gravity sink.
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
        return p

    base = make_link(root + "/base", BASE_SCALE, BASE_MASS, (0.0, 0.0, 0.0))
    arm = make_link(root + "/arm", ARM_SCALE, ARM_MASS, ARM_POS0)
    j = UsdPhysics.RevoluteJoint.Define(stage, root + "/joint")
    j.CreateBody0Rel().SetTargets([base.GetPath()])
    j.CreateBody1Rel().SetTargets([arm.GetPath()])
    j.CreateAxisAttr().Set("Y")
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(*ANCHOR_BASE))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(*ANCHOR_ARM))
    d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    d.CreateTypeAttr().Set("force")
    d.CreateStiffnessAttr().Set(DRIVE_STIFFNESS)
    d.CreateDampingAttr().Set(DRIVE_DAMPING)
    d.CreateTargetPositionAttr().Set(0.0)

    robot = Articulation(ArticulationCfg(
        prim_path=root, spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={"joint": ImplicitActuatorCfg(
            joint_names_expr=["joint"], stiffness=DRIVE_STIFFNESS, damping=DRIVE_DAMPING)},
    ))
    sim.reset()

    view = IsaacArticulationView(robot)
    # rigid masses/inertias for the reference (read before hydro augments them)
    masses = [float(view.mass[0, 0]), float(view.mass[0, 1])]
    inertias = [tuple(float(x) for x in view.inertia_diag[0, 0].tolist()),
                tuple(float(x) for x in view.inertia_diag[0, 1].tolist())]

    cfg = RobotHydroConfig.from_yaml(example_config_path("bluerov2_alpha_uvms.yaml"))
    cfg2 = RobotHydroConfig(links=cfg.links[:2], density=cfg.density)
    coeffs = resolve_coefficients(cfg2)
    coeffs.volume = torch.zeros_like(coeffs.volume)  # buoyancy OFF
    hydro = UnderwaterHydrodynamics(view, coeffs)
    hydro.reset(current_world=torch.zeros(view.num_envs, 3, device=dev))

    q_sim: list[float] = []
    base_pos_sim: list[list[float]] = []
    for k in range(steps):
        t = k * dt
        robot.set_joint_position_target(torch.tensor([[_q_cmd(t)]], device=dev))
        hydro.apply(dt)                     # per-link wrench + write_data_to_sim
        sim.step()
        robot.update(dt)
        q_sim.append(float(robot.data.joint_pos[0, 0]))
        base_pos_sim.append([float(x) for x in robot.data.body_pos_w[0, 0].tolist()])
        if (k + 1) % SAMPLE_EVERY == 0:
            print(f"PROGRESS:: t={t + dt:.2f}s  q_cmd={math.degrees(_q_cmd(t + dt)):5.1f}  "
                  f"q_sim={math.degrees(q_sim[-1]):5.1f}deg  "
                  f"base_x={base_pos_sim[-1][0]:+.4f} base_z={base_pos_sim[-1][2]:+.4f}", flush=True)

    # analytical reference fed the realized q(t)
    ref, _torch = _reference(q_sim, steps, dt, masses, inertias)
    p_sim = torch.tensor(base_pos_sim)          # [steps,3]
    p_ref = ref["base_pos"]                      # [steps,3]

    transient = int(0.2 / dt)
    err = (p_sim[transient:] - p_ref[transient:]).norm(dim=-1)
    scale = p_ref.norm(dim=-1).max().clamp_min(1e-6)
    peak_rel_error = float(err.max() / scale)
    return {
        "peak_rel_error": peak_rel_error,
        "base_x_sim": [float(p_sim[k, 0]) for k in range(SAMPLE_EVERY - 1, steps, SAMPLE_EVERY)],
        "base_x_ref": [float(p_ref[k, 0]) for k in range(SAMPLE_EVERY - 1, steps, SAMPLE_EVERY)],
        "base_z_sim": [float(p_sim[k, 2]) for k in range(SAMPLE_EVERY - 1, steps, SAMPLE_EVERY)],
        "base_z_ref": [float(p_ref[k, 2]) for k in range(SAMPLE_EVERY - 1, steps, SAMPLE_EVERY)],
        "q_sim_deg": [math.degrees(q_sim[k]) for k in range(SAMPLE_EVERY - 1, steps, SAMPLE_EVERY)],
    }


if __name__ == "__main__":
    try:
        result = run()
        ok = result["peak_rel_error"] < 0.15  # tolerance per reference_featherstone.md
        print(f"RESULT:: peak_rel_error={result['peak_rel_error']:.4f}  "
              f"base_x_sim={[f'{x:+.4f}' for x in result['base_x_sim']]}  "
              f"base_x_ref={[f'{x:+.4f}' for x in result['base_x_ref']]}  "
              f"base_z_sim={[f'{x:+.4f}' for x in result['base_z_sim']]}  "
              f"base_z_ref={[f'{x:+.4f}' for x in result['base_z_ref']]}  "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    except Exception as e:
        import traceback
        print("RUN_ERROR:", repr(e), flush=True)
        traceback.print_exc()
    finally:
        os._exit(0)
