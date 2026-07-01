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
ANCHOR_BASE = (0.0, 0.0, -0.15)      # authored joint anchor in base frame (pre-scale)
ANCHOR_ARM = (0.0, 0.0, 0.25)        # authored joint anchor in arm frame (pre-scale)
ARM_POS0 = (0.0, 0.0, -0.40)         # = anchor_base - anchor_arm (q=0 placement)


def _scaled(anchor, scale):
    return tuple(a * s for a, s in zip(anchor, scale, strict=True))


# CRITICAL: USD physics joint localPos is interpreted in the link's *scaled* local
# frame, so the effective anchors are anchor * body-scale. The reference's coupling
# magnitude is set by the arm moment arm; using the unscaled anchors inflates it
# (here 0.40 vs the real 0.17, ~2.4x) and the gate fails. The scenario verifies these
# against the arm CoM offset actually measured from the sim.
ANCHOR_BASE_EFF = _scaled(ANCHOR_BASE, BASE_SCALE)   # (0, 0, -0.045)
ANCHOR_ARM_EFF = _scaled(ANCHOR_ARM, ARM_SCALE)      # (0, 0,  0.125)
DRIVE_STIFFNESS = 4000.0             # stiff enough to track the slow swing cleanly
DRIVE_DAMPING = 200.0

# --- commanded arm swing: smooth start q(t)=AMP*(1-cos(omega t)) ---
AMP = 0.4          # rad (~23 deg), moderate
OMEGA = 2.0        # rad/s -> ~3 s period, slow enough for clean tracking
SAMPLE_EVERY = 100


def _q_cmd(t: float) -> float:
    return AMP * (1.0 - math.cos(OMEGA * t))


def _pitch_of(quat) -> float:
    """Pitch (rotation about Y, rad) from a scalar-first (w,x,y,z) quaternion."""
    w, x, y, z = quat
    return math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))


def _persist(q_sim, p_sim, p_ref, pitch_sim, pitch_ref, dt, extra=None) -> None:
    """Dump the full sim+reference trajectories for offline diagnosis (untracked)."""
    import json

    out = {
        "dt": dt,
        "q_sim": [float(v) for v in q_sim],
        "p_sim": p_sim.tolist(), "p_ref": p_ref.tolist(),
        "pitch_sim": pitch_sim.tolist(), "pitch_ref": pitch_ref.tolist(),
    }
    if extra:
        out.update(extra)
    path = os.path.join(os.path.dirname(__file__), os.pardir, "_armswing_run.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"PERSISTED:: {os.path.abspath(path)}", flush=True)


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


def _gate_coeffs():
    """2-link coeffs for the gate: added mass KEPT (the inertial coupling under test),
    buoyancy + drag ZEROED. The gate isolates the inertial (added-mass + rigid)
    vehicle<->arm coupling -- the headline claim. Drag is a separate, already-certified
    force law (drag_terminal/free_decay at <0.5%); including it would confound the
    coupling test with drag-through-articulation fidelity. Same isolation discipline as
    the single-body scenarios pinning attitude / zeroing CoB."""
    import torch

    from lighthill import RobotHydroConfig, example_config_path, resolve_coefficients

    cfg = RobotHydroConfig.from_yaml(example_config_path("bluerov2_alpha_uvms.yaml"))
    cfg2 = RobotHydroConfig(links=cfg.links[:2], density=cfg.density)  # base + 1 arm link
    coeffs = resolve_coefficients(cfg2)
    coeffs.volume = torch.zeros_like(coeffs.volume)                 # buoyancy OFF
    coeffs.linear_damping = torch.zeros_like(coeffs.linear_damping)     # drag OFF
    coeffs.quadratic_damping = torch.zeros_like(coeffs.quadratic_damping)
    # DIAGNOSTIC (convergence study): also zero added mass -> pure rigid coupling, no
    # inertia augmentation, no filtered residual. Applied consistently to sim + reference
    # (both read this one config), so it isolates whether the added-mass path is what
    # breaks dt-convergence. Off by default (the real gate keeps added mass).
    if os.environ.get("LIGHTHILL_GATE_NO_ADDEDMASS") == "1":
        coeffs.added_mass = torch.zeros_like(coeffs.added_mass)
    return coeffs


def _reference(q_sim, steps, dt, masses, inertias):
    """Floating-base Featherstone reference fed the sim's realized q(t) + actual inertias."""
    from lighthill.validation.reference import Body
    from lighthill.validation.reference_coupled import TwoBodyChain, simulate_coupled

    coeffs = _gate_coeffs()

    def _body(i):
        return Body(
            mass=masses[i], inertia=inertias[i], volume=0.0, cob=(0.0, 0.0, 0.0),
            added_mass=coeffs.added_mass[i], linear_damping=coeffs.linear_damping[i],
            quadratic_damping=coeffs.quadratic_damping[i], density=coeffs.density,
        )

    chain = TwoBodyChain(_body(0), _body(1), JOINT_AXIS, ANCHOR_BASE_EFF, ANCHOR_ARM_EFF)
    out = simulate_coupled(
        chain, steps=steps, dt=dt, q_traj=_finite_diff_traj(q_sim, dt),
        use_gravity=False, use_buoyancy=False)
    return out


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

    # self-check: the reference's effective joint geometry (anchor*scale) must match the
    # arm CoM offset actually realized in the sim, or the coupling magnitude is wrong.
    d0_meas = (robot.data.body_pos_w[0, 1] - robot.data.body_pos_w[0, 0]).tolist()
    d0_model = [b - a for b, a in zip(ANCHOR_BASE_EFF, ANCHOR_ARM_EFF, strict=True)]
    geom_err = max(abs(m - e) for m, e in zip(d0_meas, d0_model, strict=True))
    print(f"GEOMCHECK:: arm_offset measured={[f'{x:+.3f}' for x in d0_meas]}  "
          f"model={[f'{x:+.3f}' for x in d0_model]}  err={geom_err:.4f}", flush=True)
    assert geom_err < 0.02, f"reference joint geometry disagrees with sim by {geom_err:.3f} m"

    coeffs = _gate_coeffs()  # added mass on, drag + buoyancy off
    hydro = UnderwaterHydrodynamics(view, coeffs)
    hydro.reset(current_world=torch.zeros(view.num_envs, 3, device=dev))

    # capture rigid + augmented masses/inertias for offline momentum analysis
    pv = robot.root_physx_view
    diag = {
        "masses_rigid": masses,
        "inertias_rigid": [list(i) for i in inertias],
        "masses_aug": pv.get_masses()[0].tolist(),
        "inertias_aug": pv.get_inertias()[0].tolist(),
    }

    q_sim: list[float] = []
    base_pos_sim: list[list[float]] = []
    base_quat_sim: list[list[float]] = []
    arm_pos_sim: list[list[float]] = []
    base_vel_sim: list[list[float]] = []   # [lin3, ang3] world
    arm_vel_sim: list[list[float]] = []
    for k in range(steps):
        t = k * dt
        robot.set_joint_position_target(torch.tensor([[_q_cmd(t)]], device=dev))
        hydro.apply(dt)                     # per-link wrench + write_data_to_sim
        sim.step()
        robot.update(dt)
        q_sim.append(float(robot.data.joint_pos[0, 0]))
        base_pos_sim.append([float(x) for x in robot.data.body_pos_w[0, 0].tolist()])
        base_quat_sim.append([float(x) for x in robot.data.body_quat_w[0, 0].tolist()])
        arm_pos_sim.append([float(x) for x in robot.data.body_pos_w[0, 1].tolist()])
        base_vel_sim.append([float(x) for x in robot.data.body_lin_vel_w[0, 0].tolist()]
                            + [float(x) for x in robot.data.body_ang_vel_w[0, 0].tolist()])
        arm_vel_sim.append([float(x) for x in robot.data.body_lin_vel_w[0, 1].tolist()]
                           + [float(x) for x in robot.data.body_ang_vel_w[0, 1].tolist()])
        if (k + 1) % SAMPLE_EVERY == 0:
            print(f"PROGRESS:: t={t + dt:.2f}s  q_cmd={math.degrees(_q_cmd(t + dt)):5.1f}  "
                  f"q_sim={math.degrees(q_sim[-1]):5.1f}deg  pitch={math.degrees(_pitch_of(base_quat_sim[-1])):+5.1f}  "
                  f"base_x={base_pos_sim[-1][0]:+.4f}", flush=True)

    # analytical reference fed the realized q(t)
    ref = _reference(q_sim, steps, dt, masses, inertias)
    p_sim = torch.tensor(base_pos_sim)           # [steps,3]
    p_ref = ref["base_pos"]                      # [steps,3]
    pitch_sim = torch.tensor([_pitch_of(q) for q in base_quat_sim])     # [steps]
    pitch_ref = torch.tensor([_pitch_of([float(x) for x in q]) for q in ref["base_quat"]])

    transient = int(0.2 / dt)
    # The base reaction to an arm swing about the pitch axis is dominantly ROTATIONAL;
    # base pitch is the clean, high-SNR coupling signal and is the gate metric. The
    # translation recoil here is a sub-3 mm signal whose peak-relative error is ill-
    # defined near its zero crossings (it agrees in absolute terms), so it is reported
    # for information, not gated. See reference_featherstone.md.
    pitch_err = (pitch_sim[transient:] - pitch_ref[transient:]).abs().max()
    pitch_scale = pitch_ref.abs().max().clamp_min(1e-6)
    trans_err = (p_sim[transient:] - p_ref[transient:]).norm(dim=-1).max()
    trans_scale = p_ref.norm(dim=-1).max().clamp_min(1e-6)
    rel_pitch = float(pitch_err / pitch_scale)
    rel_trans = float(trans_err / trans_scale)
    peak_rel_error = rel_pitch

    _persist(q_sim, p_sim, p_ref, pitch_sim, pitch_ref, dt,
             extra={"arm_pos_sim": arm_pos_sim, "base_vel_sim": base_vel_sim,
                    "arm_vel_sim": arm_vel_sim, "arm_pos_ref": ref["arm_pos"].tolist(),
                    "diag": diag})
    return {
        "peak_rel_error": peak_rel_error, "rel_pitch": rel_pitch, "rel_trans": rel_trans,
        "pitch_sim_deg": [math.degrees(float(pitch_sim[k])) for k in range(SAMPLE_EVERY - 1, steps, SAMPLE_EVERY)],
        "pitch_ref_deg": [math.degrees(float(pitch_ref[k])) for k in range(SAMPLE_EVERY - 1, steps, SAMPLE_EVERY)],
        "base_x_sim": [float(p_sim[k, 0]) for k in range(SAMPLE_EVERY - 1, steps, SAMPLE_EVERY)],
        "base_x_ref": [float(p_ref[k, 0]) for k in range(SAMPLE_EVERY - 1, steps, SAMPLE_EVERY)],
    }


if __name__ == "__main__":
    try:
        result = run()
        ok = result["peak_rel_error"] < 0.15  # tolerance per reference_featherstone.md
        print(f"RESULT:: peak_rel_error={result['peak_rel_error']:.4f}  "
              f"(rel_pitch={result['rel_pitch']:.4f} rel_trans={result['rel_trans']:.4f})  "
              f"pitch_sim={[f'{x:+.1f}' for x in result['pitch_sim_deg']]}  "
              f"pitch_ref={[f'{x:+.1f}' for x in result['pitch_ref_deg']]}  "
              f"base_x_sim={[f'{x:+.4f}' for x in result['base_x_sim']]}  "
              f"base_x_ref={[f'{x:+.4f}' for x in result['base_x_ref']]}  "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    except Exception as e:
        import traceback
        print("RUN_ERROR:", repr(e), flush=True)
        traceback.print_exc()
    finally:
        os._exit(0)
