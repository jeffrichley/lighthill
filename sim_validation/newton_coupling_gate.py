"""Newton free-floating UVMS coupling gate -- the capstone for the Isaac (Newton) path.

Reproduces the arm-swing coupling gate (``arm_swing_reaction.py``, the PhysX gate that
under-integrates the free base) on the Isaac Lab **Newton** backend, kit-less, driving the
*real* lighthill hydro stack through ``NewtonArticulationView``. Four staged checks, each an
independent known-good measurement; a reset re-zeroes the articulation between them:

  Stage A (build)  : author the 2-body rig (free base + revolute arm) on the kit-less Newton
                     ``sim.stage``; wrap as Articulation + NewtonArticulationView. Reports dims
                     + the arm CoM offset self-check (must equal anchor_base_eff-anchor_arm_eff).
  Stage B (motion) : apply a constant WORLD wrench to the free base via the lighthill adapter;
                     base |vel| must grow from 0 -> the adapter's wrench path MOVES a free body
                     on Newton (closes the cartpole |vel|=0 plumbing caveat).
  Stage C (coast)  : set an arm initial joint velocity and coast (0 drive gains, no hydro); the
                     base reported-wy / pose-rate ratio must be ~1.0 (PhysX leaked at 1.13-1.16).
  Stage D (gate)   : drive q(t)=AMP(1-cos wt) with the PD actuator while lighthill applies the
                     added-mass coupling (buoyancy/drag OFF); base pitch peak must match the
                     analytical Featherstone reference (~2.709 deg) within 15% -- the same
                     tolerance and reference the PhysX gate FAILS. **Do not loosen to pass.**

Run (in the isaaclab-newton container, lighthill on PYTHONPATH):
  PYTHONPATH=/work/src /opt/IsaacLab/isaaclab.sh -p /work/sim_validation/newton_coupling_gate.py \
      --physics newton_mjwarp --headless --visualizer none
"""

from __future__ import annotations

import argparse
import math
import os
import sys

# so we can import the PhysX gate's geometry + analytical reference helpers verbatim
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab.physics import PhysicsCfg  # noqa: E402

# identical rig geometry + analytical reference as the PhysX gate (single source of truth)
from arm_swing_reaction import (  # noqa: E402
    AMP,
    ANCHOR_ARM,
    ANCHOR_ARM_EFF,
    ANCHOR_BASE,
    ANCHOR_BASE_EFF,
    ARM_MASS,
    ARM_POS0,
    ARM_SCALE,
    BASE_MASS,
    BASE_SCALE,
    DRIVE_DAMPING,
    DRIVE_STIFFNESS,
    OMEGA,
    _pitch_of,
    _q_cmd,
    _reference,
)
from lighthill.apply_newton import NewtonArticulationView, _to_torch  # noqa: E402

# MJWarp (float32) integrates the stiff PD drive (kp=4000) stably only at a small step;
# the standalone mjwarp_coupling_test proved dt=1.25ms holds it (5ms diverges to NaN).
DT = 0.00125
STEPS = 3200  # 4 s of driven swing
TOL = 0.15  # same as reference_featherstone.md; PhysX gate fails this

parser = argparse.ArgumentParser(description="Newton free-floating UVMS coupling gate.")
# AppLauncher does NOT register --physics (it's a hydra/task override); for this task-less
# script the launch_simulation scan reads a `physics` attr, so register it ourselves.
parser.add_argument("--physics", default="newton_mjwarp",
                    help="backend: physx | newton_mjwarp | ovphysx")
add_launcher_args(parser)
parser.set_defaults(visualizer=["none"])
args_cli, _unknown = parser.parse_known_args()


def _stage(name, fn):
    try:
        out = fn()
        print(f"NEWTON_GATE:: {name} OK  {out if out is not None else ''}", flush=True)
        return out
    except Exception as ex:  # noqa: BLE001
        import traceback
        print(f"NEWTON_GATE:: {name} FAIL -> {ex!r}", flush=True)
        traceback.print_exc()
        return None


def main() -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sim import SimulationContext
    from pxr import Gf, UsdGeom, UsdPhysics

    from lighthill.apply import UnderwaterHydrodynamics

    with launch_simulation(PhysicsCfg(), args_cli) as physics_cfg:
        dev = str(getattr(args_cli, "device", "cuda:0") or "cuda:0")
        # MJWarp defaults to the explicit "euler" integrator, which diverges under the stiff
        # PD drive (kp=4000) at dt=1.25ms. "implicitfast" integrates the drive implicitly --
        # the exact setting the standalone mjwarp_coupling_test used to hold kp=4000 and hit
        # 2.7091 deg. Switch it here (before the solver is built in SimulationContext).
        integrator = os.environ.get("LIGHTHILL_INTEGRATOR", "implicitfast")
        if getattr(physics_cfg, "solver_cfg", None) is not None:
            physics_cfg.solver_cfg.integrator = integrator
        print(f"NEWTON_GATE:: integrator={integrator}", flush=True)
        sim_cfg = sim_utils.SimulationCfg(dt=DT, device=dev, gravity=(0.0, 0.0, 0.0),
                                          physics=physics_cfg)
        sim = SimulationContext(sim_cfg)
        stage = sim.stage
        print(f"NEWTON_GATE:: backend={type(physics_cfg).__name__} device={dev} stage={stage is not None}",
              flush=True)

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

        robot = _stage("A.build", lambda: Articulation(ArticulationCfg(
            prim_path=root, spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            actuators={"joint": ImplicitActuatorCfg(
                joint_names_expr=["joint"], stiffness=DRIVE_STIFFNESS, damping=DRIVE_DAMPING)},
        )))
        if robot is None:
            return
        sim.reset()

        view = _stage("A.view", lambda: NewtonArticulationView(robot))
        if view is None:
            return
        njoints = int(_to_torch(robot.data.joint_pos).shape[-1])
        print(f"NEWTON_GATE:: dims E={view.num_envs} B={view.num_bodies} joints={njoints}", flush=True)

        pos0, quat0, _ = view.body_states()
        arm_off = (pos0[0, 1] - pos0[0, 0]).tolist()
        model_off = [b - a for b, a in zip(ANCHOR_BASE_EFF, ANCHOR_ARM_EFF, strict=True)]
        geom_err = max(abs(m - e) for m, e in zip(arm_off, model_off, strict=True))
        print(f"NEWTON_GATE:: GEOMCHECK arm_off measured={[f'{x:+.3f}' for x in arm_off]} "
              f"model={[f'{x:+.3f}' for x in model_off]} err={geom_err:.4f}", flush=True)

        # rigid masses/inertias for the reference (read BEFORE hydro augments them)
        masses = [float(view.mass[0, 0]), float(view.mass[0, 1])]
        inertias = [tuple(float(x) for x in view.inertia_diag[0, 0].tolist()),
                    tuple(float(x) for x in view.inertia_diag[0, 1].tolist())]
        print(f"NEWTON_GATE:: rigid masses={masses} inertias={inertias}", flush=True)

        # rigid baseline captured at view init (before any hydro inertia augmentation)
        rigid_mass = view.mass.clone()
        rigid_inertia = view.inertia_diag.clone()

        def _reset_state():
            e = view.num_envs
            # restore the RIGID inertias: a prior stage's hydro.__init__ wrote added-mass-augmented
            # inertias via set_inertias, which persist on the articulation. Without this, the next
            # driven run starts from a contaminated inertia and can diverge.
            view.set_body_inertias(rigid_mass, rigid_inertia)
            robot.set_joint_position_target_index(target=torch.zeros(e, njoints, device=dev))
            root_pose = torch.zeros(e, 7, device=dev)
            root_pose[:, 3] = 1.0  # identity quat wxyz
            robot.write_root_pose_to_sim_index(root_pose=root_pose)
            robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(e, 6, device=dev))
            robot.write_joint_position_to_sim_index(position=torch.zeros(e, njoints, device=dev))
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros(e, njoints, device=dev))
            robot.reset()
            for _ in range(3):  # settle the solver at zero target before the stage begins
                robot.write_data_to_sim()
                sim.step()
                robot.update(DT)

        # -- Stage B: a constant world wrench on the free base MUST produce base motion --
        def _stage_b():
            _reset_state()
            # isolate the wrench->motion path: free the joint so the stiff PD can't interfere
            robot.write_joint_stiffness_to_sim_index(stiffness=torch.zeros(view.num_envs, njoints, device=dev))
            robot.write_joint_damping_to_sim_index(damping=torch.zeros(view.num_envs, njoints, device=dev))
            e, b = view.num_envs, view.num_bodies
            wrench = torch.zeros(e, b, 6, device=dev)
            wrench[:, 0, 0] = 40.0  # +40 N on the base in world +x
            n = 400
            for _k in range(n):
                view.set_external_wrench(wrench)
                sim.step()
                robot.update(DT)
            _p, _q, vel = view.body_states()
            vmax = float(vel[0, 0, :3].norm())
            # a = F/m -> v ~= (40/13.7)*(n*DT); sanity-check we're in the physical ballpark
            v_expect = (40.0 / BASE_MASS) * (n * DT)
            return f"base |vel|={vmax:.4f} after {n} steps of +40N (expect ~{v_expect:.3f})  [must be > 0, finite]"
        _stage("B.free-body-wrench", _stage_b)

        # -- Stage C: free-joint coast; base pose must track its reported velocity (ratio ~1) --
        def _stage_c():
            _reset_state()
            robot.write_joint_stiffness_to_sim_index(stiffness=torch.zeros(view.num_envs, njoints, device=dev))
            robot.write_joint_damping_to_sim_index(damping=torch.zeros(view.num_envs, njoints, device=dev))
            robot.write_joint_velocity_to_sim_index(
                velocity=torch.full((view.num_envs, njoints), 2.0, device=dev))
            robot.update(DT)
            qw, qy, wby = [], [], []
            for _k in range(2400):
                view.set_external_wrench(torch.zeros(view.num_envs, view.num_bodies, 6, device=dev))
                sim.step()
                robot.update(DT)
                _p, quat, vel = view.body_states()
                qw.append(float(quat[0, 0, 0]))
                qy.append(float(quat[0, 0, 2]))
                wby.append(float(vel[0, 0, 4]))  # base body-frame wy (Y is the pitch axis)
            pose = [2.0 * math.atan2(y, w) for y, w in zip(qy, qw, strict=True)]
            rate = [(pose[k + 1] - pose[k - 1]) / (2 * DT) for k in range(1, len(pose) - 1)]
            wb = wby[1:-1]
            ratios = sorted(wb[i] / rate[i] for i in range(len(rate)) if abs(rate[i]) > 0.005)
            med = ratios[len(ratios) // 2] if ratios else float("nan")
            return (f"base wy peak={max(abs(x) for x in wby):.4f} pose-rate peak={max(abs(x) for x in rate):.4f} "
                    f"ratio reported/pose={med:.4f}  [~1.0 good; PhysX 1.13-1.16]")
        _stage("C.freejoint-coast", _stage_c)

        # -- Stage D: driven arm swing vs the analytical floating-base Featherstone reference --
        # Two configs, both compared to the SAME reference fed the realized q(t):
        #   D1 rigid       (added mass OFF): tests the SIMULATOR's floating-base coupling -- the
        #                  headline known-good (analytics/raw-MJWarp = 2.709 deg). PhysX fails at 18%.
        #   D2 added-mass  (added mass ON) : tests lighthill's added-mass augmentation on top. This
        #                  is an OPEN design question (fold into inertia vs push through the xfrc
        #                  residual); reported, not gated. **Do not loosen D1's tolerance to pass.**
        kp = float(os.environ.get("LIGHTHILL_KP", DRIVE_STIFFNESS))
        kd = float(os.environ.get("LIGHTHILL_KD", DRIVE_DAMPING))

        def _run_driven(tag: str, *, added_mass: bool):
            from arm_swing_reaction import _gate_coeffs  # added mass on, buoyancy/drag off
            os.environ["LIGHTHILL_GATE_NO_ADDEDMASS"] = "0" if added_mass else "1"
            _reset_state()
            robot.write_joint_stiffness_to_sim_index(
                stiffness=torch.full((view.num_envs, njoints), kp, device=dev))
            robot.write_joint_damping_to_sim_index(
                damping=torch.full((view.num_envs, njoints), kd, device=dev))
            hydro = UnderwaterHydrodynamics(view, _gate_coeffs())
            hydro.reset(current_world=torch.zeros(view.num_envs, 3, device=dev))

            q_sim, base_quat = [], []
            for k in range(STEPS):
                t = k * DT
                robot.set_joint_position_target_index(
                    target=torch.full((view.num_envs, njoints), _q_cmd(t), device=dev))
                hydro.apply(DT)              # per-link wrench + write_data_to_sim
                sim.step()
                robot.update(DT)
                q_sim.append(float(_to_torch(robot.data.joint_pos)[0, 0]))
                _p, quat, _v = view.body_states()
                base_quat.append([float(x) for x in quat[0, 0].tolist()])

            ref = _reference(q_sim, STEPS, DT, masses, inertias)
            pitch_sim = torch.tensor([_pitch_of(q) for q in base_quat])
            pitch_ref = torch.tensor([_pitch_of([float(x) for x in q]) for q in ref["base_quat"]])
            transient = int(0.2 / DT)
            err = (pitch_sim[transient:] - pitch_ref[transient:]).abs().max()
            scale = pitch_ref.abs().max().clamp_min(1e-6)
            rel = float(err / scale)
            peak_sim = math.degrees(float(pitch_sim.abs().max()))
            peak_ref = math.degrees(float(pitch_ref.abs().max()))
            print(f"NEWTON_GATE:: {tag} base_pitch_sim_peak={peak_sim:.4f}deg ref_peak={peak_ref:.4f}deg "
                  f"rel_err={rel:.4f}", flush=True)
            return rel, peak_sim, peak_ref

        # D1: the headline gate -- rigid floating-base coupling vs the 2.709 deg known-good.
        def _stage_d1():
            rel, ps, pr = _run_driven("D1.rigid", added_mass=False)
            ok = rel < TOL
            print(f"NEWTON_GATE:: D1.RESULT rigid coupling rel_err={rel:.4f} (tol {TOL}) sim={ps:.4f}deg "
                  f"ref={pr:.4f}deg -> {'PASS' if ok else 'FAIL'}", flush=True)
            assert ok, f"rigid floating-base coupling off by {rel:.1%} (> {TOL:.0%})"
            return f"rel_err={rel:.4f} sim={ps:.4f}deg ref={pr:.4f}deg (known-good 2.709) PASS"
        d1 = _stage("D1.rigid-coupling-gate", _stage_d1)

        # D2: diagnostic -- added-mass augmentation on top (open routing question, not gated).
        def _stage_d2():
            rel, ps, pr = _run_driven("D2.added-mass", added_mass=True)
            note = "MATCH" if rel < TOL else "OPEN(added-mass routing)"
            print(f"NEWTON_GATE:: D2.RESULT added-mass rel_err={rel:.4f} sim={ps:.4f}deg ref={pr:.4f}deg "
                  f"peaks-agree={abs(ps - pr) / max(pr, 1e-6):.4f} -> {note}", flush=True)
            return f"rel_err={rel:.4f} peaks {ps:.3f}/{pr:.3f}deg -> {note}"
        _stage("D2.added-mass-diagnostic", _stage_d2)

        print(f"NEWTON_GATE:: ALL_DONE  headline(D1 rigid coupling)={'PASS' if d1 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        os._exit(0)
