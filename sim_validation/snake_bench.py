"""Throughput benchmark: an N-link serpentine swimmer with lighthill hydro on the Newton
(MJWarp) backend, across E parallel GPU-replicated envs. Answers the one decision-critical
unknown for the CL study: env-steps/sec vs env count on this GPU (and, later, per node).

What it does each step: drive a traveling-wave CPG on the yaw joints, apply the real lighthill
per-link hydro (drag + added mass; buoyancy off, gravity off), step Newton, and time it. It is
a THROUGHPUT probe, not a physics/locomotion validation.

Run (in the isaaclab-newton container):
  PYTHONPATH=/work/src /opt/IsaacLab/isaaclab.sh -p /work/sim_validation/snake_bench.py \
      --physics newton_mjwarp --headless --visualizer none --num_envs 256 --links 10 --steps 300
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.physics import PhysicsCfg

from lighthill.apply_newton import NewtonArticulationView, _to_torch

SEG_LEN = 0.15      # m, segment length (along +X, the snake's long axis)
SEG_W = 0.08        # m, segment width/height
SEG_MASS = 1.0      # kg per segment (~neutral; gravity off so magnitude is not critical)
DRIVE_KP = 25.0
DRIVE_KD = 2.0
CPG_AMP = 0.35      # rad
CPG_FREQ = 1.0      # Hz
CPG_WAVES = 1.5     # spatial wavelengths along the body

parser = argparse.ArgumentParser(description="Snake swimmer throughput benchmark (Newton).")
parser.add_argument("--physics", default="newton_mjwarp")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--links", type=int, default=10)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--warmup", type=int, default=50)
parser.add_argument("--dt", type=float, default=0.005)
add_launcher_args(parser)
parser.set_defaults(visualizer=["none"])
args_cli, _unknown = parser.parse_known_args()


def _author_snake_usd(path: str, n: int) -> None:
    """Author a complete N-link yaw-plane serpentine articulation to a .usd file."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Snake")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    def link(i: int):
        p = f"/Snake/link_{i}"
        cube = UsdGeom.Cube.Define(stage, p)
        cube.GetSizeAttr().Set(1.0)
        x = UsdGeom.Xformable(cube)
        x.AddTranslateOp().Set(Gf.Vec3d(i * SEG_LEN, 0.0, 0.0))
        x.AddScaleOp().Set(Gf.Vec3f(SEG_LEN, SEG_W, SEG_W))
        pr = cube.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(pr)
        UsdPhysics.MassAPI.Apply(pr).GetMassAttr().Set(SEG_MASS)
        # Newton's default body damping is 0; hydro provides all damping, so nothing to author.
        return pr

    links = [link(i) for i in range(n)]
    for i in range(1, n):
        j = UsdPhysics.RevoluteJoint.Define(stage, f"/Snake/joint_{i}")
        j.CreateBody0Rel().SetTargets([links[i - 1].GetPath()])
        j.CreateBody1Rel().SetTargets([links[i].GetPath()])
        j.CreateAxisAttr().Set("Z")                     # yaw plane
        j.CreateLocalPos0Attr().Set(Gf.Vec3f(0.5, 0.0, 0.0))   # +X face of prev (scaled frame)
        j.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.5, 0.0, 0.0))  # -X face of this
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        d.CreateTypeAttr().Set("force")
        d.CreateStiffnessAttr().Set(DRIVE_KP)
        d.CreateDampingAttr().Set(DRIVE_KD)
        d.CreateTargetPositionAttr().Set(0.0)
    stage.GetRootLayer().Save()


def _snake_config(n: int):
    """N identical slender-cylinder segments (axis X): transverse added mass + cross-flow drag."""
    from lighthill import RobotHydroConfig, resolve_coefficients
    from lighthill.config import AddedMassSpec, LinkConfig

    def seg(i: int) -> LinkConfig:
        return LinkConfig(
            name=f"link_{i}",
            volume=0.0,  # buoyancy off for the throughput probe
            center_of_buoyancy=(0.0, 0.0, 0.0),
            added_mass=AddedMassSpec(kind="cylinder", radius=SEG_W / 2, length=SEG_LEN, axis="x"),
            # cross-flow quadratic drag: small axial, large transverse (y,z) + yaw
            linear_damping=(0.0,) * 6,
            quadratic_damping=(2.0, 40.0, 40.0, 0.2, 0.2, 0.2),
        )

    cfg = RobotHydroConfig(links=tuple(seg(i) for i in range(n)))
    return resolve_coefficients(cfg)


def main() -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils.configclass import configclass

    from lighthill.apply import UnderwaterHydrodynamics

    n = args_cli.links
    usd_path = f"/tmp/snake_{n}.usd"
    if not os.path.exists(usd_path):
        _author_snake_usd(usd_path, n)
    print(f"SNAKE_BENCH:: snake usd={usd_path} links={n}", flush=True)

    @configclass
    class SnakeSceneCfg(InteractiveSceneCfg):
        robot: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Snake",
            spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            actuators={"joints": ImplicitActuatorCfg(
                joint_names_expr=["joint_.*"], stiffness=DRIVE_KP, damping=DRIVE_KD)},
        )

    with launch_simulation(PhysicsCfg(), args_cli) as physics_cfg:
        dev = str(getattr(args_cli, "device", "cuda:0") or "cuda:0")
        if getattr(physics_cfg, "solver_cfg", None) is not None:
            physics_cfg.solver_cfg.integrator = "implicitfast"
        sim = SimulationContext(sim_utils.SimulationCfg(dt=args_cli.dt, device=dev,
                                                        gravity=(0.0, 0.0, 0.0), physics=physics_cfg))
        scene = InteractiveScene(SnakeSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0))
        sim.reset()
        robot = scene["robot"]

        view = NewtonArticulationView(robot)
        njoints = int(_to_torch(robot.data.joint_pos).shape[-1])
        print(f"SNAKE_BENCH:: E={view.num_envs} B={view.num_bodies} joints={njoints} dev={dev}", flush=True)

        coeffs = _snake_config(n)
        hydro = UnderwaterHydrodynamics(view, coeffs)
        hydro.reset(current_world=torch.zeros(view.num_envs, 3, device=dev))

        # traveling-wave CPG joint targets: q_i(t) = A sin(2pi f t - k i)
        e = view.num_envs
        idx = torch.arange(njoints, device=dev, dtype=torch.float32)
        k = 2.0 * math.pi * CPG_WAVES / max(njoints, 1)
        dt = args_cli.dt

        def drive(step: int) -> None:
            t = step * dt
            q = CPG_AMP * torch.sin(2.0 * math.pi * CPG_FREQ * t - k * idx)  # [njoints]
            robot.set_joint_position_target_index(target=q.unsqueeze(0).expand(e, njoints).contiguous())

        def step(i: int) -> None:
            drive(i)
            hydro.apply(dt)
            sim.step()
            scene.update(dt)

        for i in range(args_cli.warmup):
            step(i)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for i in range(args_cli.steps):
            step(args_cli.warmup + i)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        sps = args_cli.steps / elapsed
        env_sps = sps * e
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"SNAKE_BENCH:: RESULT envs={e} links={n} steps={args_cli.steps} "
              f"wallclock={elapsed:.3f}s  steps/s={sps:.1f}  ENV-STEPS/s={env_sps:,.0f}  "
              f"gpu_mem={mem_gb:.2f}GB", flush=True)
        print("SNAKE_BENCH:: ALL_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        os._exit(0)
