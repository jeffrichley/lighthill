"""End-to-end: run lighthill's UnderwaterHydrodynamics against a NEWTON articulation.

Wraps a live Newton-backed Isaac Lab articulation in NewtonArticulationView and drives the
*real* lighthill hydro stack through it -- proving the adapter's three surfaces work on Newton:
inertia write (set_body_inertias at init), state read (body_states each step), and wrench write
(set_external_wrench -> Newton xfrc). The carrier articulation is the cartpole; the hydro coeffs
are the first N links of the bluerov config, so this validates PLUMBING (no errors, forces
transmitted, sim steps), not UVMS physics -- that comes with the dedicated Newton scene.

Run (in the isaaclab-newton container, lighthill on PYTHONPATH):
  PYTHONPATH=/work/src ./isaaclab.sh -p /work/sim_validation/newton_hydro_test.py \
      --task Isaac-Cartpole-Direct --num_envs 4 --headless --visualizer none \
      env.sim.physics=newton_mjwarp
"""

import argparse
import os
import sys

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, default="Isaac-Cartpole-Direct")
add_launcher_args(parser)
parser.set_defaults(visualizer=["none"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def _stage(name, fn):
    try:
        out = fn()
        print(f"NEWTON_HYDRO:: {name} OK  {out if out is not None else ''}", flush=True)
        return out
    except Exception as ex:
        import traceback
        print(f"NEWTON_HYDRO:: {name} FAIL -> {ex!r}", flush=True)
        traceback.print_exc()
        return None


def main() -> None:
    from lighthill import RobotHydroConfig, example_config_path, resolve_coefficients
    from lighthill.apply import UnderwaterHydrodynamics
    from lighthill.apply_newton import NewtonArticulationView

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()
        robot = next(iter(env.unwrapped.scene.articulations.values()))
        dev = str(env_cfg.sim.device)

        view = _stage("view", lambda: NewtonArticulationView(robot))
        if view is None:
            return
        print(f"NEWTON_HYDRO:: dims E={view.num_envs} B={view.num_bodies} "
              f"mass={tuple(view.mass.shape)} inertia={tuple(view.inertia_diag.shape)}", flush=True)

        def _reads():
            pos, quat, vel = view.body_states()
            return f"pos={tuple(pos.shape)} quat={tuple(quat.shape)} vel={tuple(vel.shape)} dev={pos.device}"
        _stage("body_states", _reads)

        def _coeffs():
            cfg = RobotHydroConfig.from_yaml(example_config_path("bluerov2_alpha_uvms.yaml"))
            sub = RobotHydroConfig(links=cfg.links[:view.num_bodies], density=cfg.density)
            return resolve_coefficients(sub)
        coeffs = _stage("coeffs", _coeffs)
        if coeffs is None:
            return

        hydro = _stage("hydro_init(set_body_inertias)",
                       lambda: UnderwaterHydrodynamics(view, coeffs))
        if hydro is None:
            return
        _stage("hydro_reset", lambda: hydro.reset(current_world=torch.zeros(view.num_envs, 3, device=dev)))

        sim = env.unwrapped.sim
        dt = float(env_cfg.sim.dt)

        def _steploop():
            for _k in range(20):
                hydro.apply(dt)          # compute + apply per-body wrench via Newton xfrc
                sim.step()
                robot.update(dt)
            _p, _q, vel = view.body_states()
            return f"ran 20 steps; final |vel| max={float(vel.norm(dim=-1).max()):.4f}"
        _stage("step_loop(apply+sim.step)", _steploop)
        print("NEWTON_HYDRO:: ALL_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        os._exit(0)
