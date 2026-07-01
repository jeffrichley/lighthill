"""Probe: what does an Isaac Lab Articulation expose on the NEWTON backend?

lighthill's engine dependency is the 5-member ArticulationView Protocol. This checks,
on a live Newton-backed env, which of those the existing PhysX adapter (IsaacArticulationView)
can reuse as-is and which need a Newton path -- i.e. exactly what must change to integrate
lighthill into Newton. Read-only introspection; force-exits.
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


def main() -> None:
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()
        scene = env.unwrapped.scene
        arts = scene.articulations
        print("PROBE:: articulations =", list(arts.keys()), flush=True)
        robot = next(iter(arts.values()))
        print("PROBE:: robot class =", type(robot).__module__ + "." + type(robot).__name__, flush=True)

        # (1) read attributes the Protocol needs
        d = robot.data
        print(f"PROBE:: robot.num_bodies={robot.num_bodies} num_instances={robot.num_instances} "
              f"body_names={getattr(robot, 'body_names', '?')}", flush=True)
        for a in ["body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
                  "default_mass", "default_inertia", "root_pos_w"]:
            v = getattr(d, a, "MISSING")
            shp = f"shape={tuple(v.shape)} ndim={v.ndim}" if hasattr(v, "shape") else str(v)
            print(f"PROBE:: data.{a} -> {shp}", flush=True)

        # (2) write surfaces the adapter uses
        for a in ["set_external_force_and_torque", "write_data_to_sim",
                  "root_physx_view", "root_newton_view", "num_bodies", "num_instances"]:
            print(f"PROBE:: robot.{a} present = {hasattr(robot, a)}", flush=True)

        # (3) does the PhysX-style external wrench call work on Newton?
        try:
            e_, b_ = d.body_pos_w.shape[0], d.body_pos_w.shape[1]
            dev = str(env_cfg.sim.device)
            f = torch.zeros(e_, b_, 3, device=dev)
            robot.set_external_force_and_torque(f, f.clone(), is_global=True)
            robot.write_data_to_sim()
            print("PROBE:: set_external_force_and_torque(is_global=True) -> OK on Newton", flush=True)
        except Exception as ex:
            print(f"PROBE:: set_external_force_and_torque FAILED -> {ex!r}", flush=True)

        # (4) newton/xfrc handles anywhere on the robot?
        hits = [a for a in dir(robot) if "newton" in a.lower() or "xfrc" in a.lower()]
        print("PROBE:: newton/xfrc attrs on robot =", hits, flush=True)
        # inertia write surface
        for a in ["set_masses", "set_inertias", "get_masses", "get_inertias"]:
            src = "root_physx_view" if hasattr(robot, "root_physx_view") else None
            print(f"PROBE:: inertia write '{a}' via root_physx_view = "
                  f"{hasattr(getattr(robot, 'root_physx_view', object()), a)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        os._exit(0)
