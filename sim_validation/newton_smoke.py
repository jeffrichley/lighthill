"""Minimal proof that Isaac Lab runs a physics sim on the Newton (MJWarp) backend, headless
and kit-less (no Isaac Sim). Mirrors zero_agent's launcher setup but prints per-step output
and force-exits (Isaac-style teardown can hang).

Run (inside the isaaclab-newton container):
  ./isaaclab.sh -p /work/sim_validation/newton_smoke.py \
      --task Isaac-Cartpole-Direct --num_envs 16 --headless --visualizer none \
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

parser = argparse.ArgumentParser(description="Newton backend smoke test.")
parser.add_argument("--num_envs", type=int, default=16)
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
        print(f"NEWTON_ENV_OK backend={type(env_cfg.sim.physics).__name__} "
              f"num_envs={env.unwrapped.num_envs} device={env_cfg.sim.device}", flush=True)
        env.reset()
        try:
            act = torch.zeros(env.action_space.shape, device=str(env_cfg.sim.device))
        except Exception:
            act = torch.as_tensor(env.action_space.sample(), device=str(env_cfg.sim.device))
        for i in range(30):
            _obs, rew, _term, _trunc, _info = env.step(act)
            if i % 10 == 0:
                print(f"NEWTON_STEP {i:3d}  reward_mean={float(rew.float().mean()):+.4f}", flush=True)
        print("NEWTON_SMOKE_OK ran 30 steps on the Newton MJWarp backend", flush=True)
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        os._exit(0)
