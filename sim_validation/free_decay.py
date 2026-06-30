"""In-sim validation 1: free velocity decay, Isaac glue vs Plan A CPU reference.

A body is released with an initial surge velocity and no forcing; lighthill drag decays
it. The same body + initial condition is run through the Plan A CPU integrator and the
two surge-decay trajectories must agree. Attitude is pinned and the body is neutral with
zero CoB, isolating the translational drag law (same harness rationale as drag_terminal).

Run (in the Isaac env):  OMNI_KIT_ACCEPT_EULA=YES python sim_validation/free_decay.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

MASS = 13.73   # kg, ~neutral against volume 0.0134 m^3
U0 = 1.2       # m/s, initial surge velocity
SAMPLE_EVERY = 200  # steps between trajectory samples


def _cpu_reference(steps: int, dt: float) -> list[float]:
    """Surge-decay samples from the Plan A CPU integrator for the same body + IC."""
    import torch

    from lighthill import RobotHydroConfig, example_config_path, resolve_coefficients
    from lighthill.validation.reference import Body, simulate

    coeffs = resolve_coefficients(
        RobotHydroConfig.from_yaml(example_config_path("bluerov2_auv.yaml")))
    body = Body(
        mass=MASS, inertia=(0.26, 0.23, 0.37), volume=float(coeffs.volume[0]),
        cob=(0.0, 0.0, 0.0), added_mass=coeffs.added_mass[0],
        linear_damping=coeffs.linear_damping[0], quadratic_damping=coeffs.quadratic_damping[0],
        density=coeffs.density,
    )
    traj = simulate(body, steps=steps, dt=dt, vel0=torch.tensor([U0, 0.0, 0.0]))
    u = traj["twist"][:, 0]
    return [float(u[k]) for k in range(SAMPLE_EVERY - 1, steps, SAMPLE_EVERY)]


def run(steps: int = 800, dt: float = 0.005) -> dict:
    """Run the in-sim free-decay scenario; return {u_sim, u_ref, max_rel_error}."""
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app  # noqa: F841 — held to keep Isaac alive

    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationContext

    from lighthill import RobotHydroConfig, example_config_path, resolve_coefficients
    from lighthill.apply import UnderwaterHydrodynamics
    from lighthill.apply_isaac import IsaacArticulationView
    from lighthill.frames import quat_to_rotation_matrix

    u_ref = _cpu_reference(steps, dt)

    dev = "cuda:0"
    sim_cfg = sim_utils.SimulationCfg(
        dt=dt, device=dev, gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True),
    )
    sim = SimulationContext(sim_cfg)
    cfg = RigidObjectCfg(
        prim_path="/World/Body",
        spawn=sim_utils.CuboidCfg(
            size=(0.46, 0.34, 0.25),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    asset = RigidObject(cfg)
    sim.reset()

    view = IsaacArticulationView(asset)
    coeffs = resolve_coefficients(
        RobotHydroConfig.from_yaml(example_config_path("bluerov2_auv.yaml")))
    coeffs.center_of_buoyancy = torch.zeros_like(coeffs.center_of_buoyancy)
    hydro = UnderwaterHydrodynamics(view, coeffs)
    hydro.reset(current_world=torch.zeros(view.num_envs, 3, device=dev))

    # seed the initial surge velocity
    asset.write_root_velocity_to_sim(torch.tensor([[U0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=dev))

    u_sim = []
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).expand(view.num_envs, 4)
    for k in range(steps):
        w_body = hydro.compute_wrench(dt)
        quat = asset.data.body_quat_w
        R = quat_to_rotation_matrix(quat)
        f_world = (R @ w_body[..., 0:3].unsqueeze(-1)).squeeze(-1)
        m_world = (R @ w_body[..., 3:6].unsqueeze(-1)).squeeze(-1)
        view.set_external_wrench(torch.cat([f_world, m_world], dim=-1))
        sim.step()
        asset.update(dt)
        # pin attitude (isolate translation; preserve linear velocity)
        lin_w = asset.data.root_lin_vel_w
        asset.write_root_pose_to_sim(torch.cat([asset.data.root_pos_w, ident], dim=-1))
        asset.write_root_velocity_to_sim(torch.cat([lin_w, torch.zeros(view.num_envs, 3, device=dev)], dim=-1))
        if (k + 1) % SAMPLE_EVERY == 0:
            us = float(view.body_states()[2][0, 0, 0])
            u_sim.append(us)
            print(f"PROGRESS:: t={dt * (k + 1):.2f}s  u_sim={us:.4f}  u_ref={u_ref[len(u_sim) - 1]:.4f}",
                  flush=True)

    max_rel = max(abs(s - r) / U0 for s, r in zip(u_sim, u_ref, strict=True))
    return {"u_sim": u_sim, "u_ref": u_ref, "max_rel_error": max_rel}


if __name__ == "__main__":
    try:
        result = run()
        ok = result["max_rel_error"] < 0.05
        print(f"RESULT:: max_rel_error={result['max_rel_error']:.4f}  "
              f"u_sim={[f'{x:.3f}' for x in result['u_sim']]}  "
              f"u_ref={[f'{x:.3f}' for x in result['u_ref']]}  {'PASS' if ok else 'FAIL'}", flush=True)
    except Exception as e:
        import traceback
        print("RUN_ERROR:", repr(e), flush=True)
        traceback.print_exc()
    finally:
        os._exit(0)
