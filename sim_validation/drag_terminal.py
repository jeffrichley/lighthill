"""In-sim validation 2: drag -> terminal velocity, Isaac glue vs Plan A CPU reference.

Spawns one primitive cuboid (a BlueROV2 stand-in; lighthill's coefficients come from
``bluerov2_auv.yaml``, not the Isaac geometry), drives it with a constant surge thrust
plus the lighthill hydro wrench, and measures the terminal surge speed. The same body +
thrust is run through the Plan A CPU integrator; the two terminal speeds must agree.

Gravity is ON in both paths so PhysX weight cancels lighthill buoyancy for the ~neutral
body. The center of buoyancy is zeroed and the body's attitude is pinned to identity each
step, isolating the translational surge drag law from the Munk directional instability
(a free thrust-driven slender body tumbles; directional dynamics are validated elsewhere).

Run directly (in the Isaac env):
    OMNI_KIT_ACCEPT_EULA=YES python sim_validation/drag_terminal.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

THRUST = 141.0   # N, body-frame surge; with BlueROV2 damping -> terminal ~0.95 m/s
# Neutrally buoyant against volume 0.0134 m^3: mass = rho*g*V/g = rho*V = 1025*0.0134.
# Gravity is ON in both paths so PhysX weight cancels lighthill buoyancy -> ~zero heave,
# so the surge stays decoupled and the body does NOT pitch (no Munk moment from heave).
MASS = 13.73     # kg, ~neutral cuboid rigid mass


def _cpu_reference(steps: int, dt: float) -> float:
    """Terminal surge speed from the Plan A CPU integrator for the same body + thrust."""
    import torch

    from lighthill import RobotHydroConfig, example_config_path, resolve_coefficients
    from lighthill.validation.reference import Body, simulate

    coeffs = resolve_coefficients(
        RobotHydroConfig.from_yaml(example_config_path("bluerov2_auv.yaml"))
    )
    body = Body(
        mass=MASS,
        inertia=(0.26, 0.23, 0.37),
        volume=float(coeffs.volume[0]),
        cob=(0.0, 0.0, 0.0),  # isolate surge drag: no metacentric pitch pendulum (see run())
        added_mass=coeffs.added_mass[0],
        linear_damping=coeffs.linear_damping[0],
        quadratic_damping=coeffs.quadratic_damping[0],
        density=coeffs.density,
    )
    f_ext = torch.zeros(6)
    f_ext[0] = THRUST
    # gravity ON (default): weight cancels buoyancy for the ~neutral body, same as the sim.
    traj = simulate(body, steps=steps, dt=dt, external_force_body=f_ext)
    return float(traj["twist"][-1, 0])


def run(steps: int = 1200, dt: float = 0.005) -> dict:
    """Run the in-sim drag-terminal scenario; return {u_sim, u_ref, rel_error}."""
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app  # noqa: F841 — held to keep Isaac Sim alive for the run

    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationContext

    from lighthill import RobotHydroConfig, example_config_path, resolve_coefficients
    from lighthill.apply import UnderwaterHydrodynamics
    from lighthill.apply_isaac import IsaacArticulationView
    from lighthill.frames import quat_to_rotation_matrix

    u_ref = _cpu_reference(steps, dt)

    sim_cfg = sim_utils.SimulationCfg(dt=dt, device="cuda:0", gravity=(0.0, 0.0, -9.81))
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
        RobotHydroConfig.from_yaml(example_config_path("bluerov2_auv.yaml"))
    )
    # Isolate the surge drag law: zero the center of buoyancy so there is no
    # metacentric pitch pendulum (the cob offset + Munk coupling otherwise drives a
    # pitch oscillation). Restoring dynamics are validated separately (restoring.py).
    coeffs.center_of_buoyancy = torch.zeros_like(coeffs.center_of_buoyancy)
    hydro = UnderwaterHydrodynamics(view, coeffs)
    hydro.reset(current_world=torch.zeros(view.num_envs, 3, device="cuda:0"))

    for k in range(steps):
        w_body = hydro.compute_wrench(dt)          # [E,B,6] body frame
        w_body[..., 0] += THRUST                    # add surge thrust (body frame)
        quat = asset.data.body_quat_w               # direct read (no vel conversion needed)
        R = quat_to_rotation_matrix(quat)           # [E,B,3,3] body->world
        f_world = (R @ w_body[..., 0:3].unsqueeze(-1)).squeeze(-1)
        m_world = (R @ w_body[..., 3:6].unsqueeze(-1)).squeeze(-1)
        view.set_external_wrench(torch.cat([f_world, m_world], dim=-1))
        sim.step()
        asset.update(dt)
        # Pin attitude: a free thrust-driven slender body is Munk-unstable (it tumbles).
        # This scenario validates the translational DRAG law, so hold orientation at
        # identity and zero angular velocity, preserving position + linear velocity.
        # (Directional/restoring dynamics are validated separately in restoring.py.)
        E = view.num_envs
        ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda:0").expand(E, 4)
        pos_w = asset.data.root_pos_w
        lin_w = asset.data.root_lin_vel_w
        asset.write_root_pose_to_sim(torch.cat([pos_w, ident], dim=-1))
        asset.write_root_velocity_to_sim(torch.cat([lin_w, torch.zeros(E, 3, device="cuda:0")], dim=-1))
        if (k + 1) % 200 == 0:
            d = asset.data
            wp = d.body_pos_w[0, 0].tolist()
            wv = d.body_lin_vel_w[0, 0].tolist()
            wq = d.body_quat_w[0, 0].tolist()
            bs = float(view.body_states()[2][0, 0, 0])
            print(f"PROGRESS:: step {k + 1} pos=[{wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f}] "
                  f"wvel=[{wv[0]:.3f},{wv[1]:.3f},{wv[2]:.3f}] bsurge={bs:.3f} "
                  f"quat=[{wq[0]:.3f},{wq[1]:.3f},{wq[2]:.3f},{wq[3]:.3f}]", flush=True)

    u_sim = float(view.body_states()[2][0, 0, 0])  # terminal body-frame surge speed
    rel_error = abs(u_sim - u_ref) / abs(u_ref)
    # NOTE: do not simulation_app.close() here — Kit hangs on teardown; __main__ os._exit(0)s.
    return {"u_sim": u_sim, "u_ref": u_ref, "rel_error": rel_error}


if __name__ == "__main__":
    try:
        result = run()
        ok = result["rel_error"] < 0.05
        print(
            f"RESULT:: u_sim={result['u_sim']:.5f}  u_ref={result['u_ref']:.5f}  "
            f"rel_error={result['rel_error']:.4f}  {'PASS' if ok else 'FAIL'}",
            flush=True,
        )
    except Exception as e:
        import traceback
        print("RUN_ERROR:", repr(e), flush=True)
        traceback.print_exc()
    finally:
        # Isaac/Kit hangs on a clean close(); force exit so the shell returns.
        os._exit(0)
