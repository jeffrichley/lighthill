"""In-sim validation 3: buoyant restoring, Isaac glue vs Plan A CPU reference.

A neutrally-buoyant body with its center of buoyancy above the CoM is released at a roll
tilt; the restoring couple rocks it upright while roll drag damps it. The roll(t) trajectory
must match the Plan A CPU integrator. Unlike drag_terminal/free_decay this does NOT pin
attitude -- the rotation is exactly what is under test. Because oscillation frequency depends
on rotational inertia, the cuboid's actual inertia is read from Isaac and fed to the CPU
reference so the two are compared on equal footing.

Run (in the Isaac env):  OMNI_KIT_ACCEPT_EULA=YES python sim_validation/restoring.py
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

MASS = 13.73          # kg, ~neutral against volume 0.0134 m^3
COB_Z = 0.05          # m, center of buoyancy above CoM -> restoring couple
THETA0 = math.radians(30.0)   # initial roll
SAMPLE_EVERY = 150


def _roll_of(qw: float, qx: float) -> float:
    """Roll angle (rad) of a pure-x rotation quaternion (w,x,y,z)."""
    return 2.0 * math.atan2(qx, qw)


def _cpu_reference(steps: int, dt: float, mass: float, inertia: tuple[float, float, float]) -> list[float]:
    """Roll(t) samples from the Plan A CPU integrator for the same body + IC."""
    import torch

    from lighthill import RobotHydroConfig, example_config_path, resolve_coefficients
    from lighthill.validation.reference import Body, simulate

    coeffs = resolve_coefficients(
        RobotHydroConfig.from_yaml(example_config_path("bluerov2_auv.yaml")))
    body = Body(
        mass=mass, inertia=inertia, volume=float(coeffs.volume[0]),
        cob=(0.0, 0.0, COB_Z), added_mass=coeffs.added_mass[0],
        linear_damping=coeffs.linear_damping[0], quadratic_damping=coeffs.quadratic_damping[0],
        density=coeffs.density,
    )
    q0 = torch.tensor([math.cos(THETA0 / 2), math.sin(THETA0 / 2), 0.0, 0.0])
    traj = simulate(body, steps=steps, dt=dt, quat0=q0)
    q = traj["quat"]
    return [_roll_of(float(q[k, 0]), float(q[k, 1])) for k in range(SAMPLE_EVERY - 1, steps, SAMPLE_EVERY)]


def run(steps: int = 1200, dt: float = 0.005) -> dict:
    """Run the in-sim restoring scenario; return {roll_sim, roll_ref, max_abs_error_deg}."""
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

    dev = "cuda:0"
    sim_cfg = sim_utils.SimulationCfg(
        dt=dt, device=dev, gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True),
    )
    sim = SimulationContext(sim_cfg)
    tilt = (math.cos(THETA0 / 2), math.sin(THETA0 / 2), 0.0, 0.0)
    cfg = RigidObjectCfg(
        prim_path="/World/Body",
        spawn=sim_utils.CuboidCfg(
            size=(0.46, 0.34, 0.25),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=tilt),
    )
    asset = RigidObject(cfg)
    sim.reset()

    view = IsaacArticulationView(asset)
    # read the cuboid's actual base mass + principal inertia, feed the CPU reference
    base_mass = float(view.mass[0, 0])
    base_inertia = tuple(float(x) for x in view.inertia_diag[0, 0].tolist())
    roll_ref = _cpu_reference(steps, dt, base_mass, base_inertia)  # type: ignore[arg-type]

    coeffs = resolve_coefficients(
        RobotHydroConfig.from_yaml(example_config_path("bluerov2_auv.yaml")))
    coeffs.center_of_buoyancy = torch.tensor([[0.0, 0.0, COB_Z]], dtype=coeffs.center_of_buoyancy.dtype)
    hydro = UnderwaterHydrodynamics(view, coeffs)
    hydro.reset(current_world=torch.zeros(view.num_envs, 3, device=dev))

    origin = torch.zeros(view.num_envs, 3, device=dev)
    roll_sim = []
    for k in range(steps):
        hydro.apply(dt)                      # lighthill restoring couple + roll drag
        sim.step()
        asset.update(dt)
        # neutral body -> pure couple, ~no translation; pin position to keep it put,
        # leave rotation FREE (the dynamics under test).
        quat = asset.data.root_quat_w
        ang = asset.data.root_ang_vel_w
        asset.write_root_pose_to_sim(torch.cat([origin, quat], dim=-1))
        asset.write_root_velocity_to_sim(torch.cat([torch.zeros(view.num_envs, 3, device=dev), ang], dim=-1))
        if (k + 1) % SAMPLE_EVERY == 0:
            q = asset.data.body_quat_w[0, 0]
            rs = _roll_of(float(q[0]), float(q[1]))
            roll_sim.append(rs)
            rr = roll_ref[len(roll_sim) - 1]
            print(f"PROGRESS:: t={dt * (k + 1):.2f}s  roll_sim={math.degrees(rs):6.1f}  "
                  f"roll_ref={math.degrees(rr):6.1f} deg", flush=True)

    max_abs_deg = max(abs(math.degrees(s - r)) for s, r in zip(roll_sim, roll_ref, strict=True))
    return {"roll_sim": roll_sim, "roll_ref": roll_ref, "max_abs_error_deg": max_abs_deg}


if __name__ == "__main__":
    try:
        result = run()
        ok = result["max_abs_error_deg"] < 3.0  # within 3 deg over the whole trajectory
        print(f"RESULT:: max_abs_error_deg={result['max_abs_error_deg']:.2f}  "
              f"roll_sim={[f'{math.degrees(x):.1f}' for x in result['roll_sim']]}  "
              f"roll_ref={[f'{math.degrees(x):.1f}' for x in result['roll_ref']]}  "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    except Exception as e:
        import traceback
        print("RUN_ERROR:", repr(e), flush=True)
        traceback.print_exc()
    finally:
        os._exit(0)
