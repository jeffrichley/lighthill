import torch

from lighthill import example_config_path
from lighthill.apply import UnderwaterHydrodynamics
from lighthill.articulation import FakeArticulation
from lighthill.coefficients import resolve_coefficients
from lighthill.config import AddedMassSpec, LiftSpec, LinkConfig, RobotHydroConfig
from lighthill.forces import lift_wrench


def _auv_coeffs():
    cfg = RobotHydroConfig.from_yaml(example_config_path("bluerov2_auv.yaml"))
    return resolve_coefficients(cfg)


def _bare_link(**kw):
    """A link with every hydro term off, so compute_wrench isolates whatever is enabled."""
    base = dict(
        name="foil", volume=0.0, center_of_buoyancy=(0.0, 0.0, 0.0),
        added_mass=AddedMassSpec(kind="matrix", matrix=(0.0,) * 6),
        linear_damping=(0.0,) * 6, quadratic_damping=(0.0,) * 6,
    )
    base.update(kw)
    return LinkConfig(**base)


def test_inertias_are_augmented_at_init():
    coeffs = _auv_coeffs()
    art = FakeArticulation(num_envs=2, num_bodies=1)
    base_mass = art.mass.clone()
    UnderwaterHydrodynamics(art, coeffs)
    assert (art.mass > base_mass).all()  # mass bumped by isotropic added mass


def test_buoyant_body_gets_upward_world_wrench():
    coeffs = _auv_coeffs()
    art = FakeArticulation(num_envs=1, num_bodies=1)
    hydro = UnderwaterHydrodynamics(art, coeffs)
    hydro.reset(current_world=torch.zeros(1, 3))
    hydro.apply(dt=0.01)
    # positive-buoyancy base_link -> +Z world force recorded on the body
    assert art.last_wrench[0, 0, 2] > 0


def test_drag_opposes_forward_motion():
    coeffs = _auv_coeffs()
    art = FakeArticulation(num_envs=1, num_bodies=1)
    art.set_body_velocity(torch.tensor([[[2.0, 0, 0, 0, 0, 0]]]))  # surge +x
    hydro = UnderwaterHydrodynamics(art, coeffs)
    hydro.reset(current_world=torch.zeros(1, 3))
    w = hydro.compute_wrench(dt=0.01)
    assert w[0, 0, 0] < 0  # body-frame surge drag opposes motion


def test_wrench_shape_matches_bodies():
    cfg = RobotHydroConfig.from_yaml(example_config_path("bluerov2_alpha_uvms.yaml"))
    coeffs = resolve_coefficients(cfg)
    nb = len(cfg.links)
    art = FakeArticulation(num_envs=4, num_bodies=nb)
    hydro = UnderwaterHydrodynamics(art, coeffs)
    hydro.reset()
    w = hydro.compute_wrench(dt=0.01)
    assert w.shape == (4, nb, 6)


def test_lift_contributes_to_wrench():
    # foil at angle of attack, every other term zeroed -> the wrench IS the lift wrench.
    link = _bare_link(lift=LiftSpec(semi_axes=(0.5, 0.1, 0.1), c_kutta=1.0, c_magnus=1.0))
    coeffs = resolve_coefficients(RobotHydroConfig(links=(link,)))
    art = FakeArticulation(num_envs=1, num_bodies=1)
    art.set_body_velocity(torch.tensor([[[1.0, 0.3, 0.0, 0.0, 0.0, 0.5]]]))  # AoA + yaw spin
    hydro = UnderwaterHydrodynamics(art, coeffs)
    hydro.reset(current_world=torch.zeros(1, 3))
    w = hydro.compute_wrench(dt=0.01)
    v_rel = torch.tensor([1.0, 0.3, 0.0, 0.0, 0.0, 0.5])
    expected = lift_wrench(v_rel, torch.tensor([0.5, 0.1, 0.1]),
                           torch.tensor(1.0), torch.tensor(1.0), coeffs.density)
    assert w[0, 0, 0:3].norm() > 1.0  # non-trivial lift
    assert torch.allclose(w[0, 0], expected, rtol=1e-4, atol=1e-2)


def test_no_lift_by_default():
    # a link without a lift spec (and all else zeroed) produces no force at all.
    coeffs = resolve_coefficients(RobotHydroConfig(links=(_bare_link(),)))
    art = FakeArticulation(num_envs=1, num_bodies=1)
    art.set_body_velocity(torch.tensor([[[1.0, 0.3, 0.0, 0.0, 0.0, 0.5]]]))
    hydro = UnderwaterHydrodynamics(art, coeffs)
    hydro.reset(current_world=torch.zeros(1, 3))
    w = hydro.compute_wrench(dt=0.01)
    assert torch.allclose(w[0, 0], torch.zeros(6), atol=1e-6)


def test_apply_converts_body_wrench_to_world_frame():
    # 90 deg roll about body +x: quat = (w=0.707, x=0.707, y=0, z=0).
    # buoyancy_wrench computes body-frame buoyancy by rotating world-up [0,0,F] via R^T,
    # giving f_body = [0, F, 0].  apply() then applies R: R @ R^T @ [0,0,F] = [0,0,F].
    # Buoyancy always comes out world +z regardless of body orientation — the two
    # rotations cancel, so assert w[2] > 0.
    #
    # This test catches the wrong-rotation bug: if apply() used R^T instead of R,
    # the chain would be R^T @ R^T @ [0,0,F] = (R_x(-90))^2 @ [0,0,F] = [0,0,-F],
    # so w[2] would be negative and the assert below would fail.
    coeffs = _auv_coeffs()
    art = FakeArticulation(num_envs=1, num_bodies=1)
    art.set_body_quat(torch.tensor([[[0.70710678, 0.70710678, 0.0, 0.0]]]))  # 90 deg roll, wxyz
    hydro = UnderwaterHydrodynamics(art, coeffs)
    hydro.reset(current_world=torch.zeros(1, 3))
    hydro.apply(dt=0.01)
    w = art.last_wrench[0, 0]
    assert w[2] > 0                     # world +z carries buoyancy (correct body->world: R, not R.T)
    assert abs(w[0]) < 1e-3            # world +x is ~zero (no spurious x-component)
