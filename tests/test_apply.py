import torch

from lighthill import example_config_path
from lighthill.apply import UnderwaterHydrodynamics
from lighthill.articulation import FakeArticulation
from lighthill.coefficients import resolve_coefficients
from lighthill.config import RobotHydroConfig


def _auv_coeffs():
    cfg = RobotHydroConfig.from_yaml(example_config_path("bluerov2_auv.yaml"))
    return resolve_coefficients(cfg)


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
