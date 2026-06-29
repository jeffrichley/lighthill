from importlib.resources import files

import torch

from lighthill.coefficients import resolve_coefficients
from lighthill.config import RobotHydroConfig


def _path(name):
    return files("lighthill.configs").joinpath(name)


def test_auv_config_loads_and_resolves_to_one_link():
    cfg = RobotHydroConfig.from_yaml(_path("bluerov2_auv.yaml"))
    rc = resolve_coefficients(cfg)
    assert len(cfg.links) == 1
    assert rc.added_mass.shape == (1, 6, 6)
    assert (rc.volume > 0).all()


def test_uvms_config_has_vehicle_plus_arm_links():
    cfg = RobotHydroConfig.from_yaml(_path("bluerov2_alpha_uvms.yaml"))
    rc = resolve_coefficients(cfg)
    assert len(cfg.links) >= 6  # vehicle + 5 arm links
    assert rc.added_mass.shape[0] == len(cfg.links)
    # arm links are cylinder-modeled -> finite transverse added mass
    assert torch.isfinite(rc.added_mass).all()
