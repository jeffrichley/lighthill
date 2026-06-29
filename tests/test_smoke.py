import torch

import lighthill


def test_public_api_is_importable():
    for sym in [
        "RobotHydroConfig", "resolve_coefficients", "buoyancy_wrench",
        "drag_wrench", "added_mass_coriolis", "CurrentField", "relative_velocity",
    ]:
        assert hasattr(lighthill, sym), sym


def test_end_to_end_auv_wrench_from_shipped_config():
    cfg = lighthill.RobotHydroConfig.from_yaml(lighthill.example_config_path("bluerov2_auv.yaml"))
    rc = lighthill.resolve_coefficients(cfg)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    w = lighthill.buoyancy_wrench(quat, rc.volume, rc.center_of_buoyancy,
                                  rc.neutrally_buoyant, rc.density)
    assert w.shape == (1, 6)
    assert w[0, 2] > 0  # buoyancy points world-up
