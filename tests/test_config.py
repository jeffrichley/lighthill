import textwrap

import pytest

from lighthill.config import ConfigError, RobotHydroConfig


def _write(tmp_path, body):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(body))
    return p


VALID = """
    density: 1025.0
    links:
      - name: base
        volume: 0.0134
        center_of_buoyancy: [0.0, 0.0, 0.02]
        neutrally_buoyant: false
        added_mass: {kind: matrix, matrix: [6.4, 7.1, 18.0, 0.2, 0.2, 0.2]}
        linear_damping: [4.0, 6.2, 5.2, 0.07, 0.07, 0.07]
        quadratic_damping: [18.0, 21.0, 36.0, 1.5, 1.5, 1.5]
"""


def test_loads_valid_config(tmp_path):
    cfg = RobotHydroConfig.from_yaml(_write(tmp_path, VALID))
    assert cfg.density == 1025.0
    assert len(cfg.links) == 1
    link = cfg.links[0]
    assert link.name == "base"
    assert link.added_mass.kind == "matrix"
    assert len(link.linear_damping) == 6


def test_rejects_wrong_length_damping(tmp_path):
    bad = VALID.replace("linear_damping: [4.0, 6.2, 5.2, 0.07, 0.07, 0.07]",
                        "linear_damping: [4.0, 6.2, 5.2]")
    with pytest.raises(ConfigError, match="linear_damping"):
        RobotHydroConfig.from_yaml(_write(tmp_path, bad))


def test_rejects_asymmetric_full_matrix(tmp_path):
    asym = [0.0] * 36
    asym[1] = 5.0  # M[0,1]=5 but M[1,0]=0 -> asymmetric
    bad = VALID.replace(
        "added_mass: {kind: matrix, matrix: [6.4, 7.1, 18.0, 0.2, 0.2, 0.2]}",
        f"added_mass: {{kind: matrix, matrix: {asym}}}",
    )
    with pytest.raises(ConfigError, match="symmetric"):
        RobotHydroConfig.from_yaml(_write(tmp_path, bad))


def test_rejects_negative_volume(tmp_path):
    bad = VALID.replace("volume: 0.0134", "volume: -0.1")
    with pytest.raises(ConfigError, match="volume"):
        RobotHydroConfig.from_yaml(_write(tmp_path, bad))


def test_cylinder_added_mass_requires_radius_and_length(tmp_path):
    bad = VALID.replace(
        "added_mass: {kind: matrix, matrix: [6.4, 7.1, 18.0, 0.2, 0.2, 0.2]}",
        "added_mass: {kind: cylinder, radius: 0.025}",  # missing length/axis
    )
    with pytest.raises(ConfigError, match="cylinder"):
        RobotHydroConfig.from_yaml(_write(tmp_path, bad))
