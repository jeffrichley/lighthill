"""lighthill — per-link hydrodynamics for articulated underwater robots in Isaac Lab.

GPU-vectorized buoyancy, drag, added-mass, and current forces applied *per link*
across an articulated robot (vehicle + arm + multi-arm), so the vehicle-manipulator
coupling that single-rigid-body underwater simulators miss is modeled directly.

Status: pre-alpha. The package name is reserved; the physics engine is in active
development. Named for Sir James Lighthill, whose elongated-body theory of aquatic
locomotion underpins the reactive added-mass forces this library computes.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from .coefficients import ResolvedCoefficients, resolve_coefficients
from .config import AddedMassSpec, ConfigError, LinkConfig, RobotHydroConfig
from .current import CurrentField, relative_velocity
from .forces import (
    added_mass_coriolis,
    added_mass_residual,
    buoyancy_wrench,
    drag_wrench,
)

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "RobotHydroConfig", "LinkConfig", "AddedMassSpec", "ConfigError",
    "resolve_coefficients", "ResolvedCoefficients",
    "buoyancy_wrench", "drag_wrench", "added_mass_coriolis", "added_mass_residual",
    "CurrentField", "relative_velocity", "example_config_path",
]


def example_config_path(name: str) -> Path:
    """Absolute path to a shipped example config under lighthill/configs/."""
    return Path(str(files("lighthill.configs").joinpath(name)))
