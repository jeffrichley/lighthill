"""lighthill — per-link hydrodynamics for articulated underwater robots in Isaac Lab.

GPU-vectorized buoyancy, drag, added-mass, and current forces applied *per link*
across an articulated robot (vehicle + arm + multi-arm), so the vehicle-manipulator
coupling that single-rigid-body underwater simulators miss is modeled directly.

Status: pre-alpha. The package name is reserved; the physics engine is in active
development. Named for Sir James Lighthill, whose elongated-body theory of aquatic
locomotion underpins the reactive added-mass forces this library computes.
"""

__version__ = "0.0.1"
