# lighthill

**GPU-vectorized per-link hydrodynamics for articulated underwater robots in [NVIDIA Isaac Lab](https://developer.nvidia.com/isaac/lab).**

> ⚠️ **Pre-alpha.** The package name is reserved on PyPI; the physics engine is in
> active development. APIs are not yet stable.

## What it is

Out of the box, Isaac Lab simulates rigid bodies in air/vacuum. `lighthill` turns it
into an **underwater** simulator by applying, every physics step and **per link**:

- **buoyancy**
- **drag**
- **added mass**
- **currents**

The key difference from existing fast underwater simulators: forces are computed across
an **articulated** robot — vehicle **+** manipulator(s) — not a single rigid body. That
captures the vehicle↔arm hydrodynamic coupling that single-body simulators miss, while
staying GPU-vectorized across thousands of parallel environments for modern RL.

- **Topology-agnostic:** UV, UVMS, multi-arm, swimming-snake — configured, not hardcoded.
- **Config-driven:** declare links and coefficients; no per-robot force code.
- **Validation-first:** ships with a suite checked against standard analytical references.

It fills a real gap: fast underwater sims are single-body; multi-body underwater sims are
too slow for large-scale RL.

## Status

| | |
|---|---|
| Stage | Pre-alpha (name reservation + scaffold) |
| Python | ≥ 3.10 |
| License | MIT |

## Name

Named for **Sir James Lighthill**, whose *elongated-body theory* of aquatic locomotion
is the foundational hydrodynamic model of how slender, articulated bodies generate thrust
through reactive (added-mass) forces — exactly the physics this library computes per link.

## License

[MIT](LICENSE) © 2026 Jeff Richley
