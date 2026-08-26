# lighthill

[![CI](https://github.com/jeffrichley/lighthill/actions/workflows/ci.yml/badge.svg)](https://github.com/jeffrichley/lighthill/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jeffrichley/lighthill/branch/main/graph/badge.svg)](https://codecov.io/gh/jeffrichley/lighthill)
[![PyPI](https://img.shields.io/pypi/v/lighthill.svg)](https://pypi.org/project/lighthill/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![Renovate enabled](https://img.shields.io/badge/renovate-enabled-brightgreen.svg)](https://renovatebot.com)
[![built by agent beings](https://img.shields.io/badge/built%20by-agent%20beings%20%F0%9F%AA%B6-8A2BE2.svg)](#)

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

## Design

The force model, architecture, validation gates, and scope are specified in
[`docs/design/2026-06-28-hydrodynamics-design.md`](docs/design/2026-06-28-hydrodynamics-design.md).
The implementation plan is generated from that spec.

## Status

| | |
|---|---|
| Stage | Pre-alpha (name reservation + scaffold) |
| Python | ≥ 3.11 |
| License | MIT |

## Name

Named for **Sir James Lighthill**, whose *elongated-body theory* of aquatic locomotion
is the foundational hydrodynamic model of how slender, articulated bodies generate thrust
through reactive (added-mass) forces — exactly the physics this library computes per link.

## License

[MIT](LICENSE) © 2026 Jeff Richley
