# sim_validation — in-sim validation scenarios

Standalone scripts that run a lighthill-driven body **inside Isaac Sim** and compare the
result to the Plan A analytical CPU reference (`lighthill.validation.reference`). They
certify the *glue* — Isaac state read, world↔body frame conversion, wrench application,
inertia augmentation — independently of the physics, which the CPU reference certifies.

These need a real Isaac Sim + GPU; they are **not** part of the CPU/CI gate. Run them
directly in the Isaac environment, or via the marked pytest gate:

```bash
# direct
OMNI_KIT_ACCEPT_EULA=YES python sim_validation/drag_terminal.py
# pytest gate (each scenario runs as its own subprocess; serial to avoid GPU contention)
LIGHTHILL_REAL_SIM_OK=1 uv run pytest tests/test_apply_isaac_marker.py -p no:xdist
```

Each script exposes `run(...) -> dict` and prints a `RESULT:: ... PASS|FAIL` line.

## Scenarios & latest results

| Script | Validates | Metric | Result | Tol |
|---|---|---|---|---|
| `drag_terminal.py` | translational drag, steady state | surge terminal velocity vs CPU ref | 0.41% | < 5% |
| `free_decay.py` | translational drag, transient | surge-decay trajectory vs CPU ref | 0.02% | < 5% |
| `restoring.py` | buoyant restoring couple + roll drag | roll(t) vs CPU ref | 0.07° | < 3° |
| `arm_swing_reaction.py` | UVMS vehicle↔arm coupling (the gate) | base pitch reaction vs Featherstone ref | 7.7% | < 15% |

## Harness conventions (why each scenario is set up as it is)

- **Primitive stand-ins.** The Isaac body is a cuboid; lighthill's coefficients come from
  the YAML config, not the Isaac geometry. Geometry only carries mass/inertia.
- **Gravity ON, neutral body.** PhysX weight cancels lighthill buoyancy (the config is
  designed neutral), so there is no spurious heave. `enable_external_forces_every_iteration`
  is enabled so the applied buoyancy wrench is integrated as accurately as gravity.
- **DOF isolation.** A free thrust-driven slender body is Munk-unstable (it tumbles), so the
  translational scenarios (`drag_terminal`, `free_decay`) **pin attitude** and zero the CoB
  to measure the drag law cleanly. `restoring` does the opposite — it leaves rotation **free**
  (the dynamics under test) and pins only position.
- **Inertia parity.** `restoring`'s oscillation frequency depends on rotational inertia, so it
  reads the body's actual inertia from Isaac and feeds it to the CPU reference for a fair
  comparison.
- **The coupling gate (`arm_swing_reaction`) isolates the inertial coupling.** It is the only
  multi-body scenario (a free base + one revolute-jointed arm, vs the Featherstone reference in
  `reference_featherstone.md`). It feeds the reference the sim's *actual* realized `q(t)`,
  masses/inertias, and joint geometry (USD `localPos` is scaled by body-scale — verified by a
  `GEOMCHECK` against the measured arm offset), runs gravity/buoyancy/**drag** off to isolate the
  added-mass + rigid coupling, and gates on the base **pitch** reaction (the high-SNR signal).

See `docs/isaac-api-findings.md` for the pinned Isaac API and `docs/paper-notes.md` for the
findings these scenarios surfaced (the Munk instability, the force-application gotcha).
