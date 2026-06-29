# Justfile — single quality-command surface for lighthill.
#
# Source of truth for local + CI check commands so developers and agents
# run the same gates in the same order. Mirrors the foreman justfile,
# trimmed to a single-package library.

set windows-shell := ["cmd.exe", "/c"]

default:
    @just --list

# Composite gate (run before push; CI runs the same).
check: lock-check lint typecheck test

# Developer convenience: apply lint auto-fixes + formatter.
fix:
    uv run --no-sync ruff check --fix .
    uv run --no-sync ruff format .

# Validate uv.lock parses cleanly and is in sync with pyproject.toml.
lock-check:
    uv lock --check

# Lint.
lint:
    uv run --no-sync ruff check .

# Type-check.
typecheck:
    uv run --no-sync mypy src

# Tests (coverage + xdist config lives in pyproject [tool.pytest.ini_options]).
test:
    uv run --no-sync pytest

# Dead-code scan (not part of the gate; run on demand).
dead-code:
    uv run --no-sync vulture src

# Build sdist + wheel locally (sanity-check that release.yml would succeed).
build:
    uv build --out-dir dist/
