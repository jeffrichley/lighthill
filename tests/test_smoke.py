"""Smoke test — proves the package imports and exposes its version.

Keeps the coverage gate satisfiable while the physics engine is still a
scaffold. Real tests land alongside the per-link force model.
"""

import lighthill


def test_package_imports_and_exposes_version() -> None:
    assert isinstance(lighthill.__version__, str)
    assert lighthill.__version__.count(".") >= 2  # semver-ish: major.minor.patch
