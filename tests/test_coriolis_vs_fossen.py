"""Independent verification: lighthill's added-mass Coriolis kernel vs Fossen's `m2c`.

Every OTHER lighthill added-mass check (unit tests, reference_coupled) shares the same
kernels, so a sign/convention error would cancel and stay hidden. Fossen's `m2c` is an
external, authoritative construction (Handbook of Marine Craft, 2021, Ch. 3), so this is a
genuinely independent oracle. The added-mass Coriolis FORCE in the EOM is ``-C_A(nu) @ nu``;
lighthill's ``added_mass_coriolis`` returns exactly that, so we compare it to
``-(m2c(M_A, nu) @ nu)``.

The Fossen reference is transcribed verbatim in form (from the MSS / PythonVehicleSimulator
``lib/gnc.py``) but expressed in torch to match lighthill's stack (numpy is not a dependency).
It imports NO lighthill code, so it is a genuinely independent oracle.
"""

from __future__ import annotations

import torch

from lighthill.forces import added_mass_coriolis


def _smtrx(a: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrix S(a) with a x b = S(a) b (Fossen's Smtrx)."""
    z = torch.zeros((), dtype=a.dtype)
    return torch.stack([
        torch.stack([z, -a[2], a[1]]),
        torch.stack([a[2], z, -a[0]]),
        torch.stack([-a[1], a[0], z]),
    ])


def _m2c(M: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Fossen's Coriolis-centripetal matrix C from mass matrix M and velocity nu (6-DOF)."""
    M = 0.5 * (M + M.T)
    M11, M12 = M[0:3, 0:3], M[0:3, 3:6]
    M21, M22 = M12.T, M[3:6, 3:6]
    nu1, nu2 = nu[0:3], nu[3:6]
    dt_dnu1 = M11 @ nu1 + M12 @ nu2
    dt_dnu2 = M21 @ nu1 + M22 @ nu2
    C = torch.zeros((6, 6), dtype=M.dtype)
    C[0:3, 3:6] = -_smtrx(dt_dnu1)
    C[3:6, 0:3] = -_smtrx(dt_dnu1)
    C[3:6, 3:6] = -_smtrx(dt_dnu2)
    return C


def _random_added_mass(gen: torch.Generator) -> torch.Tensor:
    """A physical (symmetric positive-definite) 6x6 added-mass matrix."""
    A = torch.randn(6, 6, dtype=torch.float64, generator=gen)
    return A @ A.T + 6.0 * torch.eye(6, dtype=torch.float64)


def test_coriolis_matches_fossen_m2c_random():
    gen = torch.Generator().manual_seed(0)
    worst = 0.0
    for _ in range(200):
        M_A = _random_added_mass(gen)
        nu = torch.randn(6, dtype=torch.float64, generator=gen)
        f_fossen = -(_m2c(M_A, nu) @ nu)                       # Fossen reference force
        f_light = added_mass_coriolis(M_A, nu)                 # lighthill's -C_A @ nu
        rel = (f_light - f_fossen).norm() / f_fossen.norm().clamp_min(1e-12)
        worst = max(worst, float(rel))
    assert worst < 1e-10, f"lighthill Coriolis disagrees with Fossen m2c by {worst:.2e}"


def test_coriolis_matches_fossen_anisotropic_diagonal():
    # The exact anisotropic added mass used in the UVMS gate (bluerov base link).
    M_A = torch.diag(torch.tensor([6.36, 7.12, 18.68, 0.189, 0.135, 0.222], dtype=torch.float64))
    gen = torch.Generator().manual_seed(1)
    for _ in range(50):
        nu = torch.randn(6, dtype=torch.float64, generator=gen)
        f_fossen = -(_m2c(M_A, nu) @ nu)
        f_light = added_mass_coriolis(M_A, nu)
        assert torch.allclose(f_light, f_fossen, atol=1e-10)
