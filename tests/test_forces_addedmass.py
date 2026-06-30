import torch

from lighthill.forces import added_mass_coriolis, added_mass_residual


def test_coriolis_zero_velocity_is_zero():
    M = torch.diag(torch.tensor([10.0, 20.0, 30.0, 1.0, 2.0, 3.0]))
    w = added_mass_coriolis(M, torch.zeros(6))
    assert torch.allclose(w, torch.zeros(6), atol=1e-9)


def test_coriolis_pure_translation_gives_moment_only():
    # Diagonal added mass, surge+sway velocity -> coupling produces a yaw moment, no net force.
    M = torch.diag(torch.tensor([10.0, 20.0, 30.0, 0.0, 0.0, 0.0]))
    v = torch.zeros(6)
    v[0], v[1] = 1.0, 1.0  # u, v
    w = added_mass_coriolis(M, v)
    assert torch.allclose(w[:3], torch.zeros(3), atol=1e-6)  # no net force
    assert w[3:].abs().sum() > 1e-6                          # some moment


def test_coriolis_is_power_neutral():
    # Coriolis/centripetal forces do no work: w · v == 0 for any state.
    torch.manual_seed(0)
    M = torch.rand(6, 6)
    M = M + M.T  # symmetric
    v = torch.randn(6)
    w = added_mass_coriolis(M, v)
    assert torch.isclose(torch.dot(w, v), torch.tensor(0.0), atol=1e-4)


def test_residual_is_minus_matrix_times_accel():
    M_off = torch.zeros(6, 6)
    M_off[0, 4] = 2.0
    M_off[4, 0] = 2.0
    a = torch.zeros(6)
    a[4] = 3.0
    w = added_mass_residual(M_off, a)
    assert torch.isclose(w[0], torch.tensor(-6.0), atol=1e-6)
