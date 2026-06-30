import torch

from lighthill.inertia import effective_inertia, split_added_mass


def test_isotropic_linear_added_mass_goes_entirely_to_mass_bump():
    M = torch.diag(torch.tensor([5.0, 5.0, 5.0, 0.2, 0.3, 0.4])).unsqueeze(0)
    r = split_added_mass(M)
    assert torch.isclose(r.mass_bump[0], torch.tensor(5.0))
    assert torch.allclose(r.inertia_bump[0], torch.tensor([0.2, 0.3, 0.4]))
    # nothing left on the linear diagonal of the residual
    assert torch.allclose(torch.diagonal(r.residual[0])[:3], torch.zeros(3), atol=1e-6)


def test_anisotropic_linear_remainder_goes_to_residual():
    M = torch.diag(torch.tensor([6.0, 7.0, 18.0, 0.1, 0.1, 0.1])).unsqueeze(0)
    r = split_added_mass(M)
    assert torch.isclose(r.mass_bump[0], torch.tensor(6.0))  # min of 6,7,18
    diag_res = torch.diagonal(r.residual[0])[:3]
    assert torch.allclose(diag_res, torch.tensor([0.0, 1.0, 12.0]), atol=1e-6)


def test_off_diagonal_preserved_in_residual():
    M = torch.zeros(1, 6, 6)
    M[0, 0, 4] = 2.0
    M[0, 4, 0] = 2.0
    M[0, range(6), range(6)] = torch.tensor([3.0, 3.0, 3.0, 0.0, 0.0, 0.0])
    r = split_added_mass(M)
    assert r.residual[0, 0, 4] == 2.0


def test_effective_inertia_adds_bumps():
    M = torch.diag(torch.tensor([5.0, 5.0, 5.0, 0.2, 0.2, 0.2])).unsqueeze(0)
    r = split_added_mass(M)
    mass = torch.tensor([10.0])
    inertia = torch.tensor([[1.0, 1.0, 1.0]])
    m_eff, i_eff = effective_inertia(mass, inertia, r)
    assert torch.isclose(m_eff[0], torch.tensor(15.0))
    assert torch.allclose(i_eff[0], torch.tensor([1.2, 1.2, 1.2]))
