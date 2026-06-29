import torch

from lighthill.current import CurrentField, relative_velocity

IDQUAT = torch.tensor([1.0, 0.0, 0.0, 0.0])


def test_zero_current_leaves_velocity_unchanged():
    v = torch.tensor([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    out = relative_velocity(v, IDQUAT, torch.zeros(3))
    assert torch.allclose(out, v, atol=1e-6)


def test_current_subtracts_from_linear_only():
    v = torch.tensor([1.0, 0.0, 0.0, 0.5, 0.0, 0.0])
    cur = torch.tensor([0.4, 0.0, 0.0])  # 0.4 m/s along world +x; upright body
    out = relative_velocity(v, IDQUAT, cur)
    assert torch.isclose(out[0], torch.tensor(0.6), atol=1e-6)  # 1.0 - 0.4
    assert torch.allclose(out[3:], v[3:], atol=1e-6)            # angular untouched


def test_sample_magnitude_within_bounds():
    g = torch.Generator().manual_seed(0)
    field = CurrentField(max_speed=0.5)
    cur = field.sample(1000, generator=g)
    speeds = cur.norm(dim=-1)
    assert cur.shape == (1000, 3)
    assert speeds.max().item() <= 0.5 + 1e-6
    assert speeds.min().item() >= 0.0


def test_perturb_is_noop_when_std_zero():
    field = CurrentField(max_speed=0.5, noise_std=0.0)
    cur = torch.ones(4, 3)
    assert torch.allclose(field.perturb(cur), cur)
