import torch

from lighthill.articulation import FakeArticulation


def test_fake_round_trips_state_and_records_wrenches():
    art = FakeArticulation(num_envs=2, num_bodies=3)
    pos, quat, vel = art.body_states()
    assert pos.shape == (2, 3, 3)
    assert quat.shape == (2, 3, 4)
    assert vel.shape == (2, 3, 6)
    assert torch.allclose(quat, torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(2, 3, 4))
    w = torch.zeros(2, 3, 6)
    w[..., 2] = 1.0
    art.set_external_wrench(w)
    assert torch.allclose(art.last_wrench, w)


def test_fake_lets_tests_set_state():
    art = FakeArticulation(num_envs=1, num_bodies=1)
    art.set_body_velocity(torch.tensor([[[1.0, 0, 0, 0, 0, 0]]]))
    _, _, vel = art.body_states()
    assert torch.isclose(vel[0, 0, 0], torch.tensor(1.0))


def test_fake_lets_tests_set_quat():
    art = FakeArticulation(num_envs=1, num_bodies=1)
    quat_set = torch.tensor([[[0.7071, 0.7071, 0.0, 0.0]]])
    art.set_body_quat(quat_set)
    _, quat, _ = art.body_states()
    assert torch.allclose(quat, quat_set)


def test_fake_set_body_inertias_round_trips():
    art = FakeArticulation(num_envs=2, num_bodies=1)
    mass = torch.tensor([[2.0], [3.0]])
    inertia_diag = torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    art.set_body_inertias(mass, inertia_diag)
    assert torch.allclose(art.mass, mass)
    assert torch.allclose(art.inertia_diag, inertia_diag)
