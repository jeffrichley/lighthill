import torch
from lighthill.articulation import FakeArticulation


def test_fake_round_trips_state_and_records_wrenches():
    art = FakeArticulation(num_envs=2, num_bodies=3)
    pos, quat, vel = art.body_states()
    assert pos.shape == (2, 3, 3)
    assert quat.shape == (2, 3, 4)
    assert vel.shape == (2, 3, 6)
    w = torch.zeros(2, 3, 6)
    w[..., 2] = 1.0
    art.set_external_wrench(w)
    assert torch.allclose(art.last_wrench, w)


def test_fake_lets_tests_set_state():
    art = FakeArticulation(num_envs=1, num_bodies=1)
    art.set_body_velocity(torch.tensor([[[1.0, 0, 0, 0, 0, 0]]]))
    _, _, vel = art.body_states()
    assert vel[0, 0, 0] == 1.0
