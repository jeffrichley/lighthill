import pytest
import torch

from lighthill.accel import AccelerationFilter


def test_first_update_returns_zero():
    f = AccelerationFilter(shape=(2, 3), alpha=0.1)
    a = f.update(torch.ones(2, 3, 6), dt=0.01)
    assert torch.allclose(a, torch.zeros(2, 3, 6))


def test_constant_acceleration_is_tracked_after_warmup():
    f = AccelerationFilter(shape=(1, 1), alpha=0.5)
    dt = 0.01
    twist = torch.zeros(1, 1, 6)
    a = torch.zeros(1, 1, 6)
    for _k in range(200):
        twist = twist + 0.02  # constant accel of 2.0 per axis (0.02/0.01)
        a = f.update(twist, dt)
    assert torch.allclose(a, torch.full((1, 1, 6), 2.0), atol=0.05)


def test_low_alpha_attenuates_single_spike():
    f = AccelerationFilter(shape=(1, 1), alpha=0.08)
    f.update(torch.zeros(1, 1, 6), dt=0.01)            # seed prev
    spike = torch.zeros(1, 1, 6)
    spike[..., 0] = 1.0                                  # one big jump
    a = f.update(spike, dt=0.01)
    raw = 1.0 / 0.01                                     # 100
    assert a[0, 0, 0] < 0.2 * raw                        # heavily attenuated


def test_reset_clears_state():
    f = AccelerationFilter(shape=(2, 1), alpha=0.5)
    f.update(torch.ones(2, 1, 6), dt=0.01)
    f.reset()
    a = f.update(torch.ones(2, 1, 6), dt=0.01)
    assert torch.allclose(a, torch.zeros(2, 1, 6))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_update_propagates_device_through_two_calls():
    f = AccelerationFilter(shape=(2,), alpha=0.08)
    t = torch.zeros(2, 6, device="cuda")
    a1 = f.update(t, dt=0.01)
    assert a1.device.type == "cuda"          # first call: device-matched zeros
    a2 = f.update(t + 1.0, dt=0.01)          # second call must not raise a device mismatch
    assert a2.device.type == "cuda"


def test_masked_reset_zeros_selected_env_filtered_accel():
    f = AccelerationFilter(shape=(2, 1), alpha=0.5)
    f.update(torch.zeros(2, 1, 6), dt=0.1)
    f.update(torch.ones(2, 1, 6) * 0.2, dt=0.1)   # both envs now have nonzero filtered accel
    mask = torch.tensor([True, False])
    f.reset(mask)
    a = f._a_filt
    assert torch.allclose(a[0], torch.zeros(1, 6))     # env 0 filtered accel zeroed
    assert (a[1].abs() > 0).any()                      # env 1 retained
