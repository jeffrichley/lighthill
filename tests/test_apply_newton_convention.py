"""The Newton adapter must convert Isaac Lab's (x,y,z,w) body_quat_w to lighthill's (w,x,y,z).

Regression guard for the convention bug that reversed the emergent swim direction and leaked force
out of plane on the Newton backend: Isaac Lab (Newton, 3.0+) reports ``body_quat_w`` scalar-LAST
(x,y,z,w) while lighthill's frame utilities are scalar-FIRST (w,x,y,z)."""

from __future__ import annotations

import math

import torch

from lighthill.apply_newton import NewtonArticulationView


class _MockData:
    def __init__(self, quat_xyzw: torch.Tensor) -> None:
        e, b = quat_xyzw.shape[:2]
        self.body_pos_w = torch.zeros(e, b, 3)
        self.body_quat_w = quat_xyzw  # Isaac Lab convention: (x, y, z, w)
        self.body_link_lin_vel_w = torch.zeros(e, b, 3)
        self.body_link_ang_vel_w = torch.zeros(e, b, 3)
        self.body_mass = torch.ones(e, b)
        self.body_inertia = torch.eye(3).reshape(1, 1, 9).expand(e, b, 9).clone()


class _MockAsset:
    def __init__(self, quat_xyzw: torch.Tensor) -> None:
        self.num_instances = quat_xyzw.shape[0]
        self.num_bodies = quat_xyzw.shape[1]
        self.data = _MockData(quat_xyzw)


def test_identity_xyzw_becomes_identity_wxyz():
    # Isaac identity (x,y,z,w)=[0,0,0,1] must come back as lighthill identity (w,x,y,z)=[1,0,0,0].
    view = NewtonArticulationView(_MockAsset(torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])))
    _, quat, _ = view.body_states()
    assert torch.allclose(quat, torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]), atol=1e-6)


def test_yaw_quat_component_order_is_reindexed():
    # 90deg yaw about z: lighthill wants (w,x,y,z)=[c,0,0,s]; Isaac reports (x,y,z,w)=[0,0,s,c].
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    view = NewtonArticulationView(_MockAsset(torch.tensor([[[0.0, 0.0, s, c]]])))
    _, quat, _ = view.body_states()
    assert torch.allclose(quat, torch.tensor([[[c, 0.0, 0.0, s]]]), atol=1e-6)
