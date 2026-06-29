import math

import torch

from lighthill import constants, frames


def test_identity_quat_gives_identity_matrix():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # (w,x,y,z)
    R = frames.quat_to_rotation_matrix(q)
    assert torch.allclose(R, torch.eye(3).unsqueeze(0), atol=1e-6)


def test_90deg_about_z_rotates_x_to_y():
    # +90° about world/body z: body-x axis maps to world +y
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    q = torch.tensor([[c, 0.0, 0.0, s]])  # rotation pi/2 about z
    R = frames.quat_to_rotation_matrix(q)
    x_body = torch.tensor([[1.0, 0.0, 0.0]])
    x_world = (R @ x_body.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(x_world, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)


def test_world_vec_to_body_is_transpose_rotation():
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    q = torch.tensor([[c, 0.0, 0.0, s]])
    v_world = torch.tensor([[0.0, 1.0, 0.0]])
    v_body = frames.world_vec_to_body(v_world, q)
    # world +y came from body +x, so body frame sees it as +x
    assert torch.allclose(v_body, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)


def test_skew_matches_cross_product():
    a = torch.tensor([[1.0, 2.0, 3.0]])
    b = torch.tensor([[4.0, 5.0, 6.0]])
    out = (frames.skew(a) @ b.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(out, torch.cross(a, b, dim=-1), atol=1e-6)


def test_constants_are_nwu_sane():
    assert constants.RHO_SEAWATER == 1025.0
    assert constants.GRAVITY == 9.81
    assert (constants.LIN, constants.ANG) == (slice(0, 3), slice(3, 6))
