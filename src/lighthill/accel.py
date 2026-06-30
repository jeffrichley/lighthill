"""Finite-difference + EMA low-pass body-acceleration estimator for the residual term."""

from __future__ import annotations

import torch
from torch import Tensor


class AccelerationFilter:
    def __init__(self, shape: tuple[int, ...], alpha: float = 0.08) -> None:
        self.alpha = alpha
        self._prev_twist: Tensor | None = None
        self._a_filt = torch.zeros(*shape, 6)

    def update(self, twist: Tensor, dt: float) -> Tensor:
        if self._prev_twist is None:
            self._prev_twist = twist.clone()
            return self._a_filt
        a_raw = (twist - self._prev_twist) / dt
        self._a_filt = (1.0 - self.alpha) * self._a_filt + self.alpha * a_raw
        self._prev_twist = twist.clone()
        return self._a_filt

    def reset(self, mask: Tensor | None = None) -> None:
        if mask is None:
            self._prev_twist = None
            self._a_filt = torch.zeros_like(self._a_filt)
            return
        self._a_filt[mask] = 0.0
        if self._prev_twist is not None:
            self._prev_twist[mask] = 0.0
