"""Finite-difference + EMA low-pass body-acceleration estimator for the residual term."""

from __future__ import annotations

import torch
from torch import Tensor


class AccelerationFilter:
    def __init__(self, shape: tuple[int, ...], alpha: float = 0.08) -> None:
        self.alpha = alpha
        self._shape = shape
        self._prev_twist: Tensor | None = None
        self._a_filt: Tensor | None = None

    def update(self, twist: Tensor, dt: float) -> Tensor:
        if self._a_filt is None:
            self._a_filt = torch.zeros(*self._shape, 6, device=twist.device, dtype=twist.dtype)
        if self._prev_twist is None:
            self._prev_twist = twist.clone()
            return self._a_filt
        a_raw = (twist - self._prev_twist) / dt
        self._a_filt = (1.0 - self.alpha) * self._a_filt + self.alpha * a_raw
        self._prev_twist = twist.clone()
        return self._a_filt

    def reset(self, mask: Tensor | None = None) -> None:
        """Clear filter state, globally or per environment.

        Full reset (``mask is None``):
            Sets ``_prev_twist`` and ``_a_filt`` both to ``None`` so the
            *next* ``update`` re-allocates device-matched zeros and the
            first-call-returns-zero contract is re-armed.

        Masked reset (``mask`` given):
            Zeros ``_a_filt`` and ``_prev_twist`` for the selected
            environments only.  Because ``_prev_twist`` cannot be
            *partially* ``None``, the re-arm path is not triggered:
            the next ``update`` for a reset environment computes
            acceleration against ``prev_twist = 0``, which may produce
            a nonzero first-step value.  Task 4 callers must not assume
            a masked-reset environment's next ``update`` returns zero.
        """
        if mask is None:
            self._prev_twist = None
            self._a_filt = None
            return
        if self._a_filt is not None:
            self._a_filt[mask] = 0.0
        if self._prev_twist is not None:
            self._prev_twist[mask] = 0.0
