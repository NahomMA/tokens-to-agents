"""Finite-difference check of the hand-written gradients."""

from __future__ import annotations

import numpy as np

from .model import Params, backward, forward, loss


def gradient_check(params: Params, x: np.ndarray, y: np.ndarray, *, eps: float = 1e-5) -> float:
    """Largest relative error between analytic and numerical gradients over all parameters.

    Central differences agree with a correct derivative to ~1e-8 (in float64;
    pass params built with `init(..., dtype=np.float64)`). A wrong one shows up
    as errors of order 1.
    """
    h, p = forward(params, x)
    analytic = backward(params, x, y, h, p)
    worst = 0.0
    for name in ("E", "W", "b"):
        theta, grad = getattr(params, name), getattr(analytic, name)
        for idx in np.ndindex(theta.shape):
            original = theta[idx]
            theta[idx] = original + eps
            up = loss(forward(params, x)[1], y)
            theta[idx] = original - eps
            down = loss(forward(params, x)[1], y)
            theta[idx] = original
            numeric = (up - down) / (2 * eps)
            worst = max(worst, abs(numeric - grad[idx]) / max(abs(numeric) + abs(grad[idx]), 1e-12))
    return worst
