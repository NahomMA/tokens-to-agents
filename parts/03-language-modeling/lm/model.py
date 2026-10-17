"""The model and its gradients, written out.

Forward, for a batch of previous tokens `x` and next tokens `y`:

    h = E[x]                  embedding lookup       (B, D)
    z = h Wᵀ + b              logits                 (B, V)
    p = softmax(z)            next-token distribution
    L = -mean log p[y]        cross-entropy

Backward, by the chain rule — the one derivative that carries everything is
∂L/∂z = (p - onehot(y)) / B:

    ∂L/∂W = ∂zᵀ h            ∂L/∂b = Σ ∂z
    ∂L/∂h = ∂z W             ∂L/∂E[x] += ∂L/∂h   (scatter-add: repeated x accumulate)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Params:
    E: np.ndarray  # (V + 1, D): one row per output word, plus `<s>`
    W: np.ndarray  # (V, D)
    b: np.ndarray  # (V,)


@dataclass(frozen=True)
class Grads:
    E: np.ndarray
    W: np.ndarray
    b: np.ndarray


def init(vocab_size: int, dim: int, *, seed: int = 0, dtype: type = np.float32) -> Params:
    """Small random embeddings, zero output layer: the first prediction is uniform."""
    rng = np.random.default_rng(seed)
    return Params(
        E=rng.normal(0.0, 0.1, size=(vocab_size + 1, dim)).astype(dtype),
        W=np.zeros((vocab_size, dim), dtype=dtype),
        b=np.zeros(vocab_size, dtype=dtype),
    )


def forward(params: Params, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (hidden, probabilities) for a batch of previous-token ids."""
    h = params.E[x]
    z = h @ params.W.T + params.b
    z -= z.max(axis=1, keepdims=True)  # softmax is shift-invariant; this stops overflow
    np.exp(z, out=z)
    z /= z.sum(axis=1, keepdims=True)
    return h, z


def loss(p: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of the true next tokens."""
    return float(-np.log(p[np.arange(len(y)), y].astype(np.float64)).mean())


def backward(params: Params, x: np.ndarray, y: np.ndarray, h: np.ndarray, p: np.ndarray) -> Grads:
    """Gradients of the mean cross-entropy with respect to every parameter."""
    dz = p.copy()
    dz[np.arange(len(y)), y] -= 1.0
    dz /= len(y)

    dE = np.zeros_like(params.E)
    np.add.at(dE, x, dz @ params.W)
    return Grads(E=dE, W=dz.T @ h, b=dz.sum(axis=0))


def sgd_step(params: Params, grads: Grads, lr: float) -> None:
    """Plain stochastic gradient descent, in place."""
    params.E -= lr * grads.E
    params.W -= lr * grads.W
    params.b -= lr * grads.b
