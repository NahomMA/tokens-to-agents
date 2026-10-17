"""Mini-batch SGD with a per-epoch record of training loss and held-out perplexity."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import Bigrams
from .model import Params, backward, forward, init, loss, sgd_step


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    valid_ppl: list[float] = field(default_factory=list)

    @property
    def best_epoch(self) -> int:
        return int(np.argmin(self.valid_ppl))


def perplexity(params: Params, data: Bigrams, *, batch: int = 4096) -> float:
    """exp of the mean per-token negative log-likelihood over every pair in `data`."""
    total = 0.0
    for s in range(0, len(data), batch):
        x, y = data.x[s : s + batch], data.y[s : s + batch]
        total += loss(forward(params, x)[1], y) * len(y)
    return float(np.exp(total / len(data)))


def train(
    data: Bigrams,
    valid: Bigrams,
    vocab_size: int,
    *,
    dim: int = 64,
    lr: float = 0.5,
    epochs: int = 10,
    batch: int = 256,
    seed: int = 0,
) -> tuple[Params, History]:
    """Train from scratch; return the parameters at the epoch with the lowest validation perplexity."""
    rng = np.random.default_rng(seed)
    params = init(vocab_size, dim, seed=seed)
    history, best = History(), None

    for _ in range(epochs):
        order = rng.permutation(len(data))
        losses = []
        for s in range(0, len(order), batch):
            idx = order[s : s + batch]
            x, y = data.x[idx], data.y[idx]
            h, p = forward(params, x)
            losses.append(loss(p, y))
            sgd_step(params, backward(params, x, y, h, p), lr)
        history.train_loss.append(float(np.mean(losses)))
        history.valid_ppl.append(perplexity(params, valid))
        if best is None or history.valid_ppl[-1] < best[0]:
            best = (history.valid_ppl[-1], Params(params.E.copy(), params.W.copy(), params.b.copy()))

    return best[1], history
