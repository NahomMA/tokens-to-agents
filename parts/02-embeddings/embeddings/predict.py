"""The predicting path: skip-gram with negative sampling, gradients by hand.

Pure NumPy — every gradient below is written out, not derived by a framework.
For a pair (center c, context o) with negatives n₁..n_K the loss is

    -log σ(u_o·v_c) - Σₖ log σ(-u_nₖ·v_c)

and its gradients are the three lines inside the batch loop.
"""

from __future__ import annotations

import numpy as np

from .prep import Vocab


def _noise_table(counts: np.ndarray, size: int = 1_000_000) -> np.ndarray:
    """Unigram^0.75 sampling table — a word2vec trick that draws negatives in O(1)."""
    p = counts.astype(np.float64) ** 0.75
    return np.repeat(np.arange(len(counts)), np.round(p / p.sum() * size).astype(int))


def _pairs(
    sent_ids: list[np.ndarray], keep: np.ndarray, window: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """(center, context) pairs for one epoch: frequent-word subsampling + dynamic window."""
    centers, contexts = [], []
    for ids in sent_ids:
        kept = ids[rng.random(len(ids)) < keep[ids]]
        for i, c in enumerate(kept):
            r = int(rng.integers(1, window + 1))
            for j in range(max(0, i - r), min(len(kept), i + r + 1)):
                if j != i:
                    centers.append(c)
                    contexts.append(kept[j])
    return np.array(centers), np.array(contexts)


def train_sgns(
    sent_ids: list[np.ndarray],
    vocab: Vocab,
    *,
    dim: int = 100,
    window: int = 5,
    negatives: int = 5,
    epochs: int = 5,
    lr: float = 0.025,
    subsample: float = 1e-3,
    batch: int = 1024,
    seed: int = 0,
) -> np.ndarray:
    """Input-vector matrix (vocab.size × dim) after `epochs` of SGD."""
    rng = np.random.default_rng(seed)
    v, d = vocab.size, dim
    w_in = (rng.random((v, d)) - 0.5) / d
    w_out = np.zeros((v, d))

    freq = vocab.counts / vocab.counts.sum()
    keep = np.minimum(1.0, np.sqrt(subsample / freq) + subsample / freq)
    table = _noise_table(vocab.counts)

    for epoch in range(epochs):
        c_all, o_all = _pairs(sent_ids, keep, window, rng)
        order = rng.permutation(len(c_all))
        c_all, o_all = c_all[order], o_all[order]
        step = lr * (1.0 - epoch / epochs)

        for s in range(0, len(c_all), batch):
            c, o = c_all[s : s + batch], o_all[s : s + batch]
            n = table[rng.integers(0, len(table), size=(len(c), negatives))]

            v_c, u_o, u_n = w_in[c], w_out[o], w_out[n]
            g_pos = _sigmoid(np.einsum("bd,bd->b", v_c, u_o)) - 1.0  # dL/d(u_o·v_c)
            g_neg = _sigmoid(np.einsum("bd,bkd->bk", v_c, u_n))  # dL/d(u_n·v_c)

            grad_c = g_pos[:, None] * u_o + np.einsum("bk,bkd->bd", g_neg, u_n)
            np.add.at(w_in, c, -step * grad_c)
            np.add.at(w_out, o, -step * (g_pos[:, None] * v_c))
            np.add.at(w_out, n.ravel(), -step * (g_neg[:, :, None] * v_c[:, None, :]).reshape(-1, d))

    return w_in


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
