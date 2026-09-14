"""Query and evaluate a finished embedding matrix, whatever produced it."""

from __future__ import annotations

import numpy as np

from .prep import Vocab


class WordVectors:
    """Unit-normalised word vectors: similarity is a dot product."""

    def __init__(self, vocab: Vocab, matrix: np.ndarray) -> None:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.vocab = vocab
        self.unit = matrix / np.maximum(norms, 1e-12)

    def similarity(self, a: str, b: str) -> float:
        idx = self.vocab.index
        return float(self.unit[idx[a]] @ self.unit[idx[b]])

    def neighbors(self, word: str, k: int = 5) -> list[tuple[str, float]]:
        idx = self.vocab.index[word]
        sims = self.unit @ self.unit[idx]
        best = np.argsort(-sims)
        return [(self.vocab.words[i], float(sims[i])) for i in best[1 : k + 1]]


def pair_auc(
    vectors: WordVectors,
    related: list[tuple[str, str]],
    *,
    n_random: int = 500,
    seed: int = 0,
) -> float:
    """P(sim of a related pair > sim of a random pair) — 0.5 means no signal.

    The random pairs are drawn once with a fixed seed, so every model is judged
    against the identical baseline.
    """
    rng = np.random.default_rng(seed)
    v = vectors.vocab.size
    banned = {tuple(sorted(p)) for p in related}
    rand: list[tuple[int, int]] = []
    while len(rand) < n_random:
        a, b = rng.integers(0, v, size=2)
        if a != b and tuple(sorted((vectors.vocab.words[a], vectors.vocab.words[b]))) not in banned:
            rand.append((int(a), int(b)))

    rel = np.array([vectors.similarity(a, b) for a, b in related])
    rnd = np.array([float(vectors.unit[a] @ vectors.unit[b]) for a, b in rand])
    wins = (rel[:, None] > rnd[None, :]).sum() + 0.5 * (rel[:, None] == rnd[None, :]).sum()
    return float(wins / (len(rel) * len(rnd)))


def random_pair_mean(vectors: WordVectors, *, n: int = 500, seed: int = 0) -> float:
    """Mean cosine of random pairs — the floor real similarities must clear."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, vectors.vocab.size, size=n)
    b = rng.integers(0, vectors.vocab.size, size=n)
    ok = a != b
    return float(np.einsum("nd,nd->n", vectors.unit[a[ok]], vectors.unit[b[ok]]).mean())
