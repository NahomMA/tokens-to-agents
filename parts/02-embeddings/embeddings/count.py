"""The counting path: co-occurrence → PPMI → truncated SVD.

No training loop. The entire method is one matrix of counts and one linear-algebra
call — which is the point Part 2 makes.
"""

from __future__ import annotations

import numpy as np


def cooccurrence(sent_ids: list[np.ndarray], vocab_size: int, *, window: int = 5) -> np.ndarray:
    """Symmetric word-by-word co-occurrence counts within `window` tokens."""
    m = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    for ids in sent_ids:
        for i, w in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    m[w, ids[j]] += 1.0
    return m


def ppmi(cooc: np.ndarray, *, cds: float = 0.75) -> np.ndarray:
    """Positive pointwise mutual information with context-distribution smoothing.

    PMI is computed only where a co-occurrence was observed — the zero cells that
    would send `log` to -inf are exactly the cells PPMI clips to 0 anyway.
    """
    total = cooc.sum()
    p_word = cooc.sum(axis=1) / total
    ctx = cooc.sum(axis=0) ** cds
    p_ctx = ctx / ctx.sum()

    out = np.zeros_like(cooc)
    rows, cols = cooc.nonzero()
    pmi = np.log2(cooc[rows, cols] / total / (p_word[rows] * p_ctx[cols]))
    out[rows, cols] = np.maximum(pmi, 0.0)
    return out


def svd_vectors(ppmi_matrix: np.ndarray, *, dim: int = 100) -> np.ndarray:
    """Dense vectors from the top `dim` singular directions, √Σ-weighted."""
    u, s, _ = np.linalg.svd(ppmi_matrix, full_matrices=False)
    return u[:, :dim] * np.sqrt(s[:dim])
