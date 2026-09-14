"""Vocabulary and integer encoding shared by both embedding paths."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Vocab:
    words: list[str]
    counts: np.ndarray  # aligned with `words`

    @property
    def size(self) -> int:
        return len(self.words)

    def __contains__(self, word: str) -> bool:
        return word in self.index

    @property
    def index(self) -> dict[str, int]:
        return self._index

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", {w: i for i, w in enumerate(self.words)})


def build_vocab(sentences: list[list[str]], *, min_count: int = 5) -> Vocab:
    """Frequency-ordered vocabulary of words seen at least `min_count` times."""
    counts = Counter(t for s in sentences for t in s)
    words = [w for w, c in counts.most_common() if c >= min_count]
    return Vocab(words, np.array([counts[w] for w in words], dtype=np.int64))


def encode(sentences: list[list[str]], vocab: Vocab) -> list[np.ndarray]:
    """Sentences as id arrays; out-of-vocabulary tokens are simply dropped."""
    idx = vocab.index
    return [np.array([idx[t] for t in s if t in idx], dtype=np.int64) for s in sentences]
