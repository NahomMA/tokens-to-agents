"""Vocabulary closure and (previous token, next token) pairs.

The closure is Part 1's, applied to Part 1's split, so every perplexity here is
directly comparable with the count-based models: same 970 test sentences, same
4,495-word vocabulary, same tokens scored.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


@dataclass(frozen=True)
class Vocab:
    """Output words (never `<s>`) plus `<s>` as the one input-only symbol."""

    words: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", {w: i for i, w in enumerate(self.words)})

    @property
    def size(self) -> int:
        return len(self.words)

    @property
    def bos(self) -> int:
        return self.size

    def __getitem__(self, word: str) -> int:
        return self.index.get(word, self.index[UNK])


@dataclass(frozen=True)
class Bigrams:
    """Integer pairs: `x[i]` precedes `y[i]`. `x` may be `<s>`, `y` may be `</s>`."""

    x: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.x)


def close_vocabulary(train: list[list[str]], *, min_count: int = 2) -> Vocab:
    """Words seen at least `min_count` times in training, plus `<unk>` and `</s>`."""
    counts = Counter(t for s in train for t in s)
    words = sorted(t for t, c in counts.items() if c >= min_count)
    return Vocab(words + [UNK, EOS])


def bigrams(sentences: list[list[str]], vocab: Vocab) -> Bigrams:
    """Every scored (context, target) pair, exactly as Part 1's bigram model sees them."""
    xs, ys = [], []
    for s in sentences:
        ids = [vocab.bos] + [vocab[t] for t in s] + [vocab[EOS]]
        xs.extend(ids[:-1])
        ys.extend(ids[1:])
    return Bigrams(np.array(xs, dtype=np.int64), np.array(ys, dtype=np.int64))
