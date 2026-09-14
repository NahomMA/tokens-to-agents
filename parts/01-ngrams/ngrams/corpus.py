"""Corpus access for Part 1, plus the vocabulary closure n-gram models need.

Fetching, tokenization and the split live in `core.corpus` — one canonical copy
shared by every part of the series.
"""

from __future__ import annotations

from collections import Counter

from core.corpus import sentences, split, tokenize

BOS, EOS, UNK = "<s>", "</s>", "<unk>"

__all__ = ["BOS", "EOS", "UNK", "close_vocabulary", "sentences", "split", "tokenize"]


def close_vocabulary(
    train: list[list[str]], test: list[list[str]], *, min_count: int = 2
) -> tuple[list[list[str]], list[list[str]], set[str]]:
    """Map rare training words and every unseen test word to `<unk>`.

    Without this, a single out-of-vocabulary word makes test perplexity infinite
    no matter how the model is smoothed — the model would be judged on its
    vocabulary rather than on its estimates.
    """
    counts = Counter(t for s in train for t in s)
    vocab = {t for t, c in counts.items() if c >= min_count}
    vocab |= {UNK, EOS}
    keep = lambda s: [t if t in vocab else UNK for t in s]  # noqa: E731
    return [keep(s) for s in train], [keep(s) for s in test], vocab
