"""Count-based n-gram language models: MLE, add-alpha, and linear interpolation.

Every probability here comes from counting. There is no training loop and nothing
is learned by gradient descent — which is exactly why the failure modes are so
easy to see.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from math import exp, inf, log
from typing import Protocol

from .corpus import BOS, EOS

Sentence = Sequence[str]


class LanguageModel(Protocol):
    """Anything that can score the next token given a context."""

    n: int

    def prob(self, context: tuple[str, ...], token: str) -> float: ...


class NGramModel:
    """An n-gram model estimated by counting.

    `alpha=0` is unsmoothed maximum likelihood: any context-token pair missing
    from training gets probability zero. `alpha>0` is add-alpha smoothing
    (`alpha=1` is Laplace), which moves that mass onto unseen events.
    """

    def __init__(self, n: int, alpha: float = 0.0) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n, self.alpha = n, alpha
        self._ngrams: Counter[tuple[str, ...]] = Counter()
        self._contexts: Counter[tuple[str, ...]] = Counter()
        self._vocab_size = 0

    def fit(self, sentences: Iterable[Sentence]) -> NGramModel:
        vocab = set()
        for sentence in sentences:
            padded = pad(sentence, self.n)
            vocab.update(padded)
            for i in range(self.n - 1, len(padded)):
                gram = tuple(padded[i - self.n + 1 : i + 1])
                self._ngrams[gram] += 1
                self._contexts[gram[:-1]] += 1
        vocab.discard(BOS)
        self._vocab_size = len(vocab)
        return self

    def prob(self, context: tuple[str, ...], token: str) -> float:
        numerator = self._ngrams[context + (token,)] + self.alpha
        denominator = self._contexts[context] + self.alpha * self._vocab_size
        return numerator / denominator if denominator else 0.0

    def __repr__(self) -> str:
        kind = "MLE" if self.alpha == 0 else f"add-{self.alpha:g}"
        return f"NGramModel(n={self.n}, {kind})"


class InterpolatedModel:
    """Jelinek-Mercer interpolation: a weighted blend of orders 1..n.

    A trigram estimate is sharp but sparse, a unigram estimate is blunt but always
    defined. Mixing them keeps the sharpness where the counts support it and falls
    back gracefully where they do not.
    """

    def __init__(self, models: Sequence[NGramModel], weights: Sequence[float]) -> None:
        if len(models) != len(weights):
            raise ValueError("one weight per model")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError(f"weights must sum to 1, got {sum(weights)}")
        order = sorted(range(len(models)), key=lambda i: models[i].n)
        self.models = [models[i] for i in order]
        self.weights = [weights[i] for i in order]
        self.n = self.models[-1].n

    def prob(self, context: tuple[str, ...], token: str) -> float:
        return sum(
            w * m.prob(context[len(context) - (m.n - 1) :] if m.n > 1 else (), token)
            for w, m in zip(self.weights, self.models)
        )

    def __repr__(self) -> str:
        parts = ", ".join(f"{m.n}:{w:g}" for m, w in zip(self.models, self.weights))
        return f"InterpolatedModel({parts})"


def pad(sentence: Sentence, n: int) -> list[str]:
    """Add n-1 sentence-start markers and one end marker."""
    return [BOS] * (n - 1) + list(sentence) + [EOS]


def _scored(model: LanguageModel, sentences: Iterable[Sentence]):
    for sentence in sentences:
        padded = pad(sentence, model.n)
        for i in range(model.n - 1, len(padded)):
            yield model.prob(tuple(padded[i - model.n + 1 : i]), padded[i])


def perplexity(model: LanguageModel, sentences: Iterable[Sentence]) -> float:
    """Perplexity over every scored token. One zero-probability token makes it infinite."""
    total, count = 0.0, 0
    for p in _scored(model, sentences):
        if p <= 0.0:
            return inf
        total += log(p)
        count += 1
    if not count:
        raise ValueError("no tokens scored")
    return exp(-total / count)


def zero_rate(model: LanguageModel, sentences: Iterable[Sentence]) -> float:
    """Fraction of test tokens the model assigns probability zero.

    This is what an infinite perplexity is actually reporting, expressed as a
    number you can compare across model orders.
    """
    zeros, count = 0, 0
    for p in _scored(model, sentences):
        zeros += p <= 0.0
        count += 1
    return zeros / count if count else 0.0
