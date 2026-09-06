"""Metrics shared across the series, and markdown tables for READMEs and articles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def perplexity(log_probs: Sequence[float], *, base: float = np.e) -> float:
    """Perplexity from per-token log probabilities.

    Pass only *scored* tokens. Including padding is the classic way to report a
    perplexity near 1.0 that means nothing.
    """
    lp = np.asarray(log_probs, dtype=float)
    if lp.size == 0:
        raise ValueError("no tokens scored")
    if np.any(np.isneginf(lp)):
        return float("inf")
    return float(base ** (-lp.mean()))


def classification_metrics(y_true: Sequence, y_pred: Sequence) -> dict[str, float]:
    """Accuracy plus macro precision/recall/F1, computed without sklearn."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    precisions, recalls, f1s = [], [], []
    for label in np.unique(y_true):
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
    }


def markdown_table(rows: Sequence[Mapping[str, object]], *, fmt: str = "{:,.4g}") -> str:
    """Render rows as a markdown table, ready to paste into a README or article."""
    if not rows:
        return ""
    headers = list(rows[0])
    fmt_cell = lambda v: fmt.format(v) if isinstance(v, float) else str(v)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(fmt_cell(r[h]) for h in headers) + " |" for r in rows]
    return "\n".join(lines)
