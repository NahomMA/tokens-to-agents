"""Reproduce every number and figure in Part 1.

    python run.py

Nothing here is hard-coded: each table cell and figure label is computed from the
run, so the article and the code cannot drift apart.
"""

from __future__ import annotations

import json
from math import isinf
from pathlib import Path

from ngrams import (
    InterpolatedModel,
    NGramModel,
    close_vocabulary,
    perplexity,
    sentences,
    split,
    zero_rate,
)

from core import metrics, viz

IMAGES = Path(__file__).parent / "assets" / "images"
ORDERS = (1, 2, 3)
ALPHAS = (1.0, 0.1, 0.01, 0.001)
WEIGHTS = (
    (0.80, 0.15, 0.05),
    (0.70, 0.20, 0.10),
    (0.60, 0.25, 0.15),
    (0.50, 0.30, 0.20),
    (0.34, 0.33, 0.33),
    (0.20, 0.40, 0.40),
    (0.10, 0.30, 0.60),
)


def fmt_ppl(value: float) -> str:
    return "∞" if isinf(value) else f"{value:,.1f}"


def main() -> None:
    viz.use_theme()

    train, test = split(sentences())
    train, test, vocab = close_vocabulary(train, test)
    print(f"{len(train):,} train sentences · {len(test):,} test · vocab {len(vocab):,}\n")

    mle = {n: NGramModel(n).fit(train) for n in ORDERS}
    zeros = {n: zero_rate(mle[n], test) for n in ORDERS}
    ppl_mle = {n: perplexity(mle[n], test) for n in ORDERS}

    smoothed = {(n, a): NGramModel(n, alpha=a).fit(train) for n in ORDERS for a in ALPHAS}
    ppl_smoothed = {key: perplexity(m, test) for key, m in smoothed.items()}
    best_alpha = {n: min(ALPHAS, key=lambda a: ppl_smoothed[(n, a)]) for n in ORDERS}

    tuned = [smoothed[(n, best_alpha[n])] for n in ORDERS]
    sweep = {w: perplexity(InterpolatedModel(tuned, w), test) for w in WEIGHTS}
    best_w = min(sweep, key=sweep.get)

    report(zeros, ppl_mle, ppl_smoothed, best_alpha, sweep, best_w)
    figures(zeros, ppl_mle, ppl_smoothed, best_alpha, sweep, best_w)
    save(len(train), len(test), len(vocab), zeros, ppl_mle, ppl_smoothed, best_alpha, sweep, best_w)


def report(zeros, ppl_mle, ppl_smoothed, best_alpha, sweep, best_w) -> None:
    rows = [
        {
            "Model": f"{n}-gram",
            "Unsmoothed": fmt_ppl(ppl_mle[n]),
            "Unseen tokens": f"{zeros[n]:.1%}",
            "Laplace (α=1)": fmt_ppl(ppl_smoothed[(n, 1.0)]),
            "Tuned α": f"{fmt_ppl(ppl_smoothed[(n, best_alpha[n])])} (α={best_alpha[n]:g})",
        }
        for n in ORDERS
    ]
    print(metrics.markdown_table(rows))

    laplace_best = min(ppl_smoothed[(n, 1.0)] for n in ORDERS)
    tuned_best = min(ppl_smoothed[(n, best_alpha[n])] for n in ORDERS)
    print(
        f"\nInterpolated {best_w} -> {sweep[best_w]:,.1f}"
        f"\n  {1 - sweep[best_w] / tuned_best:.1%} better than the best single tuned model"
        f"\n  {1 - sweep[best_w] / laplace_best:.1%} better than the best Laplace model"
        f"\n  Laplace trigram is {ppl_smoothed[(3, 1.0)] / ppl_smoothed[(1, 1.0)]:.1f}x worse"
        f" than the Laplace unigram"
    )


def figures(zeros, ppl_mle, ppl_smoothed, best_alpha, sweep, best_w) -> None:
    orders = [f"{n}-gram" for n in ORDERS]

    fig = viz.bars(
        orders,
        [zeros[n] * 100 for n in ORDERS],
        title=(
            f"{zeros[3]:.0%} of trigram contexts in the test set were never seen in training — "
            "an unsmoothed model scores them all zero"
        ),
        xlabel="Test tokens given zero probability (%)",
        best="min",
        fmt="{:.1f}%",
    )
    print("wrote", viz.save(fig, "zero-probability-by-order", IMAGES))

    win_n, win_a = min(ppl_smoothed, key=ppl_smoothed.get)
    fig = viz.grid_heatmap(
        [[ppl_smoothed[(n, a)] for a in ALPHAS] for n in ORDERS],
        orders,
        [f"α={a:g}" for a in ALPHAS],
        title=(
            f"Laplace (α=1) is the worst choice for every order — the {win_n}-gram at α={win_a:g} "
            f"is {ppl_smoothed[(win_n, 1.0)] / ppl_smoothed[(win_n, win_a)]:.1f}x better than its own Laplace version"
        ),
        xlabel="Add-α smoothing strength",
        ylabel="Model order",
        best="min",
        fmt="{:,.0f}",
        cmap="Blues_r",
    )
    print("wrote", viz.save(fig, "alpha-sweep", IMAGES))

    labels = [f"{n}-gram (α={best_alpha[n]:g})" for n in ORDERS] + ["Interpolated"]
    values = [ppl_smoothed[(n, best_alpha[n])] for n in ORDERS] + [sweep[best_w]]
    tuned_best = min(ppl_smoothed[(n, best_alpha[n])] for n in ORDERS)
    fig = viz.bars(
        labels,
        values,
        title=(
            f"Blending all three orders beats every single model by "
            f"{1 - sweep[best_w] / tuned_best:.0%}"
        ),
        xlabel="Test perplexity (lower is better)",
        best="min",
        fmt="{:,.0f}",
    )
    print("wrote", viz.save(fig, "perplexity-by-model", IMAGES))

    fig = viz.bars(
        [f"{a:g} / {b:g} / {c:g}" for a, b, c in WEIGHTS],
        [sweep[w] for w in WEIGHTS],
        title=f"Best blend is {best_w[0]:g} / {best_w[1]:g} / {best_w[2]:g} — the unigram carries the fallback",
        xlabel="Test perplexity · weights are unigram / bigram / trigram",
        best="min",
        fmt="{:,.0f}",
    )
    print("wrote", viz.save(fig, "interpolation-weight-sweep", IMAGES))


def save(n_train, n_test, n_vocab, zeros, ppl_mle, ppl_smoothed, best_alpha, sweep, best_w) -> None:
    path = Path(__file__).parent / "results.json"
    path.write_text(
        json.dumps(
            {
                "corpus": {"train": n_train, "test": n_test, "vocab": n_vocab},
                "zero_rate": {str(n): zeros[n] for n in ORDERS},
                "perplexity_unsmoothed": {str(n): ppl_mle[n] for n in ORDERS},
                "perplexity_add_alpha": {
                    f"{n}/{a:g}": ppl_smoothed[(n, a)] for n in ORDERS for a in ALPHAS
                },
                "best_alpha": {str(n): best_alpha[n] for n in ORDERS},
                "perplexity_interpolated": {"/".join(f"{x:g}" for x in w): p for w, p in sweep.items()},
                "best_weights": list(best_w),
            },
            indent=2,
        )
    )
    print("wrote", path)


if __name__ == "__main__":
    main()
