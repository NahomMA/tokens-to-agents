"""Reproduce every number and figure in Part 3.

    uv run python parts/03-language-modeling/run.py

Nothing is hard-coded: the gradient check, both sweeps, the final test perplexity
and the Part 1 baselines it is compared with are all computed or read from disk here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from lm import (
    Bigrams,
    History,
    Params,
    Vocab,
    bigrams,
    close_vocabulary,
    forward,
    gradient_check,
    init,
    perplexity,
    train,
)

from core import metrics, viz
from core.corpus import sentences, split

HERE = Path(__file__).parent
IMAGES = HERE / "assets" / "images"
PART1 = HERE.parent / "01-ngrams" / "results.json"

CFG = {"dim": 64, "lr": 5.0, "epochs": 10, "batch": 256}
LRS = (0.2, 1.0, 5.0, 20.0)  # 100× range, centred on the chosen value
DIMS = (4, 16, 64, 256)
PROBES = ("profit", "rose", "fell", "eur")
# The claim in one picture: a verb the corpus shows often, one it shows rarely, and
# the two next words that follow both. Part 1's add-α estimator is recomputed here
# on the same pairs so the comparison is exact.
BORROW = {"contexts": ("rose", "jumped"), "nexts": ("to", "by"), "alpha": 0.01}


def main() -> None:
    viz.use_theme()
    train_s, test_s = split(sentences())
    vocab = close_vocabulary(train_s)
    fit_s, valid_s = split(train_s, test_frac=0.1, seed=0)
    fit, valid, test = (bigrams(s, vocab) for s in (fit_s, valid_s, test_s))
    print(f"{len(fit):,} training pairs · {len(valid):,} validation · {len(test):,} test · "
          f"vocab {vocab.size:,}")

    check = gradient_check(*toy_problem())
    print(f"gradient check: max relative error {check:.1e}")

    runs: dict[tuple[int, float], tuple[Params, History, float]] = {}

    def fit_model(dim: int, lr: float) -> tuple[Params, History, float]:
        """Train once per (dim, lr); the sweeps and the final model share runs."""
        if (dim, lr) not in runs:
            t0 = time.perf_counter()
            params, hist = train(fit, valid, vocab.size, **{**CFG, "dim": dim, "lr": lr})
            runs[(dim, lr)] = (params, hist, time.perf_counter() - t0)
            print(f"dim {dim:<4} lr {lr:<5} valid perplexity by epoch:",
                  " ".join(f"{v:,.0f}" for v in hist.valid_ppl), f"({runs[(dim, lr)][2]:,.0f}s)")
        return runs[(dim, lr)]

    lr_curves = {lr: fit_model(CFG["dim"], lr)[1] for lr in LRS}
    dim_best = {dim: min(fit_model(dim, CFG["lr"])[1].valid_ppl) for dim in DIMS}

    params, hist, elapsed = fit_model(CFG["dim"], CFG["lr"])
    ppl_test = perplexity(params, test)
    baselines = part1_baselines()
    print(f"\nfinal model: test perplexity {ppl_test:,.1f} "
          f"(best epoch {hist.best_epoch + 1}, {elapsed:,.0f}s)")
    print(metrics.markdown_table(
        [{"Model": k, "Test perplexity": v} for k, v in {**baselines, "Neural bigram": ppl_test}.items()],
        fmt="{:,.1f}",
    ))

    neighbors = {w: nearest(params, vocab, w) for w in PROBES}
    for w, ns in neighbors.items():
        print(f"  {w:<7}", ", ".join(f"{n} {s:.2f}" for n, s in ns))

    borrow = borrowed_rows(params, fit, vocab)
    for ctx, row in borrow.items():
        print(f"  after {ctx} (seen {row['seen']}×):",
              "  ".join(f"{n} count {c:.3f} / neural {m:.3f}" for n, (c, m) in row["p"].items()))

    figures(vocab.size, lr_curves, dim_best, ppl_test, baselines, borrow)
    save_results(fit, valid, test, vocab, check, lr_curves, dim_best, hist, ppl_test, elapsed,
                 baselines, neighbors, borrow)
    np.savez_compressed(HERE / "params.npz", E=params.E, W=params.W, b=params.b,
                        words=np.array(vocab.words))
    print("wrote", HERE / "params.npz")


def toy_problem() -> tuple[Params, np.ndarray, np.ndarray]:
    """A 7-word, 3-dimensional model with random weights: small enough to check every parameter."""
    rng = np.random.default_rng(0)
    params = init(7, 3, dtype=np.float64)
    params.W[:] = rng.normal(0.0, 0.5, params.W.shape)
    params.b[:] = rng.normal(0.0, 0.5, params.b.shape)
    return params, np.array([0, 3, 7, 3]), np.array([1, 2, 6, 2])


def part1_baselines() -> dict[str, float]:
    """Part 1's count-based perplexities on the identical test split."""
    r = json.loads(PART1.read_text())
    alpha = r["best_alpha"]["2"]
    weights = "/".join(f"{w:g}" for w in r["best_weights"])
    return {
        "Unigram": r["perplexity_unsmoothed"]["1"],
        f"Count bigram (add-{alpha:g})": r["perplexity_add_alpha"][f"2/{alpha:g}"],
        "Interpolated trigram": r["perplexity_interpolated"][weights],
    }


def nearest(params: Params, vocab: Vocab, word: str, k: int = 5) -> list[tuple[str, float]]:
    """Cosine neighbours in the embedding table the model learned for prediction."""
    e = params.E[: vocab.size].astype(np.float64)
    unit = e / np.linalg.norm(e, axis=1, keepdims=True)
    sims = unit @ unit[vocab[word]]
    sims[vocab[word]] = -np.inf
    top = np.argsort(-sims)[:k]
    return [(vocab.words[i], float(sims[i])) for i in top]


def borrowed_rows(params: Params, fit: Bigrams, vocab: Vocab) -> dict[str, dict]:
    """P(next | context) under Part 1's add-α counts and under the neural model."""
    out = {}
    for ctx in BORROW["contexts"]:
        c = vocab[ctx]
        counts = np.bincount(fit.y[fit.x == c], minlength=vocab.size).astype(float)
        smoothed = (counts + BORROW["alpha"]) / (counts.sum() + BORROW["alpha"] * vocab.size)
        neural = forward(params, np.array([c]))[1][0]
        out[ctx] = {
            "seen": int(counts.sum()),
            "p": {n: (float(smoothed[vocab[n]]), float(neural[vocab[n]])) for n in BORROW["nexts"]},
        }
    return out


def figures(vocab_size: int, lr_curves, dim_best, ppl_test: float, baselines, borrow) -> None:
    common, rare = BORROW["contexts"]
    labels = [f"{ctx} → {n}" for ctx in BORROW["contexts"] for n in BORROW["nexts"]]
    series = {
        "Neural bigram": [borrow[ctx]["p"][n][1] for ctx in BORROW["contexts"] for n in BORROW["nexts"]],
        f"Count bigram (add-{BORROW['alpha']:g})":
            [borrow[ctx]["p"][n][0] for ctx in BORROW["contexts"] for n in BORROW["nexts"]],
    }
    fig = viz.paired_bars(
        labels, series,
        title=f"Seen {borrow[rare]['seen']} times, '{rare}' gets a confident prediction — "
              f"the model learned it behaves like '{common}' (seen {borrow[common]['seen']} times)",
        xlabel="P(next word | previous word), training pairs",
    )
    print("wrote", viz.save(fig, "borrowed-row", IMAGES))

    epochs = np.arange(CFG["epochs"] + 1)
    series = {f"lr {lr:g}": [vocab_size, *h.valid_ppl] for lr, h in lr_curves.items()}
    lo, hi = min(LRS), max(LRS)
    fig = viz.trend(
        epochs, series,
        title=f"A {hi / lo:g}× range of learning rates spans 'nowhere near converged' "
              f"to 'memorising the training set'",
        xlabel="Epoch (0 = uniform guess)", ylabel="Validation perplexity", log=True,
    )
    print("wrote", viz.save(fig, "lr-sweep", IMAGES))

    gain = lambda a, b: (dim_best[a] - dim_best[b]) / dim_best[a] * 100
    fig = viz.bars(
        [f"d = {d}" for d in DIMS], [dim_best[d] for d in DIMS],
        title=f"Width buys diminishing returns: {DIMS[0]} → {DIMS[1]} dimensions cuts perplexity "
              f"{gain(DIMS[0], DIMS[1]):.0f}%, {DIMS[2]} → {DIMS[3]} only {gain(DIMS[2], DIMS[3]):.0f}% "
              f"for {DIMS[3] // DIMS[2]}× the parameters",
        xlabel="Best validation perplexity (lower is better)", best="min", fmt="{:,.0f}",
    )
    print("wrote", viz.save(fig, "dim-sweep", IMAGES))

    ours = f"Neural bigram (d={CFG['dim']})"
    count_bigram = next(v for k, v in baselines.items() if k.startswith("Count bigram"))
    fig = viz.bars(
        [*baselines, ours], [*baselines.values(), ppl_test],
        title=f"Same one word of context, no counts: {ppl_test:,.1f} against the smoothed "
              f"count bigram's {count_bigram:,.1f} — and within "
              f"{(ppl_test / baselines['Interpolated trigram'] - 1) * 100:.0f}% of the trigram",
        xlabel="Test perplexity (lower is better)", best="min", fmt="{:,.1f}", highlight=ours,
    )
    print("wrote", viz.save(fig, "perplexity-vs-counting", IMAGES))


def save_results(fit: Bigrams, valid: Bigrams, test: Bigrams, vocab: Vocab, check, lr_curves,
                 dim_best, hist, ppl_test, elapsed, baselines, neighbors, borrow) -> None:
    out = {
        "corpus": {"train_pairs": len(fit), "valid_pairs": len(valid), "test_pairs": len(test),
                   "vocab": vocab.size},
        "config": CFG,
        "gradient_check_max_rel_error": check,
        "lr_sweep": {str(lr): {"train_loss": h.train_loss, "valid_ppl": h.valid_ppl}
                     for lr, h in lr_curves.items()},
        "dim_sweep_best_valid_ppl": {str(d): v for d, v in dim_best.items()},
        "final": {"best_epoch": hist.best_epoch + 1, "valid_ppl": min(hist.valid_ppl),
                  "test_ppl": ppl_test, "train_seconds": elapsed},
        "part1_baselines": baselines,
        "neighbors": neighbors,
        "borrowed_rows": borrow,
    }
    path = HERE / "results.json"
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
