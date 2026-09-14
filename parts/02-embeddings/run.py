"""Reproduce every number and figure in Part 2.

    uv run python parts/02-embeddings/run.py            # final models (~2 min)
    uv run python parts/02-embeddings/run.py --sweep    # adds the skip-gram tuning table

Nothing is hard-coded: every table cell and figure label is computed from the run,
and the vectors the demo Space serves are written by this script.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from embeddings import (
    Vocab,
    WordVectors,
    build_vocab,
    cooccurrence,
    encode,
    pair_auc,
    ppmi,
    random_pair_mean,
    svd_vectors,
    train_sgns,
)

from core import metrics, viz
from core.corpus import sentences

HERE = Path(__file__).parent
IMAGES = HERE / "assets" / "images"

COUNT_CFG = {"window": 2, "dim": 100}
SGNS_CFG = {"window": 5, "dim": 100, "epochs": 20}  # best of the --sweep grid
SGNS_SWEEP = (
    {"epochs": 10, "window": 5, "dim": 100},
    {"epochs": 20, "window": 5, "dim": 100},
    {"epochs": 10, "window": 2, "dim": 100},
    {"epochs": 10, "window": 5, "dim": 50},
    {"epochs": 20, "window": 2, "dim": 50},
)

# Twenty human-judged related pairs, every word ≥5 corpus occurrences. The claim
# each model is scored on: related pairs should outrank random ones.
RELATED = [
    ("profit", "earnings"), ("sales", "revenue"), ("rose", "increased"),
    ("fell", "decreased"), ("grew", "increased"), ("declined", "decreased"),
    ("rise", "increase"), ("billion", "million"), ("january", "december"),
    ("cut", "reduced"), ("improve", "improved"), ("estimates", "forecast"),
    ("president", "ceo"), ("bank", "banks"), ("share", "shares"),
    ("plant", "factory"), ("staff", "employees"), ("contract", "agreement"),
    ("talks", "negotiations"), ("drop", "fell"),
]

SHOWCASE = {
    "upward moves": ["rose", "increased", "grew", "jumped", "doubled"],
    "downward moves": ["fell", "decreased", "declined", "dropped", "narrowed"],
    "profit & loss": ["profit", "loss", "earnings", "sales", "revenue", "income"],
    "people & organisations": ["ceo", "president", "employees", "staff", "company", "bank"],
}

PROMISE = [
    ("profit", "earnings"), ("rose", "increased"), ("fell", "decreased"),
    ("sales", "revenue"), ("staff", "employees"), ("talks", "negotiations"),
]


def main() -> None:
    viz.use_theme()
    sents = sentences()
    vocab = build_vocab(sents, min_count=5)
    ids = encode(sents, vocab)
    n_tokens = sum(len(s) for s in ids)
    print(f"{len(sents):,} sentences · {n_tokens:,} in-vocab tokens · vocab {vocab.size:,}")

    for a, b in RELATED:
        assert a in vocab and b in vocab, f"evaluation pair word missing: {a}/{b}"
    for w in (w for ws in SHOWCASE.values() for w in ws):
        assert w in vocab, f"showcase word missing: {w}"

    t0 = time.perf_counter()
    counted = WordVectors(
        vocab, svd_vectors(ppmi(cooccurrence(ids, vocab.size, window=COUNT_CFG["window"])),
                           dim=COUNT_CFG["dim"])
    )
    t_count = time.perf_counter() - t0

    t0 = time.perf_counter()
    predicted = WordVectors(
        vocab, train_sgns(ids, vocab, seed=0, **{k: v for k, v in SGNS_CFG.items()})
    )
    t_predict = time.perf_counter() - t0

    models = {"ppmi_svd": counted, "sgns": predicted}
    aucs = {k: pair_auc(m, RELATED) for k, m in models.items()}
    floors = {k: random_pair_mean(m) for k, m in models.items()}

    report(vocab, models, aucs, floors, t_count, t_predict)
    figures(models, aucs, floors)

    sweep = sgns_sweep(ids, vocab) if "--sweep" in sys.argv else None
    save_results(len(sents), n_tokens, vocab, models, aucs, floors, sweep)
    save_vectors(vocab, models, aucs, floors)


def report(vocab: Vocab, models, aucs, floors, t_count: float, t_predict: float) -> None:
    rows = [
        {
            "Model": name,
            "Pair AUC": aucs[key],
            "Random-pair mean cosine": f"{floors[key]:+.3f}",
            "Build time": f"{t:,.0f}s",
        }
        for (key, name), t in zip(
            [("ppmi_svd", "PPMI + SVD (counting)"), ("sgns", "Skip-gram (predicting)")],
            [t_count, t_predict],
        )
    ]
    print("\n" + metrics.markdown_table(rows, fmt="{:.3f}"))
    for key, wv in models.items():
        print(f"\n{key} neighbors")
        for w in ("profit", "rose", "fell"):
            print(f"  {w:<7}", ", ".join(f"{n} {s:.2f}" for n, s in wv.neighbors(w, 5)))


def figures(models, aucs, floors) -> None:
    counted = models["ppmi_svd"]

    words = [w for ws in SHOWCASE.values() for w in ws]
    groups = [g for g, ws in SHOWCASE.items() for _ in ws]
    sub = counted.unit[[counted.vocab.index[w] for w in words]]
    centered = sub - sub.mean(axis=0)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    fig = viz.labeled_scatter(
        u[:, :2] * s[:2], words, groups,
        title="No labels and no training loop — words arrange themselves by meaning "
              "from co-occurrence counts alone",
    )
    print("wrote", viz.save(fig, "word-map", IMAGES))

    fig = viz.bars(
        ["PPMI + SVD\n(counting)", "Skip-gram\n(predicting)"],
        [aucs["ppmi_svd"], aucs["sgns"]],
        title=f"Counting beats the neural network on this corpus — "
              f"AUC {aucs['ppmi_svd']:.2f} vs {aucs['sgns']:.2f}",
        xlabel="Related-vs-random pair AUC (0.5 = chance)",
        ref=0.5, ref_label="chance",
        fmt="{:.3f}",
    )
    print("wrote", viz.save(fig, "auc-count-vs-predict", IMAGES))

    floor = floors["ppmi_svd"]
    fig = viz.bars(
        [f"{a} · {b}" for a, b in PROMISE],
        [counted.similarity(a, b) for a, b in PROMISE],
        title="The pairs Part 1 could not connect all sit far above the random floor",
        xlabel="Cosine similarity (PPMI + SVD)",
        ref=floor, ref_label=f"random pairs {floor:+.2f}",
        fmt="{:.2f}",
    )
    print("wrote", viz.save(fig, "promise-pairs", IMAGES))


def sgns_sweep(ids, vocab: Vocab) -> list[dict]:
    rows = []
    for cfg in SGNS_SWEEP:
        wv = WordVectors(vocab, train_sgns(ids, vocab, seed=0, **cfg))
        rows.append({**cfg, "auc": pair_auc(wv, RELATED), "random_mean": random_pair_mean(wv)})
        print(f"sweep {cfg}: AUC {rows[-1]['auc']:.3f}")
    return rows


def save_results(n_sents, n_tokens, vocab: Vocab, models, aucs, floors, sweep) -> None:
    out = {
        "corpus": {"sentences": n_sents, "tokens": n_tokens, "vocab": vocab.size},
        "configs": {"ppmi_svd": COUNT_CFG, "sgns": SGNS_CFG},
        "auc": aucs,
        "random_pair_mean": floors,
        "promise_pairs": {
            f"{a}/{b}": {k: models[k].similarity(a, b) for k in models} for a, b in PROMISE
        },
        "neighbors": {
            k: {w: m.neighbors(w, 5) for w in ("profit", "rose", "fell", "eur")}
            for k, m in models.items()
        },
        "related_pairs": RELATED,
    }
    path = HERE / "results.json"
    if sweep is None and path.exists():  # keep the sweep table from a previous --sweep run
        sweep = json.loads(path.read_text()).get("sgns_sweep")
    if sweep is not None:
        out["sgns_sweep"] = sweep
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


def save_vectors(vocab: Vocab, models, aucs, floors) -> None:
    """Everything the demo Space serves, in one float16 npz — vectors, scores, showcase."""
    path = HERE / "vectors.npz"
    np.savez_compressed(
        path,
        words=np.array(vocab.words),
        counts=vocab.counts,
        ppmi_svd=models["ppmi_svd"].unit.astype(np.float16),
        sgns=models["sgns"].unit.astype(np.float16),
        auc_ppmi_svd=aucs["ppmi_svd"],
        auc_sgns=aucs["sgns"],
        floor_ppmi_svd=floors["ppmi_svd"],
        floor_sgns=floors["sgns"],
        showcase_words=np.array([w for ws in SHOWCASE.values() for w in ws]),
        showcase_groups=np.array([g for g, ws in SHOWCASE.items() for _ in ws]),
    )
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
