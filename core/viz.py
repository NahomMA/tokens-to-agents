"""Publication-grade figure defaults for the series.

Three rules every figure here obeys: the title states the takeaway, the winning
value is annotated on the plot, and the palette is identical across all parts.

    from core import viz
    viz.use_theme()
    fig = viz.bars(names, perplexities, title="Interpolation cuts perplexity 41%",
                   xlabel="Perplexity (lower is better)", best="min")
    viz.save(fig, "perplexity-by-smoothing")
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from textwrap import fill
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# Okabe-Ito: colorblind-safe. ACCENT carries the message, MUTED is context.
PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
ACCENT, MUTED, INK, FAINT = PALETTE[0], "#C4C4C4", "#1A1A1A", "#707070"
HIGHLIGHT = PALETTE[1]  # outlines a winning cell — reads against any fill colour

Best = Literal["max", "min"]

_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelcolor": FAINT,
    "axes.edgecolor": "#D0D0D0",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#EAEAEA",
    "grid.linewidth": 0.8,
    "xtick.color": FAINT,
    "ytick.color": FAINT,
    "xtick.direction": "out",
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def use_theme() -> None:
    """Apply the series theme. Call once at the top of a notebook or script."""
    plt.rcParams.update(_RC)


def save(fig: Figure, name: str, folder: str | Path = "assets/images") -> Path:
    """Write `fig` to `folder/name.png` at print resolution and close it."""
    path = Path(folder) / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _title(ax, title: str, xlabel: str = "", ylabel: str = "", width: int = 62) -> None:
    """Set a left-aligned takeaway title, wrapped so it never runs off the canvas."""
    ax.set_title(fill(title, width), loc="left", pad=14, color=INK)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def bars(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    xlabel: str = "",
    best: Best = "max",
    fmt: str = "{:,.4g}",
    log: bool = False,
    ref: float | None = None,
    ref_label: str = "",
    figsize: tuple[float, float] = (7.5, 4.2),
) -> Figure:
    """Sorted horizontal bars with the winner highlighted and every value labelled.

    Horizontal keeps category names readable — never rotate x labels to fit.
    `best` decides which end earns the accent colour.
    """
    labels, values = np.asarray(labels, dtype=object), np.asarray(values, dtype=float)
    order = np.argsort(values)
    if best == "min":
        order = order[::-1]
    labels, values = labels[order], values[order]
    win = len(values) - 1

    fig, ax = plt.subplots(figsize=figsize)
    colors = [MUTED] * len(values)
    colors[win] = ACCENT
    ax.barh(range(len(values)), values, color=colors, height=0.62)
    ax.set_yticks(range(len(values)), labels)

    if log:
        ax.set_xscale("log")
        pad = lambda v: v * 1.15
    else:
        ax.set_xlim(0, values.max() * 1.18)
        pad = lambda v: v + values.max() * 0.02

    for i, v in enumerate(values):
        ax.text(
            pad(v), i, fmt.format(v),
            va="center", fontsize=11,
            color=INK if i == win else FAINT,
            fontweight="bold" if i == win else "normal",
        )

    if ref is not None:
        ax.axvline(ref, color=HIGHLIGHT, lw=2, ls=(0, (4, 3)))
        ax.annotate(
            ref_label, xy=(ref, 1.0), xycoords=("data", "axes fraction"),
            xytext=(5, -4), textcoords="offset points", va="top",
            color=HIGHLIGHT, fontsize=10.5, fontweight="bold",
        )
    ax.grid(axis="y", visible=False)
    _title(ax, title, xlabel)
    return fig


def trend(
    x: Sequence[float],
    series: Mapping[str, Sequence[float]],
    *,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    log: bool = False,
    figsize: tuple[float, float] = (7.5, 4.2),
) -> Figure:
    """Line chart over steps/epochs; each series' final value is labelled inline.

    Inline labels replace the legend round-trip — the reader never has to
    match a colour to a key.
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = np.asarray(x, dtype=float)

    for i, (name, ys) in enumerate(series.items()):
        ys = np.asarray(ys, dtype=float)
        color = PALETTE[i % len(PALETTE)]
        ax.plot(x, ys, color=color, lw=2.2, solid_capstyle="round")
        ax.annotate(
            f"  {name}: {ys[-1]:,.4g}",
            xy=(x[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
            va="center", color=color, fontsize=11, fontweight="bold",
        )

    if log:
        ax.set_yscale("log")
    ax.margins(x=0.02)
    ax.set_xlim(x.min(), x.max() + (x.max() - x.min()) * 0.28)
    ax.set_xticks([t for t in ax.get_xticks() if x.min() <= t <= x.max()])
    ax.grid(axis="x", visible=False)
    _title(ax, title, xlabel, ylabel)
    return fig


def grid_heatmap(
    matrix: Sequence[Sequence[float]],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    *,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    best: Best = "max",
    fmt: str = "{:.3g}",
    cmap: str = "Blues",
    figsize: tuple[float, float] = (7.0, 4.6),
) -> Figure:
    """One annotated heatmap for a hyperparameter sweep — never N separate line plots.

    The winning cell is outlined so the result survives a skim.
    """
    m = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(m, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)

    # Pick label colour from the rendered cell luminance so any colormap stays legible,
    # reversed ones included.
    for r in range(m.shape[0]):
        for c in range(m.shape[1]):
            red, green, blue, _ = im.cmap(im.norm(m[r, c]))
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            ax.text(
                c, r, fmt.format(m[r, c]),
                ha="center", va="center", fontsize=11,
                color="white" if luminance < 0.55 else INK,
            )

    wr, wc = np.unravel_index(np.nanargmax(m) if best == "max" else np.nanargmin(m), m.shape)
    ax.add_patch(
        plt.Rectangle((wc - 0.5, wr - 0.5), 1, 1, fill=False, ec=HIGHLIGHT, lw=3.5, clip_on=False)
    )

    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02).outline.set_visible(False)
    _title(ax, title, xlabel, ylabel)
    return fig


def labeled_scatter(
    xy: Sequence[Sequence[float]],
    labels: Sequence[str],
    groups: Sequence[str] | None = None,
    *,
    title: str,
    figsize: tuple[float, float] = (8.2, 6.4),
) -> Figure:
    """A 2-D map of labelled points — built for word-embedding projections.

    Every point carries its label directly; axes are unitless projections, so
    ticks are dropped and the takeaway lives in the arrangement itself.
    """
    pts = np.asarray(xy, dtype=float)
    names = sorted(set(groups)) if groups is not None else ["all"]
    colour = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(names)}

    fig, ax = plt.subplots(figsize=figsize)
    texts = []
    for i, label in enumerate(labels):
        g = groups[i] if groups is not None else "all"
        ax.scatter(*pts[i], s=42, color=colour[g], zorder=3)
        texts.append(ax.annotate(
            label, xy=pts[i], xytext=(6, 4), textcoords="offset points",
            fontsize=11.5, color=INK, zorder=4,
        ))

    # Greedy de-overlap: give each label the first offset whose box is clear of
    # every box already placed. Beats a dependency; good enough for ~30 points.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    slots = [(6, 4, "left"), (6, -14, "left"), (-6, 4, "right"), (-6, -14, "right"),
             (0, 10, "center"), (0, -20, "center"), (6, 16, "left"), (-6, 16, "right")]
    placed = []
    for text in texts:
        for dx, dy, ha in slots:
            text.set_position((dx, dy))
            text.set_ha(ha)
            box = text.get_window_extent(renderer).expanded(1.05, 1.1)
            if not any(box.overlaps(prev) for prev in placed):
                break
        placed.append(text.get_window_extent(renderer).expanded(1.05, 1.1))
    if groups is not None:
        for g in names:
            ax.scatter([], [], s=42, color=colour[g], label=g)
        ax.legend(loc="best", fontsize=10.5, handletextpad=0.3, borderaxespad=0.4)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.14)
    ax.grid(visible=False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    _title(ax, title)
    return fig
