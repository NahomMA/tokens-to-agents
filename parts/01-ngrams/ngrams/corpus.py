"""Financial PhraseBank: fetch, tokenize, split.

The corpus is licensed CC BY-NC-SA 3.0, so it is downloaded on demand and never
redistributed with this repository.
"""

from __future__ import annotations

import random
import re
import zipfile
from collections import Counter
from pathlib import Path

from core.data import fetch

URL = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"
MEMBER = "FinancialPhraseBank-v1.0/Sentences_50Agree.txt"

BOS, EOS, UNK = "<s>", "</s>", "<unk>"

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*|[^\sa-z0-9]")


def _extract() -> Path:
    """Download the corpus archive once and unpack the sentence file beside it."""
    archive = fetch(URL, "financial_phrasebank.zip")
    target = archive.with_name("financial_phrasebank.txt")
    if not target.exists():
        with zipfile.ZipFile(archive) as z:
            target.write_bytes(z.read(MEMBER))
    return target


def sentences() -> list[list[str]]:
    """Every sentence as a lowercase token list, sentiment labels discarded."""
    text = _extract().read_text(encoding="latin-1")
    rows = (line.rsplit("@", 1)[0] for line in text.splitlines() if line.strip())
    return [_TOKEN.findall(row.lower()) for row in rows]


def split(
    sents: list[list[str]], *, test_frac: float = 0.2, seed: int = 0
) -> tuple[list[list[str]], list[list[str]]]:
    """Shuffle once with a fixed seed, then cut."""
    shuffled = sents[:]
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1 - test_frac))
    return shuffled[:cut], shuffled[cut:]


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
    keep = lambda s: [t if t in vocab else UNK for t in s]
    return [keep(s) for s in train], [keep(s) for s in test], vocab
