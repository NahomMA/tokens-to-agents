"""Financial PhraseBank: fetch, tokenize, split — the series' shared corpus.

Licensed CC BY-NC-SA 3.0, so it is downloaded on demand and never redistributed.
Parts 1 and 2 train on exactly these sentences, which is what lets Part 2 answer
a question Part 1 could only pose.
"""

from __future__ import annotations

import random
import re
import zipfile

from core.data import fetch

URL = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"
MEMBER = "FinancialPhraseBank-v1.0/Sentences_50Agree.txt"

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*|[^\sa-z0-9]")


def tokenize(text: str) -> list[str]:
    """The one tokenizer every part uses, applied to arbitrary text."""
    return _TOKEN.findall(text.lower())


def sentences() -> list[list[str]]:
    """Every corpus sentence as a lowercase token list, sentiment labels discarded."""
    archive = fetch(URL, "financial_phrasebank.zip")
    target = archive.with_name("financial_phrasebank.txt")
    if not target.exists():
        with zipfile.ZipFile(archive) as z:
            target.write_bytes(z.read(MEMBER))
    text = target.read_text(encoding="latin-1")
    rows = (line.rsplit("@", 1)[0] for line in text.splitlines() if line.strip())
    return [tokenize(row) for row in rows]


def split(
    sents: list[list[str]], *, test_frac: float = 0.2, seed: int = 0
) -> tuple[list[list[str]], list[list[str]]]:
    """Shuffle once with a fixed seed, then cut."""
    shuffled = sents[:]
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1 - test_frac))
    return shuffled[:cut], shuffled[cut:]
