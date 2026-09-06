"""Dataset fetching and repo-relative paths.

Datasets are never committed. Each part declares what it needs; this module
downloads it once into a gitignored cache so notebooks run from any directory.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = REPO_ROOT / "data"


def part_dir(name: str) -> Path:
    """Absolute path to a part folder, e.g. part_dir("01-ngrams")."""
    path = REPO_ROOT / "parts" / name
    if not path.is_dir():
        raise FileNotFoundError(f"no such part: {path}")
    return path


def fetch(url: str, filename: str) -> Path:
    """Download `url` into the cache once and return the local path."""
    dest = CACHE / filename
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {filename} ...")
    urllib.request.urlretrieve(url, dest)
    return dest
