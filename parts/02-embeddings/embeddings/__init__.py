"""Part 2 — word embeddings from scratch: count-based and prediction-based.

Implementations follow Jurafsky & Martin, *Speech and Language Processing*
(3rd ed. draft, 2026), Chapter 5 "Embeddings": https://web.stanford.edu/~jurafsky/slp3/
"""

from .count import cooccurrence, ppmi, svd_vectors
from .predict import train_sgns
from .prep import Vocab, build_vocab, encode
from .vectors import WordVectors, pair_auc, random_pair_mean

__all__ = [
    "Vocab",
    "WordVectors",
    "build_vocab",
    "cooccurrence",
    "encode",
    "pair_auc",
    "ppmi",
    "random_pair_mean",
    "svd_vectors",
    "train_sgns",
]
