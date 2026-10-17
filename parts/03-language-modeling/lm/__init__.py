"""Part 3 — a neural next-token model with every gradient derived by hand.

One embedding table, one linear layer, one softmax, no autograd. The model
follows the feed-forward language model of Jurafsky & Martin, *Speech and
Language Processing* (3rd ed. draft, 2026), Chapter 7: https://web.stanford.edu/~jurafsky/slp3/
"""

from .check import gradient_check
from .data import BOS, EOS, UNK, Bigrams, Vocab, bigrams, close_vocabulary
from .model import Grads, Params, backward, forward, init, sgd_step
from .train import History, perplexity, train

__all__ = [
    "BOS",
    "EOS",
    "UNK",
    "Bigrams",
    "Grads",
    "History",
    "Params",
    "Vocab",
    "backward",
    "bigrams",
    "close_vocabulary",
    "forward",
    "gradient_check",
    "init",
    "perplexity",
    "sgd_step",
    "train",
]
