# Part 3 — Language modeling

> Part 3 of **[Tokens to Agents](../../README.md)** — the foundations of agentic AI, built from scratch.

**Backprop by hand: a next-token model with no autograd.**

One embedding table, one linear layer, one softmax — and every gradient derived on paper before it is written in code. Autograd is convenient, but it hides the chain rule that everything after this depends on.

Covers: the forward pass, the cross-entropy gradient, manual SGD updates, and a learning-rate sweep showing what actually drives convergence.

**Status:** 🔧 Preparing release
