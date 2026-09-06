# Part 1 — N-grams

> Part 1 of **[Tokens to Agents](../../README.md)** — building the path from classical language models to modern AI agents from first principles.
>
> 📖 **[Read the article on Medium](https://medium.com/@nahombirhan/the-textbook-fix-that-made-my-language-model-5-worse-e8586d5639b5)**  ·  🔢 **[Try the interactive demo](https://huggingface.co/spaces/Nahom-M/tokens-to-agents-01-ngrams)**

**We start with the simplest possible language model: count what came before, and use those counts to predict what comes next.**

This part implements unigram, bigram, and trigram language models from scratch and uses them to explore three ideas that remain important throughout language modeling:

* next-token probability,
* perplexity,
* and what happens when the model encounters contexts it has never seen.

The most interesting result was unexpected: on this dataset, standard Laplace smoothing made the trigram model about **4.9× worse than the unigram**. Tuning the smoothing strength recovered much of that loss, and interpolating all three model orders produced the best result.

![Perplexity by model](assets/images/perplexity-by-model.png)

## Results

Dataset: **4,846 financial-news sentences**

* Train: 3,876
* Test: 970
* Vocabulary: 4,495 words

| Model  | Unsmoothed | Zero-probability events | Laplace α=1 |             Tuned α |
| ------ | ---------: | ----------------------: | ----------: | ------------------: |
| 1-gram |      329.1 |                    0.0% |       332.6 | **329.1** (α=0.001) |
| 2-gram |          ∞ |                   22.8% |       469.6 |  **132.6** (α=0.01) |
| 3-gram |          ∞ |                   53.0% |     1,642.4 | **358.4** (α=0.001) |

The best model was the interpolation:

```text
0.2 × unigram
+ 0.4 × bigram
+ 0.4 × trigram
```

**Perplexity: 90.0**

That is approximately **32% lower than the best individual tuned model**.

## What I learned

### 1. More context creates a sparsity problem

A higher-order n-gram can make a more specific prediction, but only when the relevant context has appeared in the training data.

In the test set, **53% of trigram events** occurred under contexts the unsmoothed model could not estimate, producing infinite perplexity.

### 2. Smoothing strength matters

Laplace smoothing removes zero probabilities by adding one to every possible event.

With a vocabulary of 4,495 words, however, that introduces a large amount of probability mass relative to the few observations available for many contexts.

For this dataset, smaller values of α worked substantially better.

### 3. Different model orders are useful in different situations

The trigram is specific but sparse.

The unigram is general but ignores context.

Interpolation combines those strengths rather than forcing one model order to handle every situation.

## Implementation

Everything important is implemented directly rather than through an NLP modeling library.

```text
ngrams/
├── corpus.py     # download, tokenize, split, and build the vocabulary
└── model.py      # NGramModel, InterpolatedModel, perplexity, zero_rate

run.py            # reproduce experiments and figures
notebook.ipynb    # walk through the experiment interactively
assets/images/    # generated figures
results.json      # experiment results
```

The modeling code is roughly 200 lines.

There is no autograd or training loop here. Probabilities come directly from counts, which makes the behavior of the model easy to inspect.

## Reproduce

From the repository root:

```bash
uv sync
uv run python parts/01-ngrams/run.py
```

The script reproduces the experiment and regenerates the figures.

The environment is pinned through the repository's `uv.lock` and `.python-version`.

## Data

This part uses the **Financial PhraseBank** dataset from Malo et al. (2014), specifically the `Sentences_50Agree` split.

It contains **4,846 financial-news sentences**.

The sentiment labels are not used; this experiment uses only the text.

The dataset is licensed **CC BY-NC-SA 3.0**. It is therefore downloaded on first use and cached locally rather than redistributed with this repository.

## Why start with n-grams?

An n-gram model makes the core language-modeling problem unusually easy to see:

```text
P(next token | context)
```

Here, that probability is estimated from counts.

Modern autoregressive language models use learned representations and neural networks to estimate a much richer conditional distribution, but next-token prediction remains the central thread connecting the two.

N-grams also expose their limitation clearly.

A count-based model cannot recognize that:

```text
profit rose
```

and

```text
earnings increased
```

express related ideas unless those relationships happen to appear through exact shared counts.

Words are only symbols. Similar meanings do not automatically produce similar representations.

That limitation motivates the next part of the series.

## Next: Embeddings

**Part 2 moves from counts to vectors.**

Instead of treating every word as an unrelated symbol, we will build representations in which words can become similar because they occur in similar contexts.

That is our first step from classical language models toward learned representations.

---

**[← Tokens to Agents](../../README.md)** · **Next: Part 2 — Embeddings →**
