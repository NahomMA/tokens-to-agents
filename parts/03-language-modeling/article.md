<!--
title: My Language Model Saw 'jumped' Seven Times and Knew What Came Next
subtitle: A next-token model with every gradient written by hand, and why 580,000 numbers out-predict a 20-million-cell table
tags: [Machine Learning, NLP, Deep Learning, Python, AI]
canonical: https://github.com/NahomMA/tokens-to-agents
images-to-upload: assets/images/borrowed-row.png (hero), assets/images/lr-sweep.png (step size), assets/images/perplexity-vs-counting.png (scoreboard), assets/images/dim-sweep.png (scoreboard)
status: ready
-->

# My Language Model Saw 'jumped' Seven Times and Knew What Came Next

### A next-token model with every gradient written by hand, and why 580,000 numbers out-predict a 20-million-cell table

I spend most of my research time thinking about **agentic AI security**: what happens when language models are given tools, make decisions, and begin interacting with systems outside the model itself.

But the further up the stack I work, the more useful it becomes to understand what is happening underneath it.

So I am rebuilding that stack from the bottom.

**Tokens to Agents** is a seven-part series that follows the path from classical language models to modern AI agents: n-grams, embeddings, neural language modeling, transformers, LLMs and prompting, agentic AI, and finally agentic AI security.

This is **Part 3: Language Modeling**.

Code: [github.com/NahomMA/tokens-to-agents](https://github.com/NahomMA/tokens-to-agents)

---

The word *jumped* appears seven times in my training data. Seven, in 86,587 word pairs of financial news.

Ask the count-based bigram from [Part 1](https://medium.com/@nahombirhan/the-textbook-fix-that-made-my-language-model-5-worse-e8586d5639b5) what follows *jumped* and it hedges. It gives *to* a probability of 0.06, and spreads the rest thinly over every other word it knows, because that is what smoothing does when the counts run out.

Ask the model in this post and it says *to* with probability 0.37. It has never been told that *jumped* and *rose* are related. It worked that out, and then it borrowed *rose*'s seventy-eight sightings to fill in *jumped*'s seven.

Both models see exactly the same thing: one previous word, nothing else. Yet on the test set the neural one scores a perplexity of **93.5** against the count table's **132.6**. That is 29% better with identical context, and within 4% of Part 1's trigram, which gets a whole extra word to look at.

The model is small enough that I could write every gradient by hand. So I did, and checked them against the arithmetic before a single line of training code ran. This post is that derivation, that check, and what the model learned.

## Two ways to fill in a table

A bigram model is a table. Rows are previous words, columns are next words, and each cell holds P(next | previous). With a 4,495-word vocabulary that is 4,495 × 4,495 = **20.2 million cells**, and I have 86,587 training pairs to fill them with. Almost every cell is empty. Part 1 spent its entire length on what to do about the empty ones.

The neural model refuses to store the table. Instead, every word gets 64 numbers (an embedding), and a single matrix turns those 64 numbers into a full row of 4,495 probabilities whenever one is needed:

```text
h = E[x]              64 numbers for the previous word
z = W h + b           one score per vocabulary word
p = softmax(z)        the row of the table, computed on demand
```

That is the whole network: about 580,000 parameters standing in for 20 million cells.

The trade is what makes it work. Two previous words with similar embeddings get similar rows, so every pair the corpus shows for one of them quietly teaches the other. *rose* shows up seventy-eight times, usually followed by *to* or *by*; if *jumped* lands near *rose* in the embedding space, it inherits that knowledge without ever earning it.

Which is exactly what happened:

![After jumped, seen seven times, the neural model borrows from rose](assets/images/borrowed-row.png)

After *rose*, with real data behind it, the count table's answers are real estimates. After *jumped* they are not. Add-0.01 smoothing across 4,495 words puts 45 phantom observations into every row's denominator, and 45 phantoms drown seven real ones: the row collapses to 0.06 and 0.02. The neural model's row for *jumped* looks like a row for a verb of motion, 0.37 for *to* and 0.10 for *by*, because that is what it learned *jumped* is. There is no smoothing parameter to tune. Sharing is the smoothing, and it fell out of the bottleneck for free.

[Part 2](https://medium.com/@nahombirhan/the-1990s-method-that-beat-my-neural-network-25e3e032f405) built this bridge between words and then just looked at it. This is the first time it carries traffic.

## One derivative

Training means: for each (previous, next) pair, compute the loss −log p[next], find how it changes with every parameter, and step the other way. A framework would do this silently. Doing it once on paper shows that the whole backward pass hangs on a single line.

Call the true next word *y*. The derivative of the loss with respect to the scores *z* is

```text
∂L/∂z = p − onehot(y)
```

For the right word, that is probability minus one, a push upward. For every wrong word it is the probability itself, a push down in proportion to how wrong the model was. Everything else is this vector routed backwards through the two matrix products:

```text
∂L/∂W    = (p − onehot(y)) hᵀ       outer product, V × D
∂L/∂b    =  p − onehot(y)
∂L/∂E[x] =  Wᵀ (p − onehot(y))      only the row that was looked up moves
```

Here is one step, with real digits. Three words, two dimensions, previous word 1, target word 2, a tiny random *W*. The forward pass gives p ≈ [0.332, 0.334, 0.334], near uniform, and a loss of about 1.10. The derivative is [0.332, 0.334, −0.666]: pull the target up by two thirds, push each wrong word down by a third. With a learning rate of 0.1 the embedding of word 1 moves from [0.000, 0.100] to [0.0017, 0.1017].

I keep that example in the repo because the code reproduces it digit for digit. If it ever stops doing so, something is broken.

## Where a framework would have covered for me

The backward pass in code is the three formulas above, batched:

```python
def backward(params, x, y, h, p):
    dz = p.copy()
    dz[np.arange(len(y)), y] -= 1.0      # p − onehot(y)
    dz /= len(y)                          # mean over the batch

    dE = np.zeros_like(params.E)
    np.add.at(dE, x, dz @ params.W)       # rows repeat inside a batch: accumulate
    return Grads(E=dE, W=dz.T @ h, b=dz.sum(axis=0))
```

The line worth staring at is `np.add.at`. If *the* appears forty times in a batch, its embedding row receives forty separate gradients, and the obvious `dE[x] = ...` would keep only the last one. Nothing crashes. The model just trains a little worse, forever, and you never find out. Autograd handles this correctly on your behalf, which is precisely why most people have never heard of it.

The update rule is stochastic gradient descent with nothing added:

```python
def sgd_step(params, grads, lr):
    params.E -= lr * grads.E
    params.W -= lr * grads.W
    params.b -= lr * grads.b
```

I resisted momentum and Adam on purpose. A hand-written gradient with a sign error can still converge under a clever optimiser. Plain SGD will not cover for me.

So I checked. Nudge each parameter by ±0.00001, measure how the loss moves, and compare that slope with what the formulas say. Over every parameter of the toy model the worst disagreement is 4.1 × 10⁻⁹, which is floating-point noise. A real mistake shows up as a disagreement of order one.

## Choosing a step size

With the gradients trusted, two decisions remain: how far to step, and how many numbers per word. I swept both, every run from the same seed, scored after every epoch on sentences held out from training.

![Learning rate sweep](assets/images/lr-sweep.png)

At a learning rate of 0.2 the model is still at perplexity 331 after ten epochs and dropping slowly. It would get there eventually; I do not have the patience. At 1.0 it reaches 168. At 5.0 it touches **122** in epoch 8, the best in the sweep.

Then there is 20. Its training loss is the lowest of all four runs. On the sentences it has never seen, it is the second-worst model, and getting worse by the epoch. With steps that large, 580,000 parameters can memorise 86,587 pairs faster than they can find the structure that generalises.

I keep relearning this one. The number on the screen during training is not the number you care about. The run that minimised it was close to the worst thing I built.

## The scoreboard

Hyperparameters chosen on the validation slice, then one pass over Part 1's untouched 970 test sentences:

![Test perplexity against Part 1's counting models](assets/images/perplexity-vs-counting.png)

* **Unigram, no context: 329.1**
* **Count bigram, best add-α: 132.6**
* **Neural bigram, this post: 93.5**
* **Interpolated trigram, best weights: 90.0**

One word of context, learned instead of counted, closes most of the distance to a model that sees two.

How many numbers per word does it take? Fewer than I expected.

![Embedding width sweep](assets/images/dim-sweep.png)

Four already works. Sixteen is 24% better than four, sixty-four is another 7%, and 256 buys 3% more for four times the parameters and twice the training time. Most of the value is in the first handful of dimensions. I kept 64.

The embeddings were trained to predict the next word and nothing else, so I was curious what they had picked up on the side. Nearest neighbours of *rose*: *fell*, *grew*, *owned*, *dropped*, *declined*. Four movement verbs, antonym included, which is the same geometry Part 2's co-occurrence model found by counting; the fifth is the kind of noise 86,587 pairs leave behind. *eur*'s nearest neighbour is *$*. Nobody asked for a word map. Prediction built one anyway.

## Why a security researcher bothers with a bigram

Look at `∂L/∂z = p − onehot(y)` once more. It is an instruction: here is how to move the scores to make *y* more likely. Route it one step further back and it becomes how to move the *input* to make *y* more likely. That is the entire engine behind gradient-based adversarial prompts against modern language models. The same chain rule, pointed at a target the model's owner did not pick. I do not think you can reason clearly about that attack surface until you have written the backward pass yourself at least once.

The other thing to notice is the borrowing. *jumped* got *rose*'s knowledge because they share a neighbourhood, and that sharing is the reason the model is good. It is also a path. Change what the model believes about one word and every word nearby shifts a little too. A handful of poisoned training sentences does not stay where it was put. Generalisation and contamination are the same mechanism, seen from opposite ends.

Part 7 picks both of these up with real agents in the loop. Here they are just honest observations about a model small enough to fit on one page.

## Try it yourself

```bash
git clone https://github.com/NahomMA/tokens-to-agents.git
cd tokens-to-agents

uv sync
uv run python parts/03-language-modeling/run.py
```

That reproduces the gradient check, both sweeps, the final model, every figure here and a `results.json` holding every number in this post. The model imports `numpy` and nothing else.

---

## Next: transformers

This model reads one word and forgets everything before it. Part 4 replaces that single lookup with attention, which lets every position read every other, and builds the transformer encoder and decoder out of the same parts used here: an embedding table, matrix products, a softmax. The only new idea is letting the model choose what to look at.

## References

The model is the feed-forward neural language model of **Bengio et al., [*A Neural Probabilistic Language Model*](https://www.jmlr.org/papers/v3/bengio03a.html), JMLR 2003**, in its one-word-of-context form. The derivation follows **Jurafsky & Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/), 3rd ed. draft, [Chapter 7: Neural Networks](https://web.stanford.edu/~jurafsky/slp3/7.pdf)**. The corpus is the Financial PhraseBank of [Malo et al., 2014](https://arxiv.org/abs/1307.5336), split and closed exactly as in Part 1.

---

**Tokens to Agents:** [1. N-grams](https://medium.com/@nahombirhan/the-textbook-fix-that-made-my-language-model-5-worse-e8586d5639b5) · [2. Embeddings](https://medium.com/@nahombirhan/the-1990s-method-that-beat-my-neural-network-25e3e032f405) · **3. Language Modeling** · 4. Transformers · 5. LLMs & Prompting · 6. Agentic AI · 7. Agentic AI Security

**Next: Part 4 — Transformers, coming soon.**

*Nahom M. Birhan — PhD student, agentic AI security.* [nahomcyberai.tech](https://nahomcyberai.tech/) · [GitHub](https://github.com/NahomMA) · [Hugging Face](https://huggingface.co/Nahom-M) · [LinkedIn](https://www.linkedin.com/in/nahombirhan)
