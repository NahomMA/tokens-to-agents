# Tokens to Agents

**Building the modern LLM stack from first principles, from n-grams to secure AI agents.**

![Python 3.12+](https://img.shields.io/badge/Python-3.12+-1F2937?style=flat-square&logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-managed-1F2937?style=flat-square&logo=uv&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-passing-1F2937?style=flat-square&logo=ruff&logoColor=white)
![License MIT](https://img.shields.io/badge/License-MIT-1F2937?style=flat-square)
![Built from scratch](https://img.shields.io/badge/built-from%20scratch-0C6F79?style=flat-square)
[![Part 1 published](https://img.shields.io/badge/Part%201-published-0C6F79?style=flat-square&logo=medium&logoColor=white)](https://medium.com/@nahombirhan/the-textbook-fix-that-made-my-language-model-5-worse-e8586d5639b5)
[![Website](https://img.shields.io/badge/Website-nahomcyberai.tech-0C6F79?style=flat-square&logo=safari&logoColor=white)](https://nahomcyberai.tech/)

This repository is a seven-part journey through the ideas that led from classical language models to today's agentic AI systems.

It starts with something simple: counting how often words appear together. From there, it moves through embeddings, neural language modeling, transformers, modern LLMs, tool-using agents, and finally the security problems that emerge when these systems begin to act in the world.

The goal is not just to use these models, but to understand how they work underneath the abstractions. Wherever practical, I build the important pieces from scratch before relying on higher-level libraries.

My research focuses on **agentic AI security**, and this series reflects the path behind that work. Understanding how agents fail starts with understanding the models, representations, attention mechanisms, and reasoning systems they are built on.

## The journey

| # | Part                                             | What gets built                                                                                   | Status               |
| - | ------------------------------------------------ | ------------------------------------------------------------------------------------------------- | -------------------- |
| 1 | [N-grams](parts/01-ngrams)                       | Count-based language models, smoothing, and perplexity                                            | ✅ [Published](https://medium.com/@nahombirhan/the-textbook-fix-that-made-my-language-model-5-worse-e8586d5639b5) |
| 2 | [Embeddings](parts/02-embeddings)                | TF-IDF, PPMI, and Skip-gram Word2Vec with negative sampling                                       | 🔧 Preparing release |
| 3 | [Language Modeling](parts/03-language-modeling)  | A neural next-token model with manually derived gradients and no autograd                         | 🔧 Preparing release |
| 4 | [Transformers](parts/04-transformers)            | Transformer encoder and decoder components from scratch, followed by modern Llama-style attention | 🔧 Preparing release |
| 5 | [LLMs & Prompting](parts/05-llms-and-prompting)  | In-context learning, prompting, reasoning behavior, prompt sensitivity, and RAG                   | 🔧 Preparing release |
| 6 | [Agentic AI](parts/06-agentic-ai)                | A provider-agnostic tool-using agent framework and multi-agent handoffs                           | 🔧 Preparing release |
| 7 | [Agentic AI Security](parts/07-agentic-security) | Layered defenses for LLM agents, evaluated through ablations and adversarial testing              | 🔧 Preparing release |

## Why build from scratch?

Modern libraries make powerful systems remarkably easy to use. They also hide many of the details that determine why those systems behave the way they do.

A tokenizer may silently truncate an input. Padding may accidentally contribute to a loss. An attention implementation may hide the information needed to understand a model's behavior. At the agent level, a seemingly small decision about tool calls, context, or message handling can become a security boundary.

Reimplementing the core ideas makes those assumptions visible.

The point is not to avoid libraries. It is to understand what they are doing before depending on them.

That becomes especially important in security research, where failures often live in the details hidden behind an abstraction.

## From language models to agents

The parts are meant to build on one another.

**N-grams** introduce probability, likelihood, smoothing, and perplexity using simple count-based models.

**Embeddings** move from discrete words to learned representations and show how distributional structure can be captured numerically.

**Neural language modeling** turns those representations into a next-token prediction problem and works through the learning process directly.

**Transformers** introduce self-attention and the architecture behind modern language models.

**LLMs and prompting** move from model internals to the behavior of pretrained systems, including in-context learning, reasoning, retrieval, and prompt sensitivity.

**Agentic AI** adds tools, state, execution loops, and communication between agents.

Finally, **agentic AI security** examines what changes once a language model is no longer only generating text, but is making decisions and taking actions through external systems.

The progression is intentional: **tokens → representations → models → transformers → LLMs → agents → secure agents.**

## What each part contains

Each part is designed to be both an implementation and a learning resource. Depending on the topic, it includes:

* reproducible source code,
* experiments and evaluation,
* a notebook for walking through the main ideas,
* figures and visualizations,
* a written technical article,
* and, where useful, an interactive Hugging Face demo.

The emphasis throughout the repository is on understanding the mechanism rather than only reproducing an API call.

## Repository layout

```text
core/     shared utilities for evaluation, visualization, and data loading
parts/    the seven technical parts, each with its own code, notebook, figures, and article
spaces/   interactive demos, published independently as Hugging Face Spaces
```

Shared functionality lives in `core/` so that evaluation, visualization, and experiment conventions remain consistent across the series.

## Stack

The dependency list is short on purpose. In the early parts the model *is* the contribution, so almost nothing is imported; libraries appear only where reimplementing them would teach nothing — loading pretrained weights, serving a demo.

| Part | Written by hand | Libraries |
| - | - | - |
| 1 · N-grams | counting, smoothing, interpolation, perplexity | ![NumPy](https://img.shields.io/badge/NumPy-4B5563?style=flat-square&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-4B5563?style=flat-square) |
| 2 · Embeddings | skip-gram, negative sampling, PPMI, TF-IDF | ![NumPy](https://img.shields.io/badge/NumPy-4B5563?style=flat-square&logo=numpy&logoColor=white) |
| 3 · Language Modeling | forward pass, gradients, SGD — no autograd | ![NumPy](https://img.shields.io/badge/NumPy-4B5563?style=flat-square&logo=numpy&logoColor=white) |
| 4 · Transformers | attention, multi-head, positional encoding, blocks | ![PyTorch](https://img.shields.io/badge/PyTorch-4B5563?style=flat-square&logo=pytorch&logoColor=white) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-4B5563?style=flat-square&logo=huggingface&logoColor=white) |
| 5 · LLMs & Prompting | eval harness, chunking, retrieval, RAG loop | ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-4B5563?style=flat-square&logo=huggingface&logoColor=white) ![Anthropic](https://img.shields.io/badge/Anthropic-4B5563?style=flat-square&logo=anthropic&logoColor=white) |
| 6 · Agentic AI | agent loop, provider abstraction, handoffs | ![Anthropic](https://img.shields.io/badge/Anthropic-4B5563?style=flat-square&logo=anthropic&logoColor=white) |
| 7 · Agentic AI Security | defense layers, adjudication, red-team corpus | ![Anthropic](https://img.shields.io/badge/Anthropic-4B5563?style=flat-square&logo=anthropic&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-4B5563?style=flat-square&logo=streamlit&logoColor=white) |

Parts 2–7 show the intended stack; only Part 1 is released. Nothing in Part 1 comes from an NLP library — NumPy is used as an array type and Matplotlib draws the figures.

## Quick start

The project uses [uv](https://docs.astral.sh/uv/) for reproducible dependency management.

```bash
git clone https://github.com/NahomMA/tokens-to-agents.git
cd tokens-to-agents

uv sync
uv run python parts/01-ngrams/run.py
```

`uv sync` recreates the environment from the committed `uv.lock` file and the Python version specified in `.python-version`, making it possible to reproduce the environment used for the experiments without manually managing a virtual environment.

<details>
<summary>Using pip instead</summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python parts/01-ngrams/run.py
```

Dependency versions installed this way are not guaranteed to match the versions pinned in `uv.lock`.

</details>

Each part includes its own README with the data, experiments, implementation details, and reproduction instructions specific to that stage of the series.

## About

I'm **Nahom M. Birhan**, a PhD researcher working on agentic AI security and robust AI.

My broader research asks how we can understand, evaluate, and defend AI systems as they evolve from language models into agents that interact with tools and external environments.

This repository is both a record of that technical journey and a collection of implementations I can continue building on in my research.

[nahomcyberai.tech](https://nahomcyberai.tech/) · [GitHub](https://github.com/NahomMA) · [Medium](https://medium.com/@nahombirhan) · [Hugging Face](https://huggingface.co/Nahom-M) · [LinkedIn](https://www.linkedin.com/in/nahombirhan)
