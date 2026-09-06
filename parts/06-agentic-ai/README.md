# Part 6 — Agentic AI

> Part 6 of **[Tokens to Agents](../../README.md)** — the foundations of agentic AI, built from scratch.

**The model gets a loop, tools, and other models to talk to.**

An agent is an LLM plus control flow. This part builds that control flow directly rather than importing a framework, so every message and handoff stays visible.

Covers: a provider-agnostic client layer across multiple model vendors, pluggable pipeline stages behind typed interfaces, a two-agent analyser-to-responder handoff on a local model, and streaming reasoning-step interception.

**Status:** 🔧 Preparing release
