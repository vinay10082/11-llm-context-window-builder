# Multi-Document Context Builder

## Description
A precision token-aware context construction and compaction pipeline for advanced LLM interactions.

## Architecture Overview
Sliding window memory mechanism utilizing `collections.deque`, with exact token measurement via `tiktoken` encodings.

## Prerequisites
* Python 3.11+
* `tiktoken`
* OpenAI/Anthropic API access for model querying.

## Environment Variables
* `MAX_CONTEXT_TOKENS`
* `SLIDING_WINDOW_MESSAGE_CAP`
* `TIKTOKEN_ENCODING_MODEL`

## Quick Start & Usage
Instantiate the context manager, inject documents, and retrieve the safely compacted context string for API submission.

## Testing & CI
Validates eviction callbacks, token counting accuracy against known text corpuses, and system prompt preservation.
