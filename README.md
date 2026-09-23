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
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in `LLM_PROVIDER` plus the matching API key.
3. Run via `main.py`:

```bash
# One-shot query with documents injected into context
python main.py --doc report.txt --doc notes.md --query "Summarize these."

# Interactive chat loop, context persists across turns via the sliding window
python main.py --doc report.txt
```

`main.py` wires together `Config`, `TokenCounter`, `DocumentStore`,
`SlidingWindowContextManager`, and `LLMClient` from the `context_builder`
package: documents are rendered into a token-budgeted block, chat turns are
tracked in a `collections.deque`-backed sliding window, and the system
prompt is always preserved at the front of the built context. When the
token budget is exceeded, the oldest window messages are evicted (logged to
stderr) before the request is sent to the configured provider.
