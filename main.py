"""Entry point for the Multi-Document Context Builder.

Usage:
    python main.py --doc report.txt --doc notes.md --query "Summarize these."
    python main.py --doc report.txt            # interactive chat loop
"""

from __future__ import annotations

import argparse
import sys

from context_builder import Config, DocumentStore, LLMClient, SlidingWindowContextManager, TokenCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Document Context Builder")
    parser.add_argument(
        "--doc",
        action="append",
        dest="docs",
        default=[],
        help="Path to a document to inject into context. Repeatable.",
    )
    parser.add_argument(
        "--query",
        help="Single question to ask. Omit to start an interactive chat loop.",
    )
    parser.add_argument(
        "--system-prompt",
        help="Override the SYSTEM_PROMPT env var for this run.",
    )
    return parser.parse_args()


def build_components(args: argparse.Namespace):
    config = Config.from_env()
    token_counter = TokenCounter(config.tiktoken_encoding_model)
    doc_store = DocumentStore(token_counter=token_counter)

    for doc_path in args.docs:
        doc_store.add_file(doc_path)

    def log_eviction(message) -> None:
        preview = message["content"][:60].replace("\n", " ")
        print(f"[context] evicted {message['role']} message: {preview!r}", file=sys.stderr)

    window = SlidingWindowContextManager(
        token_counter=token_counter,
        max_context_tokens=config.max_context_tokens,
        message_cap=config.sliding_window_message_cap,
        system_prompt=args.system_prompt or config.system_prompt,
        on_evict=log_eviction,
    )

    llm_client = LLMClient(config)
    return config, token_counter, doc_store, window, llm_client


def ask(doc_store: DocumentStore, window: SlidingWindowContextManager, llm_client: LLMClient, question: str) -> str:
    window.add_message("user", question)

    # Reserve roughly a third of the budget for injected document context so
    # conversation history is never fully crowded out.
    doc_budget = window.max_context_tokens // 3
    document_block = doc_store.render(max_tokens=doc_budget) if doc_store.documents else ""

    context = window.build_context(extra_context=document_block)
    answer = llm_client.query(context)

    window.add_message("assistant", answer)
    return answer


def main() -> None:
    args = parse_args()
    _, _, doc_store, window, llm_client = build_components(args)

    if args.query:
        print(ask(doc_store, window, llm_client, args.query))
        return

    print("Multi-Document Context Builder — interactive mode. Ctrl+C to exit.")
    try:
        while True:
            question = input("\nyou> ").strip()
            if not question:
                continue
            answer = ask(doc_store, window, llm_client, question)
            print(f"\nassistant> {answer}")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
