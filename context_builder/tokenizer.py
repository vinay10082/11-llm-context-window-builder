"""Exact token measurement via tiktoken."""

from __future__ import annotations

from typing import Iterable, Mapping

import tiktoken


class TokenCounter:
    """Wraps a tiktoken encoding for message and text token counting."""

    # Per-message overhead tokens, following OpenAI's chat message accounting
    # convention (role/name/content wrapper tokens).
    TOKENS_PER_MESSAGE = 4
    TOKENS_PER_REPLY = 2

    def __init__(self, encoding_model: str = "cl100k_base"):
        try:
            self._encoding = tiktoken.get_encoding(encoding_model)
        except ValueError:
            try:
                self._encoding = tiktoken.encoding_for_model(encoding_model)
            except KeyError:
                self._encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def count_message_tokens(self, message: Mapping[str, str]) -> int:
        tokens = self.TOKENS_PER_MESSAGE
        tokens += self.count_tokens(message.get("role", ""))
        tokens += self.count_tokens(message.get("content", ""))
        return tokens

    def count_messages_tokens(self, messages: Iterable[Mapping[str, str]]) -> int:
        total = sum(self.count_message_tokens(m) for m in messages)
        return total + self.TOKENS_PER_REPLY

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        encoded = self._encoding.encode(text)
        if len(encoded) <= max_tokens:
            return text
        return self._encoding.decode(encoded[:max_tokens])
