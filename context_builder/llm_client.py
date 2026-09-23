"""Thin client that dispatches built context to the configured LLM provider."""

from __future__ import annotations

from typing import List

from .config import Config
from .window import Message


class LLMClient:
    def __init__(self, config: Config):
        self.config = config

    def query(self, messages: List[Message]) -> str:
        if self.config.llm_provider == "anthropic":
            return self._query_anthropic(messages)
        if self.config.llm_provider == "openai":
            return self._query_openai(messages)
        raise ValueError(f"Unsupported LLM_PROVIDER: {self.config.llm_provider!r}")

    def _query_anthropic(self, messages: List[Message]) -> str:
        import anthropic

        if not self.config.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

        client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
        system_content = ""
        chat_messages = []
        for message in messages:
            if message["role"] == "system":
                system_content = message["content"]
            else:
                chat_messages.append(message)

        response = client.messages.create(
            model=self.config.anthropic_model,
            system=system_content,
            messages=chat_messages,
            max_tokens=1024,
        )
        return "".join(block.text for block in response.content if block.type == "text")

    def _query_openai(self, messages: List[Message]) -> str:
        from openai import OpenAI

        if not self.config.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        client = OpenAI(api_key=self.config.openai_api_key)
        response = client.chat.completions.create(
            model=self.config.openai_model,
            messages=messages,
        )
        return response.choices[0].message.content or ""
