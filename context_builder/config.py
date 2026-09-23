"""Environment-driven configuration for the context window builder."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


@dataclass(frozen=True)
class Config:
    max_context_tokens: int
    sliding_window_message_cap: int
    tiktoken_encoding_model: str
    llm_provider: str
    anthropic_api_key: str
    openai_api_key: str
    anthropic_model: str
    openai_model: str
    system_prompt: str

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            max_context_tokens=_env_int("MAX_CONTEXT_TOKENS", 8000),
            sliding_window_message_cap=_env_int("SLIDING_WINDOW_MESSAGE_CAP", 50),
            tiktoken_encoding_model=os.getenv("TIKTOKEN_ENCODING_MODEL", "cl100k_base"),
            llm_provider=os.getenv("LLM_PROVIDER", "anthropic").lower(),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            system_prompt=os.getenv(
                "SYSTEM_PROMPT",
                "You are a helpful assistant with access to the provided document context.",
            ),
        )
