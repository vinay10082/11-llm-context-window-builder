"""Sliding window memory mechanism with token-aware eviction."""

from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Dict, List, Optional

from .tokenizer import TokenCounter

Message = Dict[str, str]
EvictionCallback = Callable[[Message], None]


class SlidingWindowContextManager:
    """Maintains a token-bounded sliding window of chat messages.

    The system prompt is preserved separately and is always included in
    the built context, independent of the sliding window eviction.
    """

    def __init__(
        self,
        token_counter: TokenCounter,
        max_context_tokens: int,
        message_cap: int,
        system_prompt: str = "",
        on_evict: Optional[EvictionCallback] = None,
    ):
        self.token_counter = token_counter
        self.max_context_tokens = max_context_tokens
        self.message_cap = message_cap
        self.system_prompt = system_prompt
        self.on_evict = on_evict
        self._window: Deque[Message] = deque(maxlen=None)

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    def add_message(self, role: str, content: str) -> None:
        message: Message = {"role": role, "content": content}
        self._window.append(message)
        self._evict_by_cap()

    def _evict_by_cap(self) -> None:
        while len(self._window) > self.message_cap:
            evicted = self._window.popleft()
            if self.on_evict:
                self.on_evict(evicted)

    def _system_message(self) -> Optional[Message]:
        if not self.system_prompt:
            return None
        return {"role": "system", "content": self.system_prompt}

    def build_context(self, extra_context: str = "") -> List[Message]:
        """Return messages that fit within the token budget.

        System prompt (plus any extra_context, such as injected documents)
        is always preserved. Oldest window messages are evicted first when
        the budget is exceeded.
        """
        system_content = self.system_prompt
        if extra_context:
            system_content = f"{system_content}\n\n{extra_context}".strip()
        system_message = {"role": "system", "content": system_content} if system_content else None

        reserved = self.token_counter.count_message_tokens(system_message) if system_message else 0
        budget = self.max_context_tokens - reserved

        used = sum(self.token_counter.count_message_tokens(m) for m in self._window)
        while used > budget and self._window:
            evicted = self._window.popleft()
            used -= self.token_counter.count_message_tokens(evicted)
            if self.on_evict:
                self.on_evict(evicted)

        context = ([system_message] if system_message else []) + list(self._window)
        return context

    def __len__(self) -> int:
        return len(self._window)
