"""Loading and formatting of documents injected into the context window."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .tokenizer import TokenCounter


@dataclass
class Document:
    source: str
    content: str


@dataclass
class DocumentStore:
    """Holds documents to be injected as context and renders them to text."""

    token_counter: TokenCounter
    documents: List[Document] = field(default_factory=list)

    def add_text(self, source: str, content: str) -> Document:
        doc = Document(source=source, content=content)
        self.documents.append(doc)
        return doc

    def add_file(self, path: str | Path) -> Document:
        file_path = Path(path)
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return self.add_text(source=file_path.name, content=content)

    def clear(self) -> None:
        self.documents.clear()

    def render(self, max_tokens: int | None = None) -> str:
        """Render all documents into a single labeled block.

        If max_tokens is given, documents are truncated (in insertion order)
        so the rendered block fits within the budget.
        """
        blocks = []
        remaining = max_tokens
        for doc in self.documents:
            header = f"--- Document: {doc.source} ---\n"
            body = doc.content
            if remaining is not None:
                header_tokens = self.token_counter.count_tokens(header)
                budget = remaining - header_tokens
                if budget <= 0:
                    break
                body = self.token_counter.truncate_to_tokens(body, budget)
                remaining -= header_tokens + self.token_counter.count_tokens(body)
            blocks.append(header + body)
            if remaining is not None and remaining <= 0:
                break
        return "\n\n".join(blocks)
