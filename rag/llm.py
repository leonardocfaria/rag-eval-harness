"""Pluggable LLM interface + offline mocks. A real backend only needs answer(question, contexts) -> str."""
import re
from typing import Protocol

from .retrievers import tokenize


class LLM(Protocol):
    def answer(self, question: str, contexts: list[tuple[str, str]]) -> str:
        """contexts: [(doc_id, text)] in rank order. Should cite sources as [doc_id]."""


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class MockLLM:
    """Extractive: returns the context sentence with the most question-token overlap, with its citation."""

    def answer(self, question, contexts):
        q = set(tokenize(question))
        best = max(((len(q & set(tokenize(s))), -rank, s, doc_id)
                    for rank, (doc_id, text) in enumerate(contexts)
                    for s in sentences(text.split(": ", 1)[-1])), default=None)
        return f"{best[2]} [{best[3]}]" if best else "I don't know."


class UnfaithfulMockLLM:
    """Ignores the context and states something unsupported: the faithfulness checker must catch it."""

    def answer(self, question, contexts):
        cite = contexts[0][0] if contexts else "none"
        return f"This API was deprecated in Python 2.4 and replaced by a faster C implementation. [{cite}]"
