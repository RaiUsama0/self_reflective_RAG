"""Query reformulation: turn unsupported claims into targeted follow-up queries."""
from __future__ import annotations

from typing import Sequence

from ..generator.llm import (
    REFORMULATE_SYSTEM, REFORMULATE_TEMPLATE, BaseLLM, extract_json,
)
from ..utils.schema import RetrievedDoc


class QueryReformulator:
    def __init__(self, llm: BaseLLM, n_queries: int = 2, max_new_tokens: int = 256,
                 temperature: float = 0.0) -> None:
        self.llm = llm
        self.n_queries = n_queries
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def reformulate(self, question: str, unsupported: Sequence[str],
                    seen: Sequence[RetrievedDoc]) -> list[str]:
        if not unsupported:
            return []
        titles = ", ".join(d.title or d.doc_id for d in seen) or "(none)"
        raw = self.llm.complete(
            REFORMULATE_TEMPLATE.format(
                question=question,
                unsupported="\n".join(f"- {c}" for c in unsupported),
                seen_titles=titles, n=self.n_queries),
            system=REFORMULATE_SYSTEM, max_tokens=self.max_new_tokens,
            temperature=self.temperature, purpose="reformulate")
        data = extract_json(raw, default=None)
        queries: list[str] = []
        if isinstance(data, dict):
            queries = [str(q).strip() for q in data.get("queries", []) if str(q).strip()]
        elif isinstance(data, list):
            queries = [str(q).strip() for q in data if str(q).strip()]
        if not queries:
            queries = [f"{question} {c}" for c in unsupported[: self.n_queries]]
        return queries[: self.n_queries]
