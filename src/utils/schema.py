"""Shared data structures passed between modules."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    doc_id: str
    text: str
    title: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Question:
    qid: str
    question: str
    answer: str
    gold_doc_ids: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    score: float
    title: str = ""
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BaselineResult:
    qid: str
    question: str
    answer: str
    retrieved: list[RetrievedDoc] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    seconds: float = 0.0
    llm_calls: int = 0

    @property
    def retrieved_ids(self) -> list[str]:
        return [d.doc_id for d in self.retrieved]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


Verdict = str


@dataclass
class ClaimVerdict:
    """One atomic claim and the verifier's judgement of it."""

    claim: str
    verdict: Verdict = "unsupported"
    doc_ids: list[str] = field(default_factory=list)
    rationale: str = ""

    @property
    def is_supported(self) -> bool:
        return self.verdict == "supported"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationReport:
    claims: list[ClaimVerdict] = field(default_factory=list)

    @property
    def n_claims(self) -> int:
        return len(self.claims)

    @property
    def n_supported(self) -> int:
        return sum(c.is_supported for c in self.claims)

    @property
    def n_unsupported(self) -> int:
        return sum(c.verdict == "unsupported" for c in self.claims)

    @property
    def support_ratio(self) -> float:
        """Attribution: share of claims backed by a cited passage."""
        return self.n_supported / self.n_claims if self.n_claims else 1.0

    @property
    def hallucination_rate(self) -> float:
        """Share of claims with no supporting passage at all."""
        return self.n_unsupported / self.n_claims if self.n_claims else 0.0

    def unsupported_claims(self) -> list[str]:
        return [c.claim for c in self.claims if c.verdict != "supported"]

    def to_dict(self) -> dict[str, Any]:
        return {"claims": [c.to_dict() for c in self.claims]}


@dataclass
class Iteration:
    """One pass round the loop, kept so the N-iteration ablation can be
    recomputed later without re-running the experiment."""

    index: int
    query: str
    retrieved_ids: list[str]
    answer: str
    report: VerificationReport | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "query": self.query,
            "retrieved_ids": self.retrieved_ids, "answer": self.answer,
            "report": self.report.to_dict() if self.report else None,
            "seconds": self.seconds,
        }


@dataclass
class LoopResult:
    qid: str
    question: str
    answer: str
    retrieved_ids: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    iterations: list[Iteration] = field(default_factory=list)
    stop_reason: str = ""
    seconds: float = 0.0
    llm_calls: int = 0
    final_report: VerificationReport | None = None

    @property
    def n_iterations(self) -> int:
        return len(self.iterations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid, "question": self.question, "answer": self.answer,
            "retrieved_ids": self.retrieved_ids, "citations": self.citations,
            "iterations": [it.to_dict() for it in self.iterations],
            "stop_reason": self.stop_reason, "seconds": self.seconds,
            "llm_calls": self.llm_calls,
        }
