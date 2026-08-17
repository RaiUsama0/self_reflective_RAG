"""Claim-level verification - the project's contribution.

Two stages:
  1. decompose - split an answer into atomic, self-contained claims.
  2. verify    - judge each claim against the retrieved passages.

Decisions that keep the attribution metric honest:
  * Verdicts are three-way. Folding "partial" into "supported" inflates attribution;
    folding it into "unsupported" makes the loop churn on hedged phrasing.
  * A verdict citing a passage that is not in context is rejected - the verifier
    cannot have read it.
  * A "supported" verdict naming no passage is downgraded to "partial": support that
    cannot be pointed at is not attribution.
  * Anything malformed defaults to "unsupported". Errors push the score down, never up.
"""
from __future__ import annotations

import re
from typing import Sequence

from ..generator.llm import (
    DECOMPOSE_SYSTEM, DECOMPOSE_TEMPLATE, VERIFIER_SYSTEM, VERIFIER_TEMPLATE,
    BaseLLM, extract_json, format_evidence,
)
from ..utils.schema import ClaimVerdict, RetrievedDoc, VerificationReport

_VALID = {"supported", "partial", "unsupported"}
_CITE = re.compile(r"\s*\[[^\]]*\]")
_SENT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Fallback decomposition when the model returns unusable JSON."""
    return [s.strip() for s in _SENT.split(_CITE.sub("", text)) if len(s.strip()) > 3]


class Verifier:
    def __init__(self, llm: BaseLLM, max_new_tokens: int = 512,
                 temperature: float = 0.0) -> None:
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def decompose(self, answer: str) -> list[str]:
        if not answer.strip():
            return []
        raw = self.llm.complete(
            DECOMPOSE_TEMPLATE.format(answer=answer), system=DECOMPOSE_SYSTEM,
            max_tokens=self.max_new_tokens, temperature=self.temperature,
            purpose="decompose")
        data = extract_json(raw, default=None)
        claims: list[str] = []
        if isinstance(data, dict):
            claims = [str(c).strip() for c in data.get("claims", []) if str(c).strip()]
        elif isinstance(data, list):
            claims = [str(c).strip() for c in data if str(c).strip()]
        return claims or split_sentences(answer)

    def verify(self, claims: Sequence[str],
               docs: Sequence[RetrievedDoc]) -> VerificationReport:
        if not claims:
            return VerificationReport(claims=[])
        numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
        raw = self.llm.complete(
            VERIFIER_TEMPLATE.format(evidence=format_evidence(docs), claims=numbered),
            system=VERIFIER_SYSTEM, max_tokens=self.max_new_tokens,
            temperature=self.temperature, purpose="verify")
        data = extract_json(raw, default=None)
        items = data.get("verdicts", []) if isinstance(data, dict) else (data or [])
        valid_ids = {d.doc_id for d in docs}

        by_claim = {
            str(it["claim"]).strip().lower(): it
            for it in items if isinstance(it, dict) and it.get("claim")
        }

        out: list[ClaimVerdict] = []
        for i, claim in enumerate(claims):
            item = by_claim.get(claim.strip().lower())
            if item is None and i < len(items) and isinstance(items[i], dict):
                item = items[i]
            if item is None:
                out.append(ClaimVerdict(claim, "unsupported", [], "no verdict returned"))
                continue
            verdict = str(item.get("verdict", "unsupported")).strip().lower()
            if verdict not in _VALID:
                verdict = "unsupported"
            ids = [str(d) for d in (item.get("doc_ids") or []) if str(d) in valid_ids]
            if verdict == "supported" and not ids:
                verdict = "partial"
            out.append(ClaimVerdict(claim, verdict, ids,
                                    str(item.get("rationale", "")).strip()))
        return VerificationReport(claims=out)

    def run(self, answer: str, docs: Sequence[RetrievedDoc]) -> VerificationReport:
        return self.verify(self.decompose(answer), docs)
