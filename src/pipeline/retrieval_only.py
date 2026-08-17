"""Retrieval-only ablation (ISSUE 2): isolates "more evidence" from "verification"
as the source of any gain SelfReflectiveRAG shows over BaselineRAG.

SelfReflectiveRAG's later iterations both (a) expand the retrieval pool with a larger
k and (b) reformulate the query using the verifier's unsupported-claim findings. If
self-reflective beats baseline, that could be entirely explained by (a) - simply
seeing more evidence - with the verification loop in (b) contributing nothing. This
class runs exactly the same retrieval-expansion schedule (same top_k, same
expand_k_each_iteration, same corpus, same generator) as SelfReflectiveRAG, but with
the query held fixed at the original question every round (no verifier, no
claim-level judgement, no targeted reformulation) - so any gap between this arm and
baseline isolates retrieval breadth alone, and any further gap between self-reflective
and this arm isolates verification's marginal contribution beyond that.

Budget matching: retrieval-only receives the exact same top_k/expand_k schedule as
self-reflective, and by default runs for `max_iterations` rounds (or until a round
returns no new passages, "no_new_evidence" - the one stopping rule that doesn't
require verification). Because self-reflective often stops earlier than
`max_iterations` once it's satisfied, callers that want a strict per-question budget
match should pass `max_iterations=<self_reflective_result.n_iterations>` explicitly
(see scripts/run_subset_experiment.py) - the actual number of retrieval rounds and
passages used is always recorded per question, in `LoopResult.iterations`, so the
true budget is verifiable either way rather than assumed.
"""
from __future__ import annotations

import time
from typing import Sequence

from ..generator.llm import (
    FACT_VERIFICATION_SYSTEM, FACT_VERIFICATION_TEMPLATE, GENERATOR_SYSTEM,
    GENERATOR_TEMPLATE, BaseLLM, extract_citations, format_evidence,
)
from ..retrieval.retriever import DenseRetriever
from ..utils.schema import Iteration, LoopResult, RetrievedDoc


class RetrievalOnlyRAG:
    """Retrieve with an expanding budget, regenerate each round, never verify."""

    name = "retrieval_only"

    def __init__(self, retriever: DenseRetriever, llm: BaseLLM, top_k: int = 5,
                 max_iterations: int = 3, expand_k_each_iteration: int = 3,
                 max_new_tokens: int = 320, temperature: float = 0.0) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.max_iterations = max_iterations
        self.expand_k = expand_k_each_iteration
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _generate(self, question: str, docs: Sequence[RetrievedDoc],
                  task: str = "qa") -> str:
        system, template = (
            (FACT_VERIFICATION_SYSTEM, FACT_VERIFICATION_TEMPLATE) if task == "fact_verification"
            else (GENERATOR_SYSTEM, GENERATOR_TEMPLATE)
        )
        prompt = template.format(
            evidence=format_evidence(docs), question=question,
            example_id=docs[0].doc_id if docs else "doc_id")
        return self.llm.complete(prompt, system=system,
                                 max_tokens=self.max_new_tokens,
                                 temperature=self.temperature,
                                 purpose="generation").strip()

    def run(self, question: str, qid: str = "", on_iteration=None,
           task: str = "qa", max_iterations: int | None = None) -> LoopResult:
        """Takes only the question text - no gold answer or gold evidence is ever in
        scope, structurally. `max_iterations`, if given, overrides the instance
        default for this call only - used to budget-match a specific
        self-reflective run's actual iteration count (see module docstring)."""
        rounds = self.max_iterations if max_iterations is None else max(1, max_iterations)
        t0 = time.perf_counter()
        purposes_before = dict(self.llm.calls_by_purpose)
        pool: dict[str, RetrievedDoc] = {}
        iterations: list[Iteration] = []
        answer = ""
        doc_ids: list[str] = []
        citations: list[str] = []
        docs: list[RetrievedDoc] = []
        stop_reason = "max_iterations"

        for i in range(rounds):
            it_start = time.perf_counter()
            k = self.top_k + i * self.expand_k
            fresh = self.retriever.retrieve(question, k=k, exclude=set(pool) if i else ())
            if i and not fresh:
                stop_reason = "no_new_evidence"
                break
            for d in fresh:
                pool.setdefault(d.doc_id, d)

            docs = sorted(pool.values(), key=lambda d: -d.score)[:k]
            answer = self._generate(question, docs, task=task)
            citations = extract_citations(answer, {d.doc_id for d in docs})
            doc_ids = [d.doc_id for d in docs]
            purposes_now = dict(self.llm.calls_by_purpose)

            it = Iteration(
                i, question, doc_ids, answer, report=None,
                seconds=time.perf_counter() - it_start, citations=citations,
                llm_calls_so_far=sum(purposes_now.values()) - sum(purposes_before.values()),
                calls_by_purpose_so_far={
                    k2: purposes_now.get(k2, 0) - purposes_before.get(k2, 0)
                    for k2 in purposes_now
                    if purposes_now.get(k2, 0) - purposes_before.get(k2, 0)
                })
            iterations.append(it)
            if on_iteration:
                on_iteration(it)
            if i == rounds - 1:
                stop_reason = "max_iterations"

        purposes_after = dict(self.llm.calls_by_purpose)
        return LoopResult(
            qid=qid, question=question, answer=answer, retrieved_ids=doc_ids,
            citations=citations, iterations=iterations, stop_reason=stop_reason,
            seconds=time.perf_counter() - t0,
            llm_calls=sum(purposes_after.values()) - sum(purposes_before.values()),
            calls_by_purpose={
                k: purposes_after.get(k, 0) - purposes_before.get(k, 0)
                for k in purposes_after
                if purposes_after.get(k, 0) - purposes_before.get(k, 0)
            },
            final_report=None, arm=self.name,
            generator_model=self.llm.name, verifier_model="", retrieved=docs,
        )
