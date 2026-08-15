"""Self-reflective RAG: retrieve, generate, verify, refine - under a budget.

Identical retriever, generator and prompt to the baseline. The only difference is
the loop, so any measured gain is attributable to verification and nothing else.

Stopping rules, in priority order:
  1. support ratio >= min_support_ratio          -> "verified"
  2. no unsupported claims remain                -> "verified"
  3. unsupported count did not fall this round   -> "no_improvement"
  4. retrieval returned no unseen passages       -> "no_new_evidence"
  5. iteration count reached max_iterations      -> "max_iterations"

Rule 3 is what controls cost: without it the loop spends its whole budget re-asking
for evidence the corpus does not contain.
"""
from __future__ import annotations

import re
import time
from typing import Sequence

from ..generator.llm import (
    FACT_VERIFICATION_SYSTEM, FACT_VERIFICATION_TEMPLATE, GENERATOR_SYSTEM,
    GENERATOR_TEMPLATE, BaseLLM, format_evidence,
)
from ..retrieval.retriever import DenseRetriever
from ..utils.schema import Iteration, LoopResult, RetrievedDoc
from .refine import QueryReformulator
from .verifier import Verifier

_CITE = re.compile(r"\[([^\]]+)\]")


class SelfReflectiveRAG:
    name = "self_reflective"

    def __init__(self, retriever: DenseRetriever, llm: BaseLLM, verifier: Verifier,
                 reformulator: QueryReformulator, top_k: int = 5,
                 max_iterations: int = 3, min_support_ratio: float = 1.0,
                 stop_on_no_improvement: bool = True, expand_k_each_iteration: int = 3,
                 max_new_tokens: int = 320, temperature: float = 0.0) -> None:
        self.retriever = retriever
        self.llm = llm
        self.verifier = verifier
        self.reformulator = reformulator
        self.top_k = top_k
        self.max_iterations = max_iterations
        self.min_support_ratio = min_support_ratio
        self.stop_on_no_improvement = stop_on_no_improvement
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
                                 temperature=self.temperature).strip()

    def run(self, question: str, qid: str = "", on_iteration=None,
           task: str = "qa") -> LoopResult:
        """Run the loop. `on_iteration(Iteration)` is called after each round, which
        the interactive mode uses to show progress live."""
        t0 = time.perf_counter()
        calls_before = self.llm.n_calls
        query = question
        pool: dict[str, RetrievedDoc] = {}
        iterations: list[Iteration] = []
        best = (-1.0, "", [], [], None)
        prev_unsupported: int | None = None
        stop_reason = "max_iterations"

        for i in range(self.max_iterations):
            it_start = time.perf_counter()
            k = self.top_k + i * self.expand_k
            fresh = self.retriever.retrieve(query, k=k, exclude=set(pool) if i else ())
            if i and not fresh:
                stop_reason = "no_new_evidence"
                break
            for d in fresh:
                pool.setdefault(d.doc_id, d)

            docs = sorted(pool.values(), key=lambda d: -d.score)[:k]
            answer = self._generate(question, docs, task=task)
            report = self.verifier.run(answer, docs)

            valid = {d.doc_id for d in docs}
            cited = [c.strip() for m in _CITE.findall(answer) for c in m.split(",")]
            citations = [c for c in dict.fromkeys(cited) if c in valid]
            doc_ids = [d.doc_id for d in docs]

            it = Iteration(i, query, doc_ids, answer, report,
                           time.perf_counter() - it_start)
            iterations.append(it)
            if on_iteration:
                on_iteration(it)

            if report.support_ratio > best[0]:
                best = (report.support_ratio, answer, doc_ids, citations, report)

            if report.support_ratio >= self.min_support_ratio or not report.unsupported_claims():
                stop_reason = "verified"
                break

            n_unsup = len(report.unsupported_claims())
            if (self.stop_on_no_improvement and prev_unsupported is not None
                    and n_unsup >= prev_unsupported):
                stop_reason = "no_improvement"
                break
            prev_unsupported = n_unsup

            if i == self.max_iterations - 1:
                break
            queries = self.reformulator.reformulate(question,
                                                    report.unsupported_claims(), docs)
            if not queries:
                stop_reason = "no_new_query"
                break
            query = queries[0]

        _, answer, doc_ids, citations, best_report = best
        return LoopResult(
            qid=qid, question=question, answer=answer, retrieved_ids=doc_ids,
            citations=citations, iterations=iterations, stop_reason=stop_reason,
            seconds=time.perf_counter() - t0,
            llm_calls=self.llm.n_calls - calls_before,
            final_report=best_report,
        )
