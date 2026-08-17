"""Offline tests for the three experimental arms (BaselineRAG, RetrievalOnlyRAG,
SelfReflectiveRAG), independent verifier configuration, gold-leakage prevention, cost
tracking, and the loop's stopping rules.

Everything here runs without network access or an API key - all LLM roles are played
by RecordingLLM, a test double that returns scripted, trigger-matched responses and
logs every prompt it receives (used by TestNoGoldLeakage to prove gold information
never reaches a model call).

    python tests/test_arms.py
"""
from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path
from typing import Callable

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator.llm import APPROX_PRICING_USD_PER_1M, BaseLLM
from src.pipeline.baseline import BaselineRAG
from src.pipeline.refine import QueryReformulator
from src.pipeline.retrieval_only import RetrievalOnlyRAG
from src.pipeline.self_reflective import SelfReflectiveRAG
from src.pipeline.verifier import Verifier
from src.retrieval.retriever import DenseRetriever
from src.utils.schema import Document, Question


_DOC_ID = re.compile(r"\[([A-Za-z0-9_\-]+)\]")


def first_doc_id(prompt: str) -> str:
    m = _DOC_ID.search(prompt)
    return m.group(1) if m else "doc0"


class RecordingLLM(BaseLLM):
    """Trigger-matched scripted responses, plus a full log of every prompt/system/
    purpose it was called with - the log is what TestNoGoldLeakage inspects."""

    def __init__(self, name: str = "test-llm") -> None:
        super().__init__()
        self.name = name
        self.handlers: list[tuple[str, Callable[[str], str]]] = []
        self.log: list[dict] = []

    def register(self, trigger: str, handler: Callable[[str], str]) -> "RecordingLLM":
        self.handlers.append((trigger, handler))
        return self

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 512,
                 temperature: float = 0.0, purpose: str = "unspecified") -> str:
        self.log.append({"purpose": purpose, "system": system or "", "prompt": prompt})
        self.n_calls += 1
        self.calls_by_purpose[purpose] = self.calls_by_purpose.get(purpose, 0) + 1
        for trigger, handler in self.handlers:
            if trigger in prompt:
                return handler(prompt)
        return ""

    def _complete(self, prompt, system, max_tokens, temperature):
        raise NotImplementedError("RecordingLLM overrides complete() directly")


class FakeMeteredLLM(BaseLLM):
    """Returns a fixed token usage per call, to test cost estimation without a real
    API call."""

    def __init__(self, name: str, prompt_tokens: int, completion_tokens: int) -> None:
        super().__init__()
        self.name = name
        self._pt, self._ct = prompt_tokens, completion_tokens

    def _complete(self, prompt, system, max_tokens, temperature):
        return "ok", {"prompt_tokens": self._pt, "completion_tokens": self._ct}


def make_corpus_retriever(n_docs: int) -> DenseRetriever:
    docs = [Document(doc_id=f"d{i}", title=f"Doc {i}",
                     text=f"This is document number {i} about topic {i}. "
                          f"It contains unique fact {i} for retrieval testing.")
            for i in range(n_docs)]
    return DenseRetriever.build(docs, embedder_name="tfidf", backend="numpy", seed=13)


def register_common(llm: RecordingLLM, claims: list[str] | None = None,
                    queries: list[str] | None = None) -> None:
    claims = claims or ["Fact one is true"]
    queries = queries or ["alternate phrasing of fact one"]
    claims_json = '{"claims": [' + ", ".join(f'"{c}"' for c in claims) + ']}'
    queries_json = '{"queries": [' + ", ".join(f'"{q}"' for q in queries) + ']}'
    llm.register("Write a short answer",
                 lambda p: f"{claims[0]} [{first_doc_id(p)}].")
    llm.register("Split the following", lambda p: claims_json)
    llm.register("Write up to", lambda p: queries_json)


def verdict_json(claim: str, verdict: str, doc_id: str | None) -> str:
    ids = f'["{doc_id}"]' if doc_id else "[]"
    return ('{"verdicts": [{"claim": "%s", "verdict": "%s", "doc_ids": %s, '
            '"rationale": "r"}]}' % (claim, verdict, ids))


class TestBaselineRAG(unittest.TestCase):
    def test_single_generation_call_only(self):
        retriever = make_corpus_retriever(5)
        llm = RecordingLLM("gen-model")
        register_common(llm)
        system = BaselineRAG(retriever, llm, top_k=3)

        result = system.run("What is fact 1?", qid="q1")

        self.assertEqual(result.llm_calls, 1)
        self.assertEqual(result.calls_by_purpose, {"generation": 1})
        self.assertEqual(result.generator_model, "gen-model")
        self.assertEqual(len(result.retrieved), 3)
        self.assertIn(result.citations[0], {d.doc_id for d in result.retrieved})


class TestRetrievalOnlyRAG(unittest.TestCase):
    def test_expands_k_and_makes_one_generation_call_per_round(self):
        retriever = make_corpus_retriever(12)
        llm = RecordingLLM("gen-model")
        register_common(llm)
        system = RetrievalOnlyRAG(retriever, llm, top_k=2, expand_k_each_iteration=2,
                                  max_iterations=3)

        result = system.run("What is fact 1?", qid="q1")

        self.assertEqual(result.arm, "retrieval_only")
        self.assertEqual(result.n_iterations, 3)
        self.assertEqual(result.llm_calls, 3)
        self.assertEqual(set(result.calls_by_purpose), {"generation"})
        self.assertEqual([len(it.retrieved_ids) for it in result.iterations], [2, 4, 6])
        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertEqual(result.verifier_model, "")

    def test_no_new_evidence_stops_early(self):
        retriever = make_corpus_retriever(2)
        llm = RecordingLLM("gen-model")
        register_common(llm)
        system = RetrievalOnlyRAG(retriever, llm, top_k=2, expand_k_each_iteration=2,
                                  max_iterations=3)

        result = system.run("q", qid="q1")

        self.assertEqual(result.stop_reason, "no_new_evidence")
        self.assertEqual(result.n_iterations, 1)

    def test_budget_matching_override(self):
        """A caller (the experiment script) can cap retrieval-only at exactly the
        number of rounds self-reflective actually used for this question."""
        retriever = make_corpus_retriever(12)
        llm = RecordingLLM("gen-model")
        register_common(llm)
        system = RetrievalOnlyRAG(retriever, llm, top_k=2, expand_k_each_iteration=2,
                                  max_iterations=3)

        result = system.run("q", qid="q1", max_iterations=1)

        self.assertEqual(result.n_iterations, 1)
        self.assertEqual(result.llm_calls, 1)


class TestSelfReflectiveStoppingRules(unittest.TestCase):
    def _build(self, gen_llm=None, verifier_llm=None, top_k=2, expand_k=2,
              max_iterations=3, min_support_ratio=1.0, n_docs=12):
        retriever = make_corpus_retriever(n_docs)
        gen_llm = gen_llm or RecordingLLM("gen-model")
        verifier_llm = verifier_llm if verifier_llm is not None else gen_llm
        register_common(gen_llm)
        verifier = Verifier(verifier_llm)
        reformulator = QueryReformulator(gen_llm)
        system = SelfReflectiveRAG(retriever, gen_llm, verifier, reformulator,
                                   top_k=top_k, expand_k_each_iteration=expand_k,
                                   max_iterations=max_iterations,
                                   min_support_ratio=min_support_ratio)
        return system, gen_llm, verifier_llm

    def test_verified_stops_after_first_round(self):
        verifier_llm = RecordingLLM("verifier-model")
        verifier_llm.register(
            "Claims to verify",
            lambda p: verdict_json("Fact one is true", "supported", first_doc_id(p)))
        gen_llm = RecordingLLM("gen-model")
        system, gen_llm, verifier_llm = self._build(gen_llm=gen_llm, verifier_llm=verifier_llm)

        result = system.run("q", qid="q1")

        self.assertEqual(result.stop_reason, "verified")
        self.assertEqual(result.n_iterations, 1)
        self.assertEqual(result.generator_model, "gen-model")
        self.assertEqual(result.verifier_model, "verifier-model")

    def test_no_improvement_stops_when_unsupported_count_plateaus(self):
        verifier_llm = RecordingLLM("verifier-model")
        verifier_llm.register(
            "Claims to verify",
            lambda p: verdict_json("Fact one is true", "unsupported", None))
        gen_llm = RecordingLLM("gen-model")
        system, *_ = self._build(gen_llm=gen_llm, verifier_llm=verifier_llm)

        result = system.run("q", qid="q1")

        self.assertEqual(result.stop_reason, "no_improvement")
        self.assertEqual(result.n_iterations, 2)

    def test_no_new_evidence_stops_when_corpus_exhausted(self):
        verifier_llm = RecordingLLM("verifier-model")
        verifier_llm.register(
            "Claims to verify",
            lambda p: verdict_json("Fact one is true", "unsupported", None))
        gen_llm = RecordingLLM("gen-model")
        system, *_ = self._build(gen_llm=gen_llm, verifier_llm=verifier_llm,
                                 top_k=2, n_docs=2)

        result = system.run("q", qid="q1")

        self.assertEqual(result.stop_reason, "no_new_evidence")
        self.assertEqual(result.n_iterations, 1)

    def test_max_iterations_stops_the_loop_when_support_keeps_improving_but_never_completes(self):
        claims = ["claim a", "claim b", "claim c", "claim d"]
        counter = {"n": 0}

        def verify_handler(p: str) -> str:
            n_supported = counter["n"] + 1
            counter["n"] += 1
            items = []
            for i, c in enumerate(claims):
                verdict = "supported" if i < n_supported else "unsupported"
                items.append('{"claim": "%s", "verdict": "%s", "doc_ids": %s}' % (
                    c, verdict, f'["{first_doc_id(p)}"]' if verdict == "supported" else "[]"))
            return '{"verdicts": [' + ", ".join(items) + "]}"

        claims_json = '{"claims": [' + ", ".join(f'"{c}"' for c in claims) + ']}'
        verifier_llm = RecordingLLM("verifier-model")
        verifier_llm.register("Claims to verify", verify_handler)
        verifier_llm.register("Split the following", lambda p: claims_json)
        gen_llm = RecordingLLM("gen-model")
        register_common(gen_llm, claims=claims)
        system, *_ = self._build(gen_llm=gen_llm, verifier_llm=verifier_llm,
                                 max_iterations=3, min_support_ratio=1.0)

        result = system.run("q", qid="q1")

        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertEqual(result.n_iterations, 3)


class TestVerifierIndependence(unittest.TestCase):
    def test_independent_verifier_records_distinct_models_and_call_counts(self):
        gen_llm = RecordingLLM("gen-model")
        verifier_llm = RecordingLLM("verifier-model")
        verifier_llm.register(
            "Claims to verify",
            lambda p: verdict_json("Fact one is true", "supported", first_doc_id(p)))
        retriever = make_corpus_retriever(6)
        register_common(gen_llm)
        system = SelfReflectiveRAG(retriever, gen_llm, Verifier(verifier_llm),
                                   QueryReformulator(gen_llm), top_k=3)

        result = system.run("q", qid="q1")

        self.assertNotEqual(result.generator_model, result.verifier_model)
        self.assertEqual(result.generator_model, "gen-model")
        self.assertEqual(result.verifier_model, "verifier-model")
        self.assertEqual(gen_llm.calls_by_purpose.get("decompose", 0), 0)
        self.assertEqual(gen_llm.calls_by_purpose.get("verify", 0), 0)
        self.assertGreaterEqual(verifier_llm.calls_by_purpose.get("verify", 0), 1)
        self.assertGreaterEqual(verifier_llm.calls_by_purpose.get("decompose", 0), 1)
        self.assertEqual(result.llm_calls, gen_llm.n_calls + verifier_llm.n_calls)

    def test_same_model_verifier_does_not_double_count_a_shared_instance(self):
        shared = RecordingLLM("gen-model")
        shared.register(
            "Claims to verify",
            lambda p: verdict_json("Fact one is true", "supported", first_doc_id(p)))
        retriever = make_corpus_retriever(6)
        register_common(shared)
        system = SelfReflectiveRAG(retriever, shared, Verifier(shared),
                                   QueryReformulator(shared), top_k=3)

        result = system.run("q", qid="q1")

        self.assertEqual(result.generator_model, result.verifier_model)
        self.assertEqual(shared.n_calls, 3)
        self.assertEqual(result.llm_calls, 3)


class TestNoGoldLeakage(unittest.TestCase):
    """Constructs a Question whose gold answer and gold doc id are distinctive
    sentinel strings that do not appear anywhere in the retrieved corpus, runs every
    arm passing only q.question/q.qid/task (never q.answer or q.gold_doc_ids), and
    asserts the sentinels never appear in any prompt any model was actually called
    with."""

    SENTINEL_ANSWER = "SENTINEL_GOLD_ANSWER_9f3a21"
    SENTINEL_DOC = "SENTINEL_GOLD_DOC_9f3a21"

    def _question(self) -> Question:
        return Question(qid="q1", question="What is fact 1?",
                        answer=self.SENTINEL_ANSWER, gold_doc_ids=[self.SENTINEL_DOC])

    def _assert_clean(self, *llms: RecordingLLM) -> None:
        for llm in llms:
            for entry in llm.log:
                self.assertNotIn(self.SENTINEL_ANSWER, entry["prompt"])
                self.assertNotIn(self.SENTINEL_DOC, entry["prompt"])
                self.assertNotIn(self.SENTINEL_ANSWER, entry["system"])
                self.assertNotIn(self.SENTINEL_DOC, entry["system"])

    def test_baseline_never_sees_gold(self):
        retriever = make_corpus_retriever(5)
        llm = RecordingLLM("gen-model")
        register_common(llm)
        q = self._question()

        BaselineRAG(retriever, llm, top_k=3).run(q.question, qid=q.qid,
                                                  task=q.meta.get("task", "qa"))

        self._assert_clean(llm)

    def test_retrieval_only_never_sees_gold(self):
        retriever = make_corpus_retriever(8)
        llm = RecordingLLM("gen-model")
        register_common(llm)
        q = self._question()

        RetrievalOnlyRAG(retriever, llm, top_k=2, max_iterations=2).run(
            q.question, qid=q.qid, task=q.meta.get("task", "qa"))

        self._assert_clean(llm)

    def test_self_reflective_never_sees_gold_even_with_independent_verifier(self):
        retriever = make_corpus_retriever(8)
        gen_llm = RecordingLLM("gen-model")
        verifier_llm = RecordingLLM("verifier-model")
        verifier_llm.register(
            "Claims to verify",
            lambda p: verdict_json("Fact one is true", "unsupported", None))
        register_common(gen_llm)
        q = self._question()

        SelfReflectiveRAG(retriever, gen_llm, Verifier(verifier_llm),
                          QueryReformulator(gen_llm), top_k=2, max_iterations=2).run(
            q.question, qid=q.qid, task=q.meta.get("task", "qa"))

        self._assert_clean(gen_llm, verifier_llm)


class TestCostTracking(unittest.TestCase):
    def test_known_model_estimates_cost_from_tracked_tokens(self):
        llm = FakeMeteredLLM("openai:gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
        llm.complete("p", purpose="generation")
        llm.complete("p", purpose="generation")

        cost = llm.estimated_cost_usd()
        in_price, out_price = APPROX_PRICING_USD_PER_1M["gpt-4o-mini"]
        expected = (2000 * in_price + 1000 * out_price) / 1e6
        self.assertAlmostEqual(cost, expected, places=8)
        self.assertEqual(llm.calls_by_purpose, {"generation": 2})

    def test_unknown_model_reports_no_cost(self):
        llm = FakeMeteredLLM("some-unpriced-local-model", prompt_tokens=100, completion_tokens=50)
        llm.complete("p", purpose="generation")
        self.assertIsNone(llm.estimated_cost_usd())

    def test_usage_report_breaks_down_by_purpose(self):
        llm = FakeMeteredLLM("openai:gpt-4o-mini", prompt_tokens=10, completion_tokens=5)
        llm.complete("p", purpose="generation")
        llm.complete("p", purpose="verify")
        llm.complete("p", purpose="verify")
        report = llm.usage_report()
        self.assertEqual(report["calls_by_purpose"], {"generation": 1, "verify": 2})
        self.assertEqual(report["n_calls"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
