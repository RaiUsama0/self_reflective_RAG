"""Most tests are offline; TestGenerator.test_call_counting calls the real
OpenAI API and needs OPENAI_API_KEY set.

    python tests/test_pipeline.py
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.data.ingest import chunk_text, load_file_as_documents
from src.data.preprocess import (
    clean_fever_text, make_doc_id, normalise_ws, preprocess_fever,
)
from src.embeddings.embedder import TfidfEmbedder, l2_normalize
from src.generator.llm import (
    BaseLLM, build_llm, extract_json, extract_verdict, format_evidence,
)
from src.pipeline.baseline import (
    aggregate, answer_in_prediction, evaluate_one, exact_match, precision_at_k,
    recall_at_k, token_f1,
)
from src.retrieval.faiss_index import VectorIndex
from src.utils.schema import BaselineResult, Question, RetrievedDoc

import numpy as np


class ScriptedLLM(BaseLLM):
    """Test-only stub: routes prompts to handlers by substring match.

    Used solely to force the exact malformed/adversarial LLM responses that the
    Verifier's defensive parsing is meant to handle (missing doc_ids, citations to
    passages that were never retrieved, missing verdicts, unparseable output) - a
    real model call cannot be made to reliably reproduce those on demand.
    """

    def __init__(self) -> None:
        super().__init__()
        self.handlers: list[tuple[str, Callable[[str], str]]] = []

    def register(self, trigger: str, handler: Callable[[str], str]) -> "ScriptedLLM":
        self.handlers.append((trigger, handler))
        return self

    def _complete(self, prompt: str, system: str | None,
                  max_tokens: int, temperature: float) -> tuple[str, None]:
        for trigger, handler in self.handlers:
            if trigger in prompt:
                return handler(prompt), None
        return "", None


class TestPreprocess(unittest.TestCase):
    def test_doc_ids_and_cleaning(self):
        self.assertEqual(make_doc_id("Marie  Curie"), "Marie_Curie")
        self.assertEqual(clean_fever_text("Paris -LRB- France -RRB-"), "Paris ( France )")
        self.assertEqual(normalise_ws("  a\n b  "), "a b")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    def test_fever_collapses_multi_evidence_rows_into_one_question_per_claim(self):
        """FEVER's raw dump has one row per (claim, valid-evidence-set) pair - the
        same claim id can appear many times with different evidence_wiki_url values.
        Regression test for the bug where every row became its own "question",
        inflating some claims' sampling weight up to ~59x in a real n=200 draw."""
        with tempfile.TemporaryDirectory() as td:
            raw_dir = Path(td) / "raw"
            raw_dir.mkdir()
            self._write_jsonl(raw_dir / "claims_labelled_dev.jsonl", [
                {"id": 1, "claim": "Paris is in France.", "label": "SUPPORTS",
                 "evidence_wiki_url": "Paris"},
                {"id": 1, "claim": "Paris is in France.", "label": "SUPPORTS",
                 "evidence_wiki_url": "France"},
                {"id": 1, "claim": "Paris is in France.", "label": "SUPPORTS",
                 "evidence_wiki_url": "Paris"},
                {"id": 2, "claim": "The sky is green.", "label": "REFUTES",
                 "evidence_wiki_url": "Sky"},
                {"id": 3, "claim": "Unverifiable claim.", "label": "NOT ENOUGH INFO",
                 "evidence_wiki_url": None},
            ])
            self._write_jsonl(raw_dir / "wiki_pages.jsonl", [
                {"title": "Paris", "text": "Paris is the capital of France."},
                {"title": "France", "text": "France is a country in Europe."},
                {"title": "Sky", "text": "The sky is blue during the day."},
            ])

            out_dir = preprocess_fever(raw_dir=raw_dir, out_dir=Path(td) / "out")
            questions = [json.loads(l) for l in
                        (out_dir / "questions.jsonl").read_text(encoding="utf-8").splitlines()]

            self.assertEqual(len(questions), 2)
            by_qid = {q["qid"]: q for q in questions}
            self.assertEqual(sorted(by_qid["1"]["gold_doc_ids"]), ["France", "Paris"])
            self.assertEqual(by_qid["2"]["gold_doc_ids"], ["Sky"])

    def test_fever_rejects_inconsistent_labels_for_the_same_claim_id(self):
        with tempfile.TemporaryDirectory() as td:
            raw_dir = Path(td) / "raw"
            raw_dir.mkdir()
            self._write_jsonl(raw_dir / "claims_labelled_dev.jsonl", [
                {"id": 1, "claim": "x", "label": "SUPPORTS", "evidence_wiki_url": "A"},
                {"id": 1, "claim": "x", "label": "REFUTES", "evidence_wiki_url": "B"},
            ])
            self._write_jsonl(raw_dir / "wiki_pages.jsonl", [])
            with self.assertRaises(ValueError):
                preprocess_fever(raw_dir=raw_dir, out_dir=Path(td) / "out")


class TestIngest(unittest.TestCase):
    def test_chunking_packs_short_paragraphs_together(self):
        text = "One two three.\n\nFour five six.\n\nSeven eight nine."
        chunks = chunk_text(text, chunk_words=100)
        self.assertEqual(chunks, ["One two three. Four five six. Seven eight nine."])

    def test_chunking_splits_an_oversized_paragraph(self):
        para = " ".join(f"word{i}." for i in range(200))
        chunks = chunk_text(para, chunk_words=50)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c.split()), 60)

    def test_load_file_as_documents_round_trips_through_disk(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "notes.txt"
            path.write_text("Alpha bravo charlie.\n\nDelta echo foxtrot.", encoding="utf-8")
            docs = load_file_as_documents(path, chunk_words=100)
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].title, "notes")
            self.assertIn("Alpha bravo charlie.", docs[0].text)

    def test_unsupported_extension_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "notes.docx"
            path.write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_file_as_documents(path)


class TestEmbedder(unittest.TestCase):
    def test_normalisation_gives_unit_rows(self):
        v = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype="float32"))
        self.assertAlmostEqual(float(np.linalg.norm(v[0])), 1.0, places=5)
        self.assertFalse(np.isnan(v).any())

    def test_tfidf_requires_fitting(self):
        with self.assertRaises(RuntimeError):
            TfidfEmbedder().encode(["text"])


class TestIndex(unittest.TestCase):
    def test_exact_self_retrieval(self):
        v = np.random.RandomState(0).rand(40, 12).astype("float32")
        idx = VectorIndex(v)
        _, i = idx.search(v[:5], 1)
        self.assertTrue((i[:, 0] == np.arange(5)).all())

    def test_ivf_falls_back_when_corpus_too_small(self):
        idx = VectorIndex(np.random.rand(20, 8).astype("float32"), index_type="ivf")
        self.assertEqual(idx.size, 20)

    def test_save_and_load_round_trip(self):
        v = np.random.RandomState(1).rand(30, 8).astype("float32")
        with tempfile.TemporaryDirectory() as td:
            VectorIndex(v).save(td)
            loaded = VectorIndex.load(td)
            a, _ = VectorIndex(v).search(v[:3], 3)
            b, _ = loaded.search(v[:3], 3)
            self.assertTrue(np.allclose(a, b))


class TestGenerator(unittest.TestCase):
    def test_evidence_is_formatted_with_citable_ids(self):
        text = format_evidence([RetrievedDoc("Warsaw", "Warsaw is in Poland.", 0.9, "Warsaw")])
        self.assertIn("[Warsaw]", text)

    def test_json_recovery(self):
        self.assertEqual(extract_json('```json\n{"a": 1,}\n```'), {"a": 1})
        self.assertEqual(extract_json("prose [1, 2] tail"), [1, 2])
        self.assertEqual(extract_json("nothing", default={}), {})

    def test_call_counting(self):
        llm = build_llm("openai:gpt-4o-mini")
        llm.complete("Reply with the single word: ack")
        llm.complete("Reply with the single word: ack")
        self.assertEqual(llm.n_calls, 2)

    def test_extract_verdict(self):
        self.assertEqual(extract_verdict("SUPPORTS\nFox 2000 released it [d1]."), "SUPPORTS")
        self.assertEqual(extract_verdict("REFUTES\nSony released it, not Fox [d1]."), "REFUTES")
        self.assertEqual(extract_verdict("INSUFFICIENT EVIDENCE"), "INSUFFICIENT EVIDENCE")
        self.assertEqual(extract_verdict("I'm not sure about this claim."), "UNKNOWN")

    def test_fact_verification_scored_by_verdict_not_em(self):
        result = BaselineResult(qid="q1", question="claim", answer="SUPPORTS\nGrounded [d1].")
        q = Question("q1", "claim", "SUPPORTS", meta={"task": "fact_verification"})
        metrics = evaluate_one(result, q)
        self.assertEqual(metrics["verdict_accuracy"], 1.0)
        self.assertEqual(metrics["em"], 0.0)

    def test_qa_question_has_no_verdict_accuracy(self):
        result = BaselineResult(qid="q1", question="q", answer="Paris [d1].")
        q = Question("q1", "q", "Paris")
        metrics = evaluate_one(result, q)
        self.assertNotIn("verdict_accuracy", metrics)


class TestMetrics(unittest.TestCase):
    def test_answer_metrics(self):
        self.assertEqual(exact_match("The Vistula.", "the vistula"), 1.0)
        self.assertAlmostEqual(token_f1("a b c", "b c d"), 0.8, places=6)
        self.assertAlmostEqual(token_f1("x y z", "y z w"), 2 / 3, places=6)
        self.assertEqual(answer_in_prediction("It lies on the Vistula.", "Vistula"), 1.0)

    def test_retrieval_metrics(self):
        self.assertEqual(recall_at_k(["a", "b"], ["a", "c"]), 0.5)
        self.assertEqual(precision_at_k(["a", "b"], ["a", "c"]), 0.5)
        self.assertTrue(recall_at_k(["a"], []) != recall_at_k(["a"], []))

    def test_aggregate_skips_missing_and_nan(self):
        out = aggregate([{"em": 1.0}, {"em": 0.0, "f1": 0.5},
                         {"em": float("nan"), "f1": 0.5}])
        self.assertAlmostEqual(out["em"], 0.5)
        self.assertAlmostEqual(out["f1"], 0.5)
        self.assertEqual(out["n_examples"], 3.0)


from src.pipeline.verifier import Verifier, split_sentences


class TestVerifier(unittest.TestCase):
    def test_supported_without_evidence_ids_is_downgraded(self):
        llm = ScriptedLLM()
        llm.register("Split the following", lambda p: '{"claims":["c1"]}')
        llm.register("Claims to verify",
                     lambda p: '{"verdicts":[{"claim":"c1","verdict":"supported","doc_ids":[]}]}')
        rep = Verifier(llm).run("c1", [RetrievedDoc("d1", "t", 1.0)])
        self.assertEqual(rep.claims[0].verdict, "partial")

    def test_citation_to_absent_passage_is_rejected(self):
        llm = ScriptedLLM()
        llm.register("Split the following", lambda p: '{"claims":["c1"]}')
        llm.register("Claims to verify",
                     lambda p: '{"verdicts":[{"claim":"c1","verdict":"supported","doc_ids":["ghost"]}]}')
        rep = Verifier(llm).run("c1", [RetrievedDoc("d1", "t", 1.0)])
        self.assertEqual(rep.claims[0].doc_ids, [])
        self.assertEqual(rep.claims[0].verdict, "partial")

    def test_missing_verdict_defaults_to_unsupported(self):
        llm = ScriptedLLM()
        llm.register("Split the following", lambda p: '{"claims":["c1","c2"]}')
        llm.register("Claims to verify",
                     lambda p: '{"verdicts":[{"claim":"c1","verdict":"supported","doc_ids":["d1"]}]}')
        rep = Verifier(llm).run("c1 c2", [RetrievedDoc("d1", "t", 1.0)])
        self.assertEqual(rep.claims[1].verdict, "unsupported")

    def test_unparseable_output_falls_back_to_sentences(self):
        rep = Verifier(ScriptedLLM()).run("First sentence. Second sentence.",
                                          [RetrievedDoc("d1", "t", 1.0)])
        self.assertEqual(rep.n_claims, 2)
        self.assertEqual(rep.support_ratio, 0.0)

    def test_split_sentences_strips_citations(self):
        self.assertEqual(split_sentences("A is B [d1]. C is D."), ["A is B.", "C is D."])


if __name__ == "__main__":
    unittest.main(verbosity=2)
