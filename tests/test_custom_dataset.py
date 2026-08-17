"""Tests for the custom FEVER-like knowledge-base dataset: the claim/label/evidence
questions.json + knowledge_base/*.txt loader (src/data/custom_dataset.py) that
replaced the old ad hoc `ask --file knowledge_base` flow for gold-labelled
evaluation. Fully offline - no network or API key needed.

    python tests/test_custom_dataset.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.custom_dataset import build_custom_subset
from src.data.subset import load_subset
from src.generator.llm import BaseLLM
from src.pipeline.baseline import BaselineRAG
from src.pipeline.refine import QueryReformulator
from src.pipeline.retrieval_only import RetrievalOnlyRAG
from src.pipeline.self_reflective import SelfReflectiveRAG
from src.pipeline.verifier import Verifier
from src.retrieval.retriever import DenseRetriever


def _write_kb(kb_dir: Path, files: dict[str, str]) -> None:
    kb_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (kb_dir / name).write_text(text, encoding="utf-8")


class TestBuildCustomSubset(unittest.TestCase):
    def test_maps_id_claim_label_evidence_to_question_schema(self):
        with tempfile.TemporaryDirectory() as td:
            kb_dir = Path(td) / "kb"
            _write_kb(kb_dir, {
                "hospital.txt": "CityCare Hospital was founded in 1985 in Manchester.",
                "airline.txt": "SkyBridge Airlines is headquartered in Dublin.",
            })
            q_path = Path(td) / "questions.json"
            q_path.write_text(json.dumps([
                {"id": "q001", "claim": "CityCare Hospital was founded in 1985.",
                 "label": "SUPPORTS", "evidence": ["hospital.txt"]},
                {"id": "q002", "claim": "CityCare Hospital has 500 beds.",
                 "label": "REFUTES", "evidence": ["hospital.txt"]},
            ]), encoding="utf-8")

            out_dir = build_custom_subset(kb_dir, q_path, out_dir=Path(td) / "out",
                                          chunk_words=500)
            questions, documents = load_subset(out_dir)

            self.assertEqual(len(questions), 2)
            self.assertEqual(len(documents), 2)
            q1 = next(q for q in questions if q.qid == "q001")
            self.assertEqual(q1.question, "CityCare Hospital was founded in 1985.")
            self.assertEqual(q1.answer, "SUPPORTS")
            self.assertEqual(q1.meta["task"], "fact_verification")
            hospital_doc_id = next(d.doc_id for d in documents if d.title == "hospital")
            self.assertEqual(q1.gold_doc_ids, [hospital_doc_id])

            q2 = next(q for q in questions if q.qid == "q002")
            self.assertEqual(q2.answer, "REFUTES")

    def test_evidence_union_across_multiple_chunks_of_one_file(self):
        """A file long enough to be split into several chunks should resolve its
        gold_doc_ids to *every* chunk belonging to it, not just one."""
        with tempfile.TemporaryDirectory() as td:
            kb_dir = Path(td) / "kb"
            long_text = "\n\n".join(f"Paragraph {i} contains fact number {i}." for i in range(40))
            _write_kb(kb_dir, {"museum.txt": long_text})
            q_path = Path(td) / "questions.json"
            q_path.write_text(json.dumps([
                {"id": "q001", "claim": "The museum has many facts.",
                 "label": "SUPPORTS", "evidence": ["museum.txt"]},
            ]), encoding="utf-8")

            out_dir = build_custom_subset(kb_dir, q_path, out_dir=Path(td) / "out",
                                          chunk_words=20)
            questions, documents = load_subset(out_dir)

            self.assertGreater(len(documents), 1)
            self.assertEqual(set(questions[0].gold_doc_ids), {d.doc_id for d in documents})

    def test_unknown_evidence_file_raises(self):
        with tempfile.TemporaryDirectory() as td:
            kb_dir = Path(td) / "kb"
            _write_kb(kb_dir, {"hospital.txt": "Some hospital text."})
            q_path = Path(td) / "questions.json"
            q_path.write_text(json.dumps([
                {"id": "q001", "claim": "x", "label": "SUPPORTS",
                 "evidence": ["nonexistent.txt"]},
            ]), encoding="utf-8")
            with self.assertRaises(KeyError):
                build_custom_subset(kb_dir, q_path, out_dir=Path(td) / "out")


class RecordingLLM(BaseLLM):
    """Same pattern as tests/test_arms.py's double, redefined here so this file
    stays independently runnable: logs every prompt/system/purpose it receives and
    returns a scripted response by trigger substring match."""

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
        raise NotImplementedError


_DOC_ID = re.compile(r"\[([A-Za-z0-9_\-]+)\]")


def _first_doc_id(prompt: str) -> str:
    m = _DOC_ID.search(prompt)
    return m.group(1) if m else "doc0"


class TestNoGoldLeakageCustomDataset(unittest.TestCase):
    """Builds a custom subset whose gold label is a distinctive sentinel string that
    never appears in the claim text or the knowledge-base documents, runs all three
    arms passing only the claim, and asserts the sentinel never appears in any prompt
    any model was actually called with.

    Note on scope: the gold *evidence filename* is deliberately not used as a second
    sentinel here, unlike the FEVER/HotpotQA gold-leakage tests. There, the sentinel
    doc id was chosen to not exist in the corpus at all, so its absence from prompts
    was structural. Here, an evidence filename necessarily names a real, retrievable
    knowledge-base document - its doc id legitimately appears in the evidence block
    whenever the retriever (correctly) retrieves it, which is retrieval doing its
    job, not gold information leaking. The label is the only piece of information a
    generator/verifier/reformulator prompt could not derive from retrieval, so it is
    the one that actually tests leakage."""

    SENTINEL_LABEL = "SENTINEL_GOLD_LABEL_7c2f"

    def _build(self):
        td = tempfile.TemporaryDirectory()
        kb_dir = Path(td.name) / "kb"
        _write_kb(kb_dir, {
            "sentinel_doc.txt": "This document contains an unrelated fact "
                                "about a fictional town called Rivermeade.",
            "other.txt": "This document is about a different fictional town called "
                        "Ashford.",
        })
        q_path = Path(td.name) / "questions.json"
        q_path.write_text(json.dumps([
            {"id": "q001", "claim": "Rivermeade is a fictional town.",
             "label": self.SENTINEL_LABEL, "evidence": ["sentinel_doc.txt"]},
        ]), encoding="utf-8")
        out_dir = build_custom_subset(kb_dir, q_path, out_dir=Path(td.name) / "subset",
                                      chunk_words=500)
        questions, documents = load_subset(out_dir)
        retriever = DenseRetriever.build(documents, embedder_name="tfidf", backend="numpy")
        return td, questions[0], retriever

    def _assert_clean(self, *llms: RecordingLLM) -> None:
        for llm in llms:
            for entry in llm.log:
                self.assertNotIn(self.SENTINEL_LABEL, entry["prompt"])
                self.assertNotIn(self.SENTINEL_LABEL, entry["system"])

    def test_baseline_never_sees_gold(self):
        td, q, retriever = self._build()
        try:
            llm = RecordingLLM("gen")
            llm.register("Decide whether the evidence",
                         lambda p: f"SUPPORTS\nExplanation [{_first_doc_id(p)}].")
            BaselineRAG(retriever, llm, top_k=2).run(q.question, qid=q.qid,
                                                     task=q.meta.get("task", "qa"))
            self._assert_clean(llm)
        finally:
            td.cleanup()

    def test_retrieval_only_never_sees_gold(self):
        td, q, retriever = self._build()
        try:
            llm = RecordingLLM("gen")
            llm.register("Decide whether the evidence",
                         lambda p: f"SUPPORTS\nExplanation [{_first_doc_id(p)}].")
            RetrievalOnlyRAG(retriever, llm, top_k=2, max_iterations=2).run(
                q.question, qid=q.qid, task=q.meta.get("task", "qa"))
            self._assert_clean(llm)
        finally:
            td.cleanup()

    def test_self_reflective_never_sees_gold_with_independent_verifier(self):
        td, q, retriever = self._build()
        try:
            gen_llm = RecordingLLM("gen")
            gen_llm.register("Decide whether the evidence",
                             lambda p: f"SUPPORTS\nExplanation [{_first_doc_id(p)}].")
            gen_llm.register("Write up to", lambda p: '{"queries": ["alt query"]}')
            verifier_llm = RecordingLLM("verifier")
            verifier_llm.register("Split the following", lambda p: '{"claims": ["c1"]}')
            verifier_llm.register(
                "Claims to verify",
                lambda p: ('{"verdicts": [{"claim": "c1", "verdict": "unsupported", '
                          '"doc_ids": []}]}'))

            SelfReflectiveRAG(retriever, gen_llm, Verifier(verifier_llm),
                              QueryReformulator(gen_llm), top_k=2, max_iterations=2).run(
                q.question, qid=q.qid, task=q.meta.get("task", "qa"))

            self._assert_clean(gen_llm, verifier_llm)
        finally:
            td.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
