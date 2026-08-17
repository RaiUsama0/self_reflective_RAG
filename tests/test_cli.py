"""Tests for run_subset_experiment.py's unified CLI: --dataset {fever,hotpotqa,custom}
parsing, --comparison/--debug, and that quiet mode actually produces no terminal
output (not just "should" - captured and asserted). Fully offline - build_llm is
monkeypatched so no network/API key is needed.

An independently-configurable verifier ("Experiment B") is explicitly out of
dissertation scope - the verifier always uses the same model as the generator, and
TestArgumentParsing.test_no_verifier_flags_exist guards against it being reintroduced.

    python tests/test_cli.py
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable
from unittest.mock import patch

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import run_subset_experiment as rse
from src.generator.llm import BaseLLM
from src.utils.io import write_json, write_jsonl


def _set_argv(*args: str) -> None:
    sys.argv = ["run_subset_experiment.py", *args]


class TestArgumentParsing(unittest.TestCase):
    def test_fever_comparison(self):
        _set_argv("--dataset", "fever", "--comparison")
        args = rse.parse_args()
        self.assertEqual(args.dataset, "fever")
        self.assertTrue(args.comparison)
        self.assertFalse(args.debug)

    def test_hotpotqa_comparison(self):
        _set_argv("--dataset", "hotpotqa", "--comparison")
        args = rse.parse_args()
        self.assertEqual(args.dataset, "hotpotqa")

    def test_custom_comparison(self):
        _set_argv("--dataset", "custom", "--comparison")
        args = rse.parse_args()
        self.assertEqual(args.dataset, "custom")

    def test_debug_flag(self):
        _set_argv("--dataset", "custom", "--comparison", "--debug")
        args = rse.parse_args()
        self.assertTrue(args.debug)

    def test_no_verifier_flags_exist(self):
        """An independently-configurable verifier ("Experiment B") is out of
        dissertation scope - guards against --verifier/--verifier-llm being
        reintroduced to this script's CLI."""
        for flag in ("--verifier", "--verifier-llm"):
            _set_argv("--dataset", "fever", "--comparison", flag, "independent")
            with self.assertRaises(SystemExit):
                rse.parse_args()

    def test_missing_dataset_errors(self):
        _set_argv("--comparison")
        with self.assertRaises(SystemExit):
            rse.parse_args()

    def test_invalid_dataset_errors(self):
        _set_argv("--dataset", "squad", "--comparison")
        with self.assertRaises(SystemExit):
            rse.parse_args()

    def test_no_separate_single_arm_flags_exist(self):
        """Confirms --baseline/--retrieval-only/--self-reflective were not
        reintroduced as separate normal commands - --comparison always runs all
        three arms together."""
        _set_argv("--dataset", "fever", "--baseline")
        with self.assertRaises(SystemExit):
            rse.parse_args()


class RecordingLLM(BaseLLM):
    def __init__(self, name: str = "test-llm") -> None:
        super().__init__()
        self.name = name
        self.handlers: list[tuple[str, Callable[[str], str]]] = []

    def register(self, trigger: str, handler: Callable[[str], str]) -> "RecordingLLM":
        self.handlers.append((trigger, handler))
        return self

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 512,
                 temperature: float = 0.0, purpose: str = "unspecified") -> str:
        self.n_calls += 1
        self.calls_by_purpose[purpose] = self.calls_by_purpose.get(purpose, 0) + 1
        for trigger, handler in self.handlers:
            if trigger in prompt:
                return handler(prompt)
        return ""

    def _complete(self, prompt, system, max_tokens, temperature):
        raise NotImplementedError


_DOC_ID = re.compile(r"\[([A-Za-z0-9_\-]+)\]")


def _make_llm() -> RecordingLLM:
    llm = RecordingLLM("fake-model")
    llm.register("Decide whether the evidence",
                 lambda p: f"SUPPORTS\nExplanation [{(_DOC_ID.search(p) or ['doc0']).group(1) if _DOC_ID.search(p) else 'doc0'}].")
    llm.register("Split the following", lambda p: '{"claims": ["c1"]}')
    llm.register("Claims to verify",
                 lambda p: '{"verdicts": [{"claim": "c1", "verdict": "supported", "doc_ids": []}]}')
    llm.register("Write up to", lambda p: '{"queries": ["alt"]}')
    return llm


def _write_tiny_subset(root: Path) -> tuple[Path, Path]:
    """A minimal on-disk frozen subset (2 fact_verification questions, 2 passages),
    in the exact shape load_subset()/build_subset() already produce - not a new
    format."""
    from src.embeddings.embedder import build_embedder
    from src.retrieval.faiss_index import VectorIndex
    from src.retrieval.retriever import DenseRetriever
    from src.utils.schema import Document

    subset_dir = root / "tiny_subset"
    subset_dir.mkdir(parents=True)
    write_jsonl(subset_dir / "questions.jsonl", [
        {"qid": "q1", "question": "Paris is the capital of France.", "answer": "SUPPORTS",
         "gold_doc_ids": ["paris"], "meta": {"task": "fact_verification", "dataset": "test"}},
        {"qid": "q2", "question": "The sky is green.", "answer": "REFUTES",
         "gold_doc_ids": ["sky"], "meta": {"task": "fact_verification", "dataset": "test"}},
    ])
    docs = [Document(doc_id="paris", title="paris", text="Paris is the capital of France."),
           Document(doc_id="sky", title="sky", text="The sky is blue during the day.")]
    write_jsonl(subset_dir / "corpus.jsonl", (d.to_dict() for d in docs))
    write_json(subset_dir / "manifest.json", {"name": "tiny_subset", "dataset": "test",
                                              "questions_checksum": "x"})

    index_dir = subset_dir / "index_tfidf"
    retriever = DenseRetriever.build(docs, embedder_name="tfidf", backend="numpy")
    retriever.save(index_dir)
    return subset_dir, index_dir


class TestQuietOutput(unittest.TestCase):
    """Proves --comparison's quiet mode produces no terminal noise, and that the
    non-quiet path still does (so a future change can't silently break FEVER/
    HotpotQA's normal debug visibility while "fixing" the summary)."""

    def test_quiet_run_prints_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            subset_dir, index_dir = _write_tiny_subset(Path(td))
            with patch.object(rse, "build_llm", return_value=_make_llm()):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rse.run_experiment(subset_dir, index_dir, "fake:model", None,
                                       name="tiny_quiet", quiet=True)
            self.assertEqual(buf.getvalue().strip(), "")

    def test_non_quiet_run_prints_progress(self):
        with tempfile.TemporaryDirectory() as td:
            subset_dir, index_dir = _write_tiny_subset(Path(td))
            with patch.object(rse, "build_llm", return_value=_make_llm()):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rse.run_experiment(subset_dir, index_dir, "fake:model", None,
                                       name="tiny_debug", quiet=False)
            out = buf.getvalue()
            self.assertIn("[1/2]", out)
            self.assertIn("subset:", out)

    def test_comparison_summary_reads_back_correctly(self):
        """End-to-end: quiet run -> results.json -> print_comparison_summary()
        produces the FEVER-style table with real (not placeholder) numbers."""
        with tempfile.TemporaryDirectory() as td:
            subset_dir, index_dir = _write_tiny_subset(Path(td))
            with patch.object(rse, "build_llm", return_value=_make_llm()):
                out_dir = rse.run_experiment(subset_dir, index_dir, "fake:model", None,
                                             name="tiny_summary", quiet=True)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rse.print_comparison_summary(out_dir / "results.json", "fever")
            printed = buf.getvalue()
            self.assertIn("SELF-REFLECTIVE RAG — FEVER — n=2", printed)
            self.assertIn("Verdict Accuracy", printed)
            self.assertIn("STATISTICAL COMPARISON", printed)
            self.assertIn("KEY FINDING", printed)
            self.assertIn("EXPERIMENT COMPLETED", printed)
            self.assertNotIn("0.xxx", printed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
