"""Baseline single-pass RAG: retrieve once, generate once.

The control condition. The self-reflective system is this plus a verification loop,
so any measured gain has to be attributable to the loop and nothing else - same
retriever, same index, same generator, same prompt.

Also holds the evaluation metrics, since the baseline is the first thing that needs
scoring and the self-reflective arm will reuse them unchanged.
"""
from __future__ import annotations

import re
import string
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Sequence

from ..config.config import RESULTS_DIR, RunConfig
from ..generator.llm import (
    GENERATOR_SYSTEM, GENERATOR_TEMPLATE, INSUFFICIENT, BaseLLM, format_evidence,
)
from ..retrieval.retriever import DenseRetriever
from ..utils.io import write_json, write_jsonl
from ..utils.logging_utils import get_logger
from ..utils.schema import BaselineResult, Question

log = get_logger()

_CITE = re.compile(r"\[([^\]]+)\]")
_ARTICLES = re.compile(r"\b(a|an|the)\b")


class BaselineRAG:
    """Dense retrieval over a FAISS index, then one generation pass."""

    name = "baseline"

    def __init__(self, retriever: DenseRetriever, llm: BaseLLM, top_k: int = 5,
                 max_new_tokens: int = 320, temperature: float = 0.0) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def run(self, question: str, qid: str = "") -> BaselineResult:
        t0 = time.perf_counter()
        calls_before = self.llm.n_calls

        docs = self.retriever.retrieve(question, k=self.top_k)
        prompt = GENERATOR_TEMPLATE.format(
            evidence=format_evidence(docs), question=question,
            example_id=docs[0].doc_id if docs else "doc_id")
        answer = self.llm.complete(prompt, system=GENERATOR_SYSTEM,
                                   max_tokens=self.max_new_tokens,
                                   temperature=self.temperature).strip()

        valid = {d.doc_id for d in docs}
        cited = [c.strip() for m in _CITE.findall(answer) for c in m.split(",")]
        citations = [c for c in dict.fromkeys(cited) if c in valid]

        return BaselineResult(
            qid=qid, question=question, answer=answer, retrieved=docs,
            citations=citations, seconds=time.perf_counter() - t0,
            llm_calls=self.llm.n_calls - calls_before,
        )


def normalize_answer(s: str) -> str:
    """SQuAD/HotpotQA normalisation: lowercase, strip punctuation and articles."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(_ARTICLES.sub(" ", s).split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision, recall = n / len(p), n / len(g)
    return 2 * precision * recall / (precision + recall)


def answer_in_prediction(pred: str, gold: str) -> float:
    """Lenient credit: the gold span appears in a free-form answer.

    Generated answers are sentences, not spans, so strict EM understates them.
    Report EM, F1 and this together rather than choosing one.
    """
    return float(normalize_answer(gold) in normalize_answer(pred))


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    if not gold:
        return float("nan")
    return len(set(retrieved) & set(gold)) / len(set(gold))


def precision_at_k(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    if not retrieved:
        return 0.0
    return len(set(retrieved) & set(gold)) / len(set(retrieved))


def evaluate_one(result: BaselineResult, question: Question) -> dict[str, float]:
    gold_ids = question.gold_doc_ids
    return {
        "em": exact_match(result.answer, question.answer),
        "f1": token_f1(result.answer, question.answer),
        "answer_recall": answer_in_prediction(result.answer, question.answer),
        "retrieval_recall": recall_at_k(result.retrieved_ids, gold_ids),
        "retrieval_precision": precision_at_k(result.retrieved_ids, gold_ids),
        "all_gold_retrieved": float(set(gold_ids) <= set(result.retrieved_ids)) if gold_ids else float("nan"),
        "n_citations": float(len(result.citations)),
        "abstained": float(result.answer.upper().startswith(INSUFFICIENT)),
        "llm_calls": float(result.llm_calls),
        "seconds": result.seconds,
    }


def aggregate(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric over the rows that report it (NaN-safe)."""
    out: dict[str, float] = {}
    for key in sorted({k for r in rows for k in r}):
        vals = [r[key] for r in rows if key in r and r[key] == r[key]]
        if vals:
            out[key] = mean(vals)
    out["n_examples"] = float(len(rows))
    return out


def run_baseline(system: BaselineRAG, questions: Sequence[Question],
                 cfg: RunConfig, out_dir: Path | None = None) -> dict[str, float]:
    """Run every question, write per-question traces and an aggregate summary."""
    out_dir = Path(out_dir or RESULTS_DIR / cfg.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.json")

    rows, records = [], []
    t0 = time.perf_counter()
    for i, q in enumerate(questions, 1):
        result = system.run(q.question, qid=q.qid)
        row = evaluate_one(result, q)
        rows.append(row)
        records.append({
            "qid": q.qid, "question": q.question, "gold_answer": q.answer,
            "gold_doc_ids": q.gold_doc_ids, "metrics": row, "result": result.to_dict(),
        })
        if i % 25 == 0 or i == len(questions):
            log.info("  %d/%d  f1=%.3f  recall=%.3f", i, len(questions),
                     mean(r["f1"] for r in rows),
                     mean(r["retrieval_recall"] for r in rows
                          if r["retrieval_recall"] == r["retrieval_recall"]))

    write_jsonl(out_dir / "results.jsonl", records)
    summary = aggregate(rows)
    summary["wall_clock_seconds"] = time.perf_counter() - t0
    summary["system"] = "baseline"
    write_json(out_dir / "summary.json", summary)
    log.info("wrote %s", out_dir)
    return summary
