"""Real preliminary experiment for the RAG poster charts: baseline vs self-reflective
RAG over a gold-labeled question set built from any local file, plus a retrieval-only
top-k sweep.

This is NOT the dissertation's HotpotQA/FEVER evaluation - it is a small, real,
honestly-labeled pilot run on one document, for genuine numbers to chart rather than
placeholder data. Re-run against the full experiment once that lands.

Usage
-----
1. Find passage IDs to write gold_doc_ids against (skips all LLM calls):

    python scripts/run_preliminary_experiment.py --file yourfile.pdf --list-chunks

2. Write a questions file, one JSON object per line (gold_doc_ids may be [] for an
   intentionally out-of-scope question - that question is then excluded from the
   top-k/retrieval-recall sweep but still scored for confidence/hallucination/latency):

    {"question": "Who is the medical director?", "answer": "Dr. Robert Evans", "gold_doc_ids": ["yourfile_0000"]}
    {"question": "What is the parking fee?", "answer": "not stated", "gold_doc_ids": []}

3. Run the experiment:

    python scripts/run_preliminary_experiment.py --file yourfile.pdf \\
        --questions questions.jsonl --name yourfile_eval

   Writes results/<name>/results.json - load that straight into the poster artifact's
   "Load results.json" file picker to redraw every chart with the new data.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.config.config import RESULTS_DIR, SUBSETS_DIR
from src.data.ingest import load_file_as_documents
from src.generator.llm import build_llm
from src.pipeline.baseline import BaselineRAG, evaluate_one
from src.pipeline.refine import QueryReformulator
from src.pipeline.self_reflective import SelfReflectiveRAG
from src.pipeline.verifier import Verifier
from src.retrieval.retriever import DenseRetriever
from src.utils.io import write_json, write_jsonl
from src.utils.schema import Question
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help=".txt/.md/.pdf to build the corpus from")
    ap.add_argument("--questions", default=None,
                    help="JSONL: one {question, answer, gold_doc_ids} per line - "
                         "required unless --list-chunks")
    ap.add_argument("--list-chunks", action="store_true",
                    help="print each passage's id and text, then exit - use this "
                         "first to find gold_doc_ids for your questions file")
    ap.add_argument("--name", default=None,
                    help="subset/results directory name (default: derived from --file)")
    ap.add_argument("--llm", default="openai:gpt-4o-mini")
    ap.add_argument("--chunk-words", type=int, default=20,
                    help="small on purpose - keeps enough passages for the top-k "
                         "sweep to show variation instead of collapsing to 1 chunk")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--topk-sweep", default="1,2,3,4,6",
                    help="comma-separated k values for the retrieval-only sweep")
    ap.add_argument("--seed", type=int, default=13)
    return ap.parse_args()


def load_questions(path: Path) -> list[Question]:
    questions = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        questions.append(Question(
            qid=row.get("qid", f"q{i}"), question=row["question"],
            answer=row.get("answer", ""), gold_doc_ids=row.get("gold_doc_ids", [])))
    return questions


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    src_path = Path(args.file)
    name = args.name or (src_path.stem + "_eval")

    documents = load_file_as_documents(src_path, chunk_words=args.chunk_words)

    if args.list_chunks:
        for d in documents:
            print(f"=== {d.doc_id} ===\n{d.text}\n")
        print(f"{len(documents)} passages")
        return 0

    if not args.questions:
        raise SystemExit("--questions is required (or pass --list-chunks to inspect "
                         "passage ids first)")
    questions = load_questions(Path(args.questions))
    if not questions:
        raise SystemExit(f"{args.questions} has no questions")

    subset_dir = SUBSETS_DIR / name
    subset_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(subset_dir / "corpus.jsonl", (d.to_dict() for d in documents))
    write_jsonl(subset_dir / "questions.jsonl", (q.to_dict() for q in questions))
    print(f"corpus: {len(documents)} passages, {len(questions)} gold questions -> {subset_dir}")

    retriever = DenseRetriever.build(documents, embedder_name="sentence-transformers/all-MiniLM-L6-v2",
                                     seed=args.seed)
    retriever.save(subset_dir / "index_all_MiniLM_L6_v2")

    topk_values = [int(k) for k in args.topk_sweep.split(",")]
    print("\n--- top-k sweep (retrieval recall, no generation) ---")
    topk_results = []
    gold_questions = [q for q in questions if q.gold_doc_ids]
    for k in topk_values:
        recalls = []
        for q in gold_questions:
            hits = retriever.retrieve(q.question, k=k)
            got_ids = {d.doc_id for d in hits}
            recalls.append(len(got_ids & set(q.gold_doc_ids)) / len(q.gold_doc_ids))
        mean_recall = sum(recalls) / len(recalls) if recalls else float("nan")
        topk_results.append({"k": k, "retrieval_recall": mean_recall})
        print(f"  k={k:<3} retrieval_recall={mean_recall:.3f}")

    llm = build_llm(args.llm)
    verifier = Verifier(llm)
    baseline = BaselineRAG(retriever, llm, top_k=args.top_k)
    proposed = SelfReflectiveRAG(retriever, llm, verifier, QueryReformulator(llm),
                                 top_k=args.top_k, max_iterations=3)

    print(f"\n--- baseline vs self-reflective at top_k={args.top_k} ({len(questions)} questions) ---")
    per_question = []
    for q in questions:
        t0 = time.perf_counter()
        b = baseline.run(q.question, qid=q.qid)
        b_row = evaluate_one(b, q)
        b_verify = verifier.run(b.answer, b.retrieved)
        b_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        p = proposed.run(q.question, qid=q.qid)
        p_row = evaluate_one(p, q)
        p_seconds = time.perf_counter() - t0
        p_report = p.final_report

        per_question.append({
            "qid": q.qid, "question": q.question,
            "baseline": {**b_row, "confidence": b_verify.support_ratio,
                        "hallucination": b_verify.hallucination_rate, "seconds": b_seconds},
            "self_reflective": {**p_row, "confidence": p_report.support_ratio if p_report else 0.0,
                                "hallucination": p_report.hallucination_rate if p_report else 1.0,
                                "seconds": p_seconds},
        })
        print(f"  {q.qid}: baseline conf={per_question[-1]['baseline']['confidence']:.2f} "
              f"hall={per_question[-1]['baseline']['hallucination']:.2f}  |  "
              f"self-reflective conf={per_question[-1]['self_reflective']['confidence']:.2f} "
              f"hall={per_question[-1]['self_reflective']['hallucination']:.2f}")

    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "results.json", {
        "meta": {"src_file": str(src_path), "chunk_words": args.chunk_words, "llm": args.llm,
                 "top_k": args.top_k, "n_questions": len(questions), "seed": args.seed},
        "topk_sweep": topk_results,
        "per_question": per_question,
    })
    print(f"\nwrote {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
