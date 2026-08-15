"""Real preliminary experiment for the RAG poster charts, over a frozen dataset
subset (HotpotQA/FEVER/toy) built by `main.py subset` + `main.py index`, instead of
an ad hoc uploaded file (see run_preliminary_experiment.py for that path).

Baseline vs self-reflective RAG on every question in the subset, plus a retrieval-only
top-k sweep. Writes results/<subset name>/results.json in the same schema the poster
artifact's "Load results.json" file picker expects.

Usage
-----
    python main.py subset  --dataset fever --n 20 --seed 13
    python main.py index   --subset data/subsets/fever_n20_seed13 \\
                            --embedder sentence-transformers/all-MiniLM-L6-v2
    python scripts/run_subset_experiment.py --subset data/subsets/fever_n20_seed13 \\
                            --index data/subsets/fever_n20_seed13/index_all_MiniLM_L6_v2
"""
import argparse
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

from src.config.config import RESULTS_DIR
from src.data.subset import load_subset
from src.generator.llm import build_llm
from src.pipeline.baseline import BaselineRAG, evaluate_one
from src.pipeline.refine import QueryReformulator
from src.pipeline.self_reflective import SelfReflectiveRAG
from src.pipeline.verifier import Verifier
from src.retrieval.retriever import DenseRetriever
from src.utils.io import write_json
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", required=True, help="frozen subset dir, e.g. data/subsets/fever_n20_seed13")
    ap.add_argument("--index", required=True, help="prebuilt index dir inside that subset")
    ap.add_argument("--llm", default="openai:gpt-4o-mini")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--topk-sweep", default="1,3,5,10,20")
    ap.add_argument("--backend", default="auto", choices=["auto", "faiss", "numpy"])
    ap.add_argument("--seed", type=int, default=13)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    subset_dir = Path(args.subset)
    questions, documents = load_subset(subset_dir)
    retriever = DenseRetriever.load(args.index, backend=args.backend)
    if len(retriever.documents) != len(documents):
        raise SystemExit(f"index has {len(retriever.documents)} passages but the "
                         f"subset has {len(documents)} - rebuild the index")
    print(f"subset: {subset_dir.name} - {len(questions)} questions, {len(documents)} passages")

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
    for i, q in enumerate(questions, 1):
        task = q.meta.get("task", "qa")
        t0 = time.perf_counter()
        b = baseline.run(q.question, qid=q.qid, task=task)
        b_row = evaluate_one(b, q)
        b_verify = verifier.run(b.answer, b.retrieved)
        b_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        p = proposed.run(q.question, qid=q.qid, task=task)
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
        print(f"  [{i}/{len(questions)}] {q.qid}: baseline conf="
              f"{per_question[-1]['baseline']['confidence']:.2f} hall="
              f"{per_question[-1]['baseline']['hallucination']:.2f}  |  self-reflective conf="
              f"{per_question[-1]['self_reflective']['confidence']:.2f} hall="
              f"{per_question[-1]['self_reflective']['hallucination']:.2f}")

    out_dir = RESULTS_DIR / subset_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "results.json", {
        "meta": {"src_file": subset_dir.name, "dataset": subset_dir.name.split("_")[0],
                 "llm": args.llm, "top_k": args.top_k, "n_questions": len(questions),
                 "seed": args.seed},
        "topk_sweep": topk_results,
        "per_question": per_question,
    })
    print(f"\nwrote {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
