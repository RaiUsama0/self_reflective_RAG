"""N-iteration ablation, computed after the fact from a completed run - no extra
LLM calls, no re-running the experiment.

run_subset_experiment.py now saves each question's full iteration history (answer,
retrieved passages, citations, verification report, and cumulative LLM calls at every
round). This script replays that history: for each candidate cap N, it re-applies the
loop's own "best support ratio so far" selection to iterations[:N], scores the result
exactly as evaluate_one() would, and reports accuracy/attribution/hallucination against
the LLM calls actually spent to get there. That answers "does each extra refinement
pass earn its cost, or do returns diminish after 1-2 iterations?" from data you already
paid for once.

Requires a results.json written by the current run_subset_experiment.py (with its
"iterations" field) - older results files predate this and cannot be ablated.

Usage
-----
    python scripts/ablation_n_iterations.py --results results/fever_n20_seed13/results.json
"""
import argparse
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator.llm import INSUFFICIENT
from src.pipeline.baseline import (
    aggregate, answer_in_prediction, exact_match, precision_at_k, recall_at_k,
    token_f1, verdict_accuracy,
)
from src.utils.io import read_json, write_json
from src.utils.schema import ClaimVerdict, VerificationReport


def _report_from_dict(d: dict | None) -> VerificationReport:
    if not d:
        return VerificationReport()
    claims = [
        ClaimVerdict(claim=c["claim"], verdict=c.get("verdict", "unsupported"),
                    doc_ids=c.get("doc_ids", []), rationale=c.get("rationale", ""))
        for c in d.get("claims", [])
    ]
    return VerificationReport(claims=claims)


def _best_at_n(iterations: list[dict], n: int) -> dict:
    """Replay self_reflective.py's own selection rule - `if report.support_ratio >
    best[0]: best = ...` - over the first n recorded rounds (or all of them, if the
    loop stopped before reaching n)."""
    capped = iterations[: min(n, len(iterations))]
    return max(capped, key=lambda it: _report_from_dict(it.get("report")).support_ratio)


def _score(best_it: dict, question: dict) -> dict[str, float]:
    answer = best_it["answer"]
    retrieved_ids = best_it["retrieved_ids"]
    citations = best_it.get("citations", [])
    gold_answer = question.get("gold_answer")
    gold_ids = question.get("gold_doc_ids", [])
    task = question.get("task", "qa")
    report = _report_from_dict(best_it.get("report"))

    metrics = {
        "n_citations": float(len(citations)),
        "abstained": float(answer.upper().startswith(INSUFFICIENT)),
        "confidence": report.support_ratio,
        "hallucination": report.hallucination_rate,
        "llm_calls_so_far": float(best_it.get("llm_calls_so_far", 0)),
    }
    # gold_doc_ids/gold_answer are only available for frozen dataset subsets
    # (run_subset_experiment.py) - main.py ask has neither, since it takes ad hoc
    # questions with no ground truth, so it can only ablate confidence/hallucination.
    if gold_ids:
        metrics["retrieval_recall"] = recall_at_k(retrieved_ids, gold_ids)
        metrics["retrieval_precision"] = precision_at_k(retrieved_ids, gold_ids)
    if gold_answer:
        if task == "fact_verification":
            metrics["verdict_accuracy"] = verdict_accuracy(answer, gold_answer)
        else:
            metrics["em"] = exact_match(answer, gold_answer)
            metrics["f1"] = token_f1(answer, gold_answer)
            metrics["answer_recall"] = answer_in_prediction(answer, gold_answer)
    return metrics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True,
                    help="results.json written by run_subset_experiment.py")
    ap.add_argument("--max-n", type=int, default=None,
                    help="highest N to report (default: longest recorded iteration "
                         "history in this file)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    data = read_json(args.results)
    questions = data.get("per_question", [])

    with_iterations = [q for q in questions if q.get("iterations")]
    if not with_iterations:
        raise SystemExit(
            f"{args.results} has no per-question 'iterations' data - it was written "
            "before this instrumentation existed. Re-run run_subset_experiment.py to "
            "get ablation-ready results.")

    max_n = args.max_n or max(len(q["iterations"]) for q in with_iterations)
    print(f"{len(with_iterations)}/{len(questions)} questions have iteration history "
          f"(deepest recorded: {max_n} iterations)\n")

    sweep = []
    for n in range(1, max_n + 1):
        rows = [_score(_best_at_n(q["iterations"], n), q) for q in with_iterations]
        summary = aggregate(rows)
        summary["n"] = n
        sweep.append(summary)
        metric_key = ("verdict_accuracy" if "verdict_accuracy" in summary
                     else "f1" if "f1" in summary else None)
        line = (f"N={n:<3} mean_llm_calls={summary['llm_calls_so_far']:.2f}  "
               f"confidence={summary['confidence']:.3f}  "
               f"hallucination={summary['hallucination']:.3f}")
        if metric_key:
            line += f"  {metric_key}={summary[metric_key]:.3f}"
        print(line)

    out_path = Path(args.results).parent / "ablation_n_iterations.json"
    write_json(out_path, {
        "source": str(args.results), "meta": data.get("meta", {}),
        "n_questions": len(with_iterations), "sweep": sweep,
    })
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
