"""Interactive comparison: ask a question, get one tabular report.

One report shape only, used identically whether the question comes from a dataset
subset (HotpotQA/FEVER, via --subset) or an ingested file (e.g. knowledge_base, via
--file), and whether it is asked once (--question), in bulk (--questions-file), or
live at the REPL prompt. Each question gets a detail table (initial answer,
reflection stats, final answer, call counts); a batch of more than one question also
gets a closing summary table with the mean confidence/hallucination across all of
them.
"""
from __future__ import annotations

import textwrap

from ..utils.schema import VerificationReport
from .baseline import BaselineRAG
from .self_reflective import SelfReflectiveRAG

_EXIT_WORDS = {"end", "quit", "exit", "q"}

_FIELD_W = 39
_VALUE_W = 56


def _wrap_cell(value: str, width: int = _VALUE_W) -> list[str]:
    text = str(value).strip() or "(empty)"
    return textwrap.wrap(text, width=width) or [""]


def _rule(corner: str = "+") -> str:
    return corner + "-" * (_FIELD_W + 2) + corner + "-" * (_VALUE_W + 2) + corner


def _row(field: str, value: str) -> None:
    lines = _wrap_cell(value)
    print(f"| {field:<{_FIELD_W}} | {lines[0]:<{_VALUE_W}} |")
    for cont in lines[1:]:
        print(f"| {'':<{_FIELD_W}} | {cont:<{_VALUE_W}} |")


def print_table_report(question: str, baseline: BaselineRAG, proposed: SelfReflectiveRAG,
                       index: int | None = None, total: int | None = None
                       ) -> dict[str, float]:
    """Run both systems on one question and print its detail table.

    'Initial answer' and the reflection stats (hallucination/confidence/claims) are
    the self-reflective loop's own first-pass results, before any refinement -
    'Final answer' is what it settled on after verifying and, if needed, retrying.
    Baseline is run only for its LLM-call count, the cost half of the comparison.
    """
    result = proposed.run(question)
    b = baseline.run(question)

    first = result.iterations[0] if result.iterations else None
    initial_answer = first.answer if first else result.answer
    report = (first.report if first and first.report else result.final_report) or VerificationReport()

    label = f"QUESTION {index}/{total}" if index and total else "QUESTION"
    print(f"\n{label}: {question}")
    print(_rule())
    _row("Field", "Value")
    print(_rule())
    _row("Initial answer", initial_answer)
    _row("Hallucination rate", f"{report.hallucination_rate:.2f}")
    _row("Is supporting evidence located?",
         "Yes" if report.n_supported > 0 else "No")
    _row("Claims supported / unsupported",
         f"{report.n_supported} / {report.n_unsupported} (of {report.n_claims})")
    _row("Confidence rate", f"{report.support_ratio:.2f}")
    _row("Final answer", result.answer)
    _row("Loop calls: self-reflective iterations", str(result.n_iterations))
    _row("LLM calls: self-reflective", str(result.llm_calls))
    _row("LLM calls: baseline", str(b.llm_calls))
    print(_rule())

    return {"question": question, "confidence": report.support_ratio,
            "hallucination": report.hallucination_rate,
            "iterations": float(result.n_iterations),
            "self_calls": float(result.llm_calls),
            "baseline_calls": float(b.llm_calls)}


def _print_summary_table(rows: list[dict[str, float]]) -> None:
    if len(rows) <= 1:
        return
    width = 100
    print("\n" + "=" * width)
    print(f"SUMMARY ({len(rows)} questions) - mean confidence and hallucination")
    print(f"{'#':<4}{'question':<74}{'confidence':>10}{'hallucination':>14}")
    for i, r in enumerate(rows, 1):
        q_short = r["question"] if len(r["question"]) <= 73 else r["question"][:70] + "..."
        print(f"{i:<4}{q_short:<74}{r['confidence']:>10.2f}{r['hallucination']:>14.2f}")
    mean_conf = sum(r["confidence"] for r in rows) / len(rows)
    mean_hall = sum(r["hallucination"] for r in rows) / len(rows)
    print("-" * width)
    print(f"{'mean':<78}{mean_conf:>10.2f}{mean_hall:>14.2f}")
    print("=" * width)
    total_iter = sum(r["iterations"] for r in rows)
    total_self = sum(r["self_calls"] for r in rows)
    total_base = sum(r["baseline_calls"] for r in rows)
    print(f"total loop calls: self-reflective iterations {total_iter:.0f}, "
          f"self-reflective llm calls {total_self:.0f}, baseline llm calls {total_base:.0f}")


def batch_table_ask(questions: list[str], baseline: BaselineRAG,
                    proposed: SelfReflectiveRAG) -> list[dict[str, float]]:
    """Run every question through both systems: one detail table each, then one
    summary table with the mean confidence/hallucination. Source-agnostic - the
    caller decides whether `questions` came from --subset or --file."""
    rows = [print_table_report(q, baseline, proposed, i, len(questions))
            for i, q in enumerate(questions, 1)]
    _print_summary_table(rows)
    return rows


def repl_table(baseline: BaselineRAG, proposed: SelfReflectiveRAG,
              intro: str | None = None) -> None:
    """Live question loop: one detail table per question typed. Prints the summary
    table on exit if more than one question was asked."""
    if intro:
        print(f"\n{intro}")
    else:
        print("\nType a question, or 'end' to exit.")
        print("Questions must be answerable from the indexed corpus - the systems can "
              "only use retrieved passages.\n")
    rows: list[dict[str, float]] = []
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in _EXIT_WORDS:
            break
        try:
            rows.append(print_table_report(q, baseline, proposed))
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}")
    _print_summary_table(rows)
