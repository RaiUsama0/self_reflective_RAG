"""The dissertation's only experiment runner: BaselineRAG vs. RetrievalOnlyRAG vs.
SelfReflectiveRAG, over exactly three supported datasets - FEVER, HotpotQA, and a
custom FEVER-like knowledge base.

    python scripts/run_subset_experiment.py --dataset fever --comparison
    python scripts/run_subset_experiment.py --dataset hotpotqa --comparison
    python scripts/run_subset_experiment.py --dataset custom --comparison

    # Experiment B - independent verifier (a different model judges the generator):
    python scripts/run_subset_experiment.py --dataset fever --comparison --verifier independent

    # verbose per-question/retrieval/indexing logging instead of the clean summary:
    python scripts/run_subset_experiment.py --dataset custom --comparison --debug

Each prints one final comparison table (add --debug for verbose per-question output)
and writes results/<name>/results.json. There is no separate command to run a single
arm, and no separate ablation/comparison-experiment scripts - --comparison always
means "run the same questions through all three arms, then compare."

Why three arms, not two
------------------------
SelfReflectiveRAG differs from BaselineRAG in two entangled ways: it retrieves more
evidence over multiple rounds, AND it uses claim-level verification to decide what to
retrieve next and when to stop. If self-reflective beats baseline, that alone cannot
tell you which of those two mechanisms is responsible. RetrievalOnlyRAG runs the
identical retrieval-expansion schedule with no verifier at all, so:

    BaselineRAG vs RetrievalOnlyRAG        -> does more evidence alone help?
    RetrievalOnlyRAG vs SelfReflectiveRAG  -> does verification help beyond that?
    BaselineRAG vs SelfReflectiveRAG       -> the total effect (the two above, summed)

Budget matching
----------------
RetrievalOnlyRAG is run once per question with `max_iterations` set to exactly the
number of rounds SelfReflectiveRAG actually used for that same question (at least 1).
This is a per-question exact match, not an average - the true retrieval budget each
arm used is recorded in `iterations` for every question either way.

Attribution scoring for arms with no built-in verifier
--------------------------------------------------------
SelfReflectiveRAG's confidence/hallucination come from its own internal verification
(part of its mechanism). BaselineRAG and RetrievalOnlyRAG have no verifier at all, so
for the comparison table we run one *evaluation-only* verification pass over each of
their final answers - this is bookkeeping for the results table, not part of either
arm's operation, and its cost is tracked separately (`attribution_scoring_calls`) so
the cost/benefit numbers are never inflated by evaluation overhead.

No gold leakage
-----------------
Every arm's `.run()` is called with only `question.question`/`qid`/`task`. Gold
answers and gold doc ids are read from `question` here, in this script, strictly
after generation, purely to compute metrics - never before, never passed to any
retriever/generator/verifier call. See tests/test_arms.py::TestNoGoldLeakage and
tests/test_custom_dataset.py::TestNoGoldLeakageCustomDataset for the structural proof.

The generator and verifier are independently configurable - not a separate experiment
mode, just a parameter of the one comparison every invocation runs, same as --top-k
or --max-iterations:

    --verifier same (default)   verifier uses the same model as the generator
                                 (Experiment A - tests the loop with no independence
                                 assumption at all)
    --verifier independent      verifier uses DEFAULT_INDEPENDENT_VERIFIER_MODEL, a
                                 different model from the generator (Experiment B)
    --verifier-llm <spec>       explicit override, takes precedence over --verifier
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))
load_dotenv(PROJECT_ROOT / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.config.config import (
    DEFAULT_GENERATOR_MODEL, DEFAULT_INDEPENDENT_VERIFIER_MODEL, DEFAULT_VERIFIER_MODEL,
    EmbeddingConfig, GenerationConfig, ModelConfig, PROCESSED_DIR, RESULTS_DIR,
    RetrievalConfig, RunConfig, SUBSETS_DIR,
)
from src.data.custom_dataset import build_custom_subset
from src.data.subset import build_subset, load_subset
from src.generator.llm import build_llm
from src.pipeline.baseline import BaselineRAG, aggregate, evaluate_one
from src.pipeline.refine import QueryReformulator
from src.pipeline.retrieval_only import RetrievalOnlyRAG
from src.pipeline.self_reflective import SelfReflectiveRAG
from src.pipeline.verifier import Verifier
from src.retrieval.retriever import DenseRetriever
from src.utils.io import read_json, write_json
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

from analyze_results import COMPARISONS, key_metrics_for, paired_significance

log = get_logger()

_DATASET_TITLE = {"fever": "FEVER", "hotpotqa": "HotpotQA", "custom": "Custom FEVER-like"}


def _score_with_evaluation_only_verifier(answer: str, docs, verifier: Verifier) -> dict:
    """Runs decompose+verify purely to report confidence/hallucination for an arm
    that has no built-in verifier (baseline, retrieval-only). Not part of that arm's
    mechanism - cost is returned separately so it never counts toward the arm's own
    operational cost."""
    before = dict(verifier.llm.calls_by_purpose)
    report = verifier.run(answer, docs)
    after = dict(verifier.llm.calls_by_purpose)
    calls = {k: after.get(k, 0) - before.get(k, 0) for k in after
            if after.get(k, 0) - before.get(k, 0)}
    return {"confidence": report.support_ratio, "hallucination": report.hallucination_rate,
            "attribution_scoring_calls": calls}


def run_experiment(subset_dir: Path, index_dir: Path, generator_spec: str,
                   verifier_spec: str | None, top_k: int = 5, expand_k: int = 3,
                   max_iterations: int = 3, min_support_ratio: float = 1.0,
                   backend: str = "auto", seed: int = 13, name: str | None = None,
                   resume: bool = True, checkpoint_every: int = 20,
                   quiet: bool = False) -> Path:
    """Runs BaselineRAG, RetrievalOnlyRAG (budget-matched), and SelfReflectiveRAG
    over every question in `subset_dir`, and writes results/<name>/results.json +
    config.json. Returns the output directory.

    Checkpointed: writes progress to results/<name>/results.partial.json every
    `checkpoint_every` questions and again immediately if the loop raises, so a
    transient failure (network timeout, rate limit, ^C) partway through a long run
    does not silently discard the API spend and time already sunk into it. With
    `resume=True` (the default), a second invocation with identical settings picks up
    from that checkpoint instead of starting over.

    `quiet=True` suppresses the per-question progress lines and intermediate prints -
    the comparison summary is printed separately by print_comparison_summary().
    """
    set_seed(seed)
    subset_dir, index_dir = Path(subset_dir), Path(index_dir)
    questions, documents = load_subset(subset_dir)
    retriever = DenseRetriever.load(index_dir, backend=backend)
    if len(retriever.documents) != len(documents):
        raise SystemExit(f"index has {len(retriever.documents)} passages but the "
                         f"subset has {len(documents)} - rebuild the index")

    verifier_spec = verifier_spec or generator_spec
    generator_llm = build_llm(generator_spec)
    verifier_llm = generator_llm if verifier_spec == generator_spec else build_llm(verifier_spec)
    models = ModelConfig(generator=generator_spec, verifier=verifier_spec)
    log.info("generator=%s  verifier=%s  independent=%s",
             models.generator, models.verifier, models.verifier_is_independent)

    verifier = Verifier(verifier_llm)
    reformulator = QueryReformulator(generator_llm)
    baseline = BaselineRAG(retriever, generator_llm, top_k=top_k)
    retrieval_only = RetrievalOnlyRAG(retriever, generator_llm, top_k=top_k,
                                      expand_k_each_iteration=expand_k,
                                      max_iterations=max_iterations)
    self_reflective = SelfReflectiveRAG(
        retriever, generator_llm, verifier, reformulator, top_k=top_k,
        max_iterations=max_iterations, min_support_ratio=min_support_ratio,
        expand_k_each_iteration=expand_k)

    manifest = read_json(subset_dir / "manifest.json") if (subset_dir / "manifest.json").exists() else {}
    dataset = manifest.get("dataset", subset_dir.name.split("_")[0])
    subset_checksum = manifest.get("questions_checksum", "")

    if not quiet:
        print(f"subset: {subset_dir.name} - {len(questions)} questions, {len(documents)} passages")

    out_name = name or subset_dir.name
    out_dir = RESULTS_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "results.partial.json"

    per_question: list[dict] = []
    skip_count = 0
    if resume and partial_path.exists():
        per_question = read_json(partial_path).get("per_question", [])
        skip_count = len(per_question)
        checkpoint_qids = [row["qid"] for row in per_question]
        live_qids = [q.qid for q in questions[:skip_count]]
        if checkpoint_qids != live_qids:
            raise SystemExit(
                f"{partial_path} does not match this subset's question order - "
                "refusing to resume from a mismatched checkpoint. Delete it or pass "
                "resume=False to start over.")
        if not quiet:
            print(f"resuming from checkpoint: {skip_count}/{len(questions)} questions "
                 f"already completed ({partial_path})")

    if not quiet:
        print(f"\n--- baseline vs retrieval-only vs self-reflective at top_k={top_k} "
              f"({len(questions)} questions) ---")
    try:
        _run_questions(questions, skip_count, baseline, retrieval_only, self_reflective,
                       verifier, per_question, partial_path, checkpoint_every, quiet)
    except Exception:
        write_json(partial_path, {"per_question": per_question})
        log.error("run failed at %d/%d questions - progress saved to %s "
                 "(re-run with the same --name to resume from here)",
                 len(per_question), len(questions), partial_path)
        raise

    if partial_path.exists():
        partial_path.unlink()

    cfg = RunConfig(
        name=out_name, arm="baseline+retrieval_only+self_reflective",
        subset=str(subset_dir), subset_checksum=subset_checksum,
        index_dir=str(index_dir), seed=seed, dataset=dataset,
        embedding=EmbeddingConfig(model=retriever.embedder.name),
        retrieval=RetrievalConfig(top_k=top_k, backend=retriever.index.backend,
                                  index_type=retriever.index.index_type,
                                  expand_k_each_iteration=expand_k,
                                  max_iterations=max_iterations),
        generation=GenerationConfig(model=generator_spec),
        models=models, min_support_ratio=min_support_ratio,
    )
    cfg.save(out_dir / "config.json")

    write_json(out_dir / "results.json", {
        "meta": {
            "src_file": subset_dir.name, "dataset": dataset, "seed": seed,
            "n_questions": len(questions), "top_k": top_k,
            "generator_model": models.generator, "verifier_model": models.verifier,
            "reformulator_model": models.reformulator,
            "verifier_is_independent": models.verifier_is_independent,
            "schema_version": 2,
            "total_calls": {
                "generator_llm": generator_llm.usage_report(),
                "verifier_llm": (verifier_llm.usage_report()
                                if verifier_llm is not generator_llm else None),
            },
        },
        "per_question": per_question,
    })
    if not quiet:
        print(f"\nwrote {out_dir / 'results.json'} and {out_dir / 'config.json'}")
    return out_dir


def _run_questions(questions, skip_count, baseline, retrieval_only, self_reflective,
                   verifier, per_question, partial_path, checkpoint_every, quiet=False):
    for i, q in enumerate(questions, 1):
        if i <= skip_count:
            continue
        task = q.meta.get("task", "qa")

        t0 = time.perf_counter()
        b = baseline.run(q.question, qid=q.qid, task=task)
        b_row = evaluate_one(b, q)
        b_score = _score_with_evaluation_only_verifier(b.answer, b.retrieved, verifier)
        b_row.update({"confidence": b_score["confidence"],
                     "hallucination": b_score["hallucination"], "seconds": time.perf_counter() - t0})

        t0 = time.perf_counter()
        p = self_reflective.run(q.question, qid=q.qid, task=task)
        p_row = evaluate_one(p, q)
        p_report = p.final_report
        p_row.update({"confidence": p_report.support_ratio if p_report else 0.0,
                     "hallucination": p_report.hallucination_rate if p_report else 1.0,
                     "seconds": time.perf_counter() - t0})

        t0 = time.perf_counter()
        r = retrieval_only.run(q.question, qid=q.qid, task=task,
                               max_iterations=max(1, p.n_iterations))
        r_row = evaluate_one(r, q)
        r_score = _score_with_evaluation_only_verifier(r.answer, r.retrieved, verifier)
        r_row.update({"confidence": r_score["confidence"],
                     "hallucination": r_score["hallucination"], "seconds": time.perf_counter() - t0})

        per_question.append({
            "qid": q.qid, "question": q.question, "task": task,
            "gold_answer": q.answer, "gold_doc_ids": q.gold_doc_ids,
            "baseline": {**b_row, "generator_model": b.generator_model,
                        "calls_by_purpose": b.calls_by_purpose,
                        "attribution_scoring_calls": b_score["attribution_scoring_calls"]},
            "retrieval_only": {**r_row, "generator_model": r.generator_model,
                              "calls_by_purpose": r.calls_by_purpose,
                              "attribution_scoring_calls": r_score["attribution_scoring_calls"],
                              "budget_matched_to_iterations": max(1, p.n_iterations),
                              "iterations": [it.to_dict() for it in r.iterations]},
            "self_reflective": {**p_row, "generator_model": p.generator_model,
                                "verifier_model": p.verifier_model,
                                "calls_by_purpose": p.calls_by_purpose,
                                "stop_reason": p.stop_reason,
                                "iterations": [it.to_dict() for it in p.iterations]},
        })
        if not quiet:
            print(f"  [{i}/{len(questions)}] {q.qid}: "
                  f"baseline conf={b_row['confidence']:.2f} hall={b_row['hallucination']:.2f}  |  "
                  f"retrieval_only(n={max(1, p.n_iterations)}) conf={r_row['confidence']:.2f} "
                  f"hall={r_row['hallucination']:.2f}  |  "
                  f"self_reflective(n={p.n_iterations},{p.stop_reason}) "
                  f"conf={p_row['confidence']:.2f} hall={p_row['hallucination']:.2f}")

        if i % checkpoint_every == 0:
            write_json(partial_path, {"per_question": per_question})


_ARM_LABEL = {"baseline": "Baseline", "retrieval_only": "Retrieval-Only",
             "self_reflective": "Self-Reflective"}
_FACT_VERIFICATION_METRICS = [
    ("verdict_accuracy", "Verdict Accuracy"), ("retrieval_recall", "Retrieval Recall"),
    ("retrieval_precision", "Retrieval Precision"),
    ("all_gold_retrieved", "All Gold Retrieved"), ("confidence", "Support Ratio"),
    ("hallucination", "Hallucination Rate"), ("llm_calls", "LLM Calls/Q"),
    ("seconds", "Latency"),
]
_QA_METRICS = [
    ("em", "Exact Match"), ("f1", "F1 Score"), ("answer_recall", "Answer Recall"),
    ("retrieval_recall", "Retrieval Recall"), ("retrieval_precision", "Retrieval Precision"),
    ("all_gold_retrieved", "All Gold Retrieved"), ("confidence", "Support Ratio"),
    ("hallucination", "Hallucination Rate"), ("llm_calls", "LLM Calls/Q"),
    ("seconds", "Latency"),
]


def print_comparison_summary(results_path: Path, dataset: str) -> None:
    """The one clean final table --comparison prints: only metrics that are actually
    present in results.json (never invented), computed the same way evaluate_one()/
    aggregate() already compute them - a formatting layer over existing evaluation
    code, not a second one. Column set is chosen by the subset's actual task type
    (fact_verification -> FEVER/custom metrics, qa -> HotpotQA metrics)."""
    data = read_json(results_path)
    pq = data["per_question"]
    task = pq[0].get("task", "qa") if pq else "qa"
    metrics = _FACT_VERIFICATION_METRICS if task == "fact_verification" else _QA_METRICS

    means: dict[str, dict[str, float]] = {}
    for arm in ("baseline", "retrieval_only", "self_reflective"):
        rows = [{k: v for k, v in q[arm].items() if isinstance(v, (int, float))}
               for q in pq]
        means[arm] = aggregate(rows)

    bar = "=" * 68
    thin = "-" * 68
    title = _DATASET_TITLE.get(dataset, dataset.upper())
    print(f"\n{bar}\nSELF-REFLECTIVE RAG — {title.upper()} — n={len(pq)}\n{bar}\n")

    present = [(key, label) for key, label in metrics
              if any(key in means[arm] for arm in means)]
    header = f"{'Metric':<22}{'Baseline':>12}{'Retrieval-Only':>18}{'Self-Reflective':>18}"
    print(header)
    print(thin)
    for key, label in present:
        def fmt(arm: str) -> str:
            v = means[arm].get(key)
            return f"{v:.3f}" if isinstance(v, float) else "-"
        print(f"{label:<22}{fmt('baseline'):>12}{fmt('retrieval_only'):>18}"
             f"{fmt('self_reflective'):>18}")

    cost = data["meta"].get("total_calls", {}).get("generator_llm", {}).get("estimated_cost_usd")
    if cost is not None:
        print(f"\nEstimated total cost : ${cost:.4f} (whole run, all arms - "
             "not separable per-arm from one shared generator instance)")

    print(f"\n{thin}\nSTATISTICAL COMPARISON\n{thin}")
    print(_statistical_comparison(pq))

    print(f"\n{thin}\nKEY FINDING\n{thin}\n")
    print(_key_finding(means, len(pq), task))
    print(f"\n{bar}\nEXPERIMENT COMPLETED\n{bar}")


def _statistical_comparison(pq: list[dict]) -> str:
    """Paired Wilcoxon signed-rank test + bootstrap 95% CI on the mean difference,
    for each of the three pairwise arm comparisons - reusing
    scripts/analyze_results.py's paired_significance() exactly, not a second
    implementation. n this small will often not reach p<0.05 on any metric; that is
    a true, honestly-reported result, not a bug to hide."""
    lines = []
    for arm_a, arm_b in COMPARISONS:
        lines.append(f"\n{_ARM_LABEL[arm_a]} → {_ARM_LABEL[arm_b]}")
        any_metric = False
        for m in key_metrics_for(pq):
            s = paired_significance(pq, arm_a, arm_b, m)
            if s is None:
                continue
            any_metric = True
            flag = "  *p<0.05*" if s["wilcoxon_p"] == s["wilcoxon_p"] and s["wilcoxon_p"] < 0.05 else ""
            lines.append(f"    {m:<18} {s['mean_a']:.3f} -> {s['mean_b']:.3f}  "
                        f"diff={s['mean_diff']:+.3f}  "
                        f"95% CI=[{s['ci95_low']:+.3f}, {s['ci95_high']:+.3f}]  "
                        f"p={s['wilcoxon_p']:.4f}{flag}")
        if not any_metric:
            lines.append("    (fewer than 3 paired observations for every metric - "
                         "not enough data for a significance test)")
    return "\n".join(lines)


def _key_finding(means: dict[str, dict[str, float]], n: int, task: str) -> str:
    """Generated from the actual aggregated numbers - never hard-coded."""
    b, s = means["baseline"], means["self_reflective"]
    lines = []
    accuracy_key = "verdict_accuracy" if task == "fact_verification" else "f1"
    accuracy_label = "verdict accuracy" if task == "fact_verification" else "F1"
    if accuracy_key in b:
        best_arm = max(("baseline", "retrieval_only", "self_reflective"),
                       key=lambda a: means[a].get(accuracy_key, -1))
        lines.append(f"Highest {accuracy_label}: {_ARM_LABEL[best_arm]} "
                     f"({means[best_arm][accuracy_key]:.3f}).")
    if "confidence" in b and "confidence" in s:
        delta_conf = s["confidence"] - b["confidence"]
        lines.append(f"Self-Reflective vs. Baseline support ratio: "
                     f"{b['confidence']:.3f} -> {s['confidence']:.3f} "
                     f"({delta_conf:+.3f}).")
    if "hallucination" in b and "hallucination" in s:
        delta_hall = s["hallucination"] - b["hallucination"]
        lines.append(f"Self-Reflective vs. Baseline hallucination rate: "
                     f"{b['hallucination']:.3f} -> {s['hallucination']:.3f} "
                     f"({delta_hall:+.3f}).")
    if "llm_calls" in b and "llm_calls" in s:
        lines.append(f"Cost: Self-Reflective used {s['llm_calls']:.1f}x LLM calls/question "
                     f"vs. Baseline's {b['llm_calls']:.1f}.")
    lines.append(f"n={n} questions - treat this as a functional comparison, not a "
                "statistical-significance claim, unless n is large enough to support one.")
    return "\n".join(lines)


def _resolve_subset(dataset: str, n: int, seed: int) -> tuple[Path, Path]:
    """FEVER/HotpotQA: builds (or reuses, if already built for this n/seed) the
    frozen subset and its index, via build_subset() and DenseRetriever.build() - the
    exact functions `main.py subset`/`main.py index` already wrap, not reimplemented.
    Raises a clear, actionable error if the dataset hasn't been downloaded/
    preprocessed yet, rather than silently starting a large network download -
    acquisition stays a distinct, explicit step."""
    processed_path = PROCESSED_DIR / dataset / "questions.jsonl"
    if not processed_path.exists():
        raise SystemExit(
            f"{processed_path} not found - download and preprocess {dataset} first:\n"
            f"  python main.py download --dataset {dataset}"
            + (" --max-pages 200000" if dataset == "fever" else "") + "\n"
            f"  python main.py preprocess --dataset {dataset}")

    subset_name = f"{dataset}_n{n}_seed{seed}"
    subset_dir = SUBSETS_DIR / subset_name
    if not (subset_dir / "manifest.json").exists():
        subset_dir = build_subset(dataset, n=n, seed=seed, name=subset_name)

    index_dir = subset_dir / "index_all_MiniLM_L6_v2"
    if not (index_dir / "vectors.npy").exists():
        _, documents = load_subset(subset_dir)
        retriever = DenseRetriever.build(documents)
        retriever.save(index_dir)
    return subset_dir, index_dir


def resolve_verifier_spec(verifier_mode: str, verifier_llm_override: str | None,
                          generator_spec: str) -> str | None:
    """--verifier-llm, if given, always wins (it is the precise, explicit form).
    Otherwise --verifier same/independent picks the model: same -> None (run_
    experiment() then defaults the verifier to the generator's own spec, Experiment
    A); independent -> DEFAULT_INDEPENDENT_VERIFIER_MODEL (Experiment B), reusing the
    existing independent-verifier machinery (ModelConfig.verifier_is_independent,
    separate BaseLLM instances in run_experiment()) rather than a new one."""
    if verifier_llm_override:
        return verifier_llm_override
    if verifier_mode == "independent":
        return DEFAULT_INDEPENDENT_VERIFIER_MODEL
    return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["fever", "hotpotqa", "custom"],
                    help="which of the three supported experiments to run")
    ap.add_argument("--comparison", action="store_true",
                    help="print one clean final summary table (every run compares "
                         "all three arms regardless; this also selects the quiet "
                         "summary output over verbose per-question logging)")
    ap.add_argument("--debug", action="store_true",
                    help="verbose per-question/retrieval/indexing output instead of "
                         "the clean summary - a logging mode, not a second experiment")
    ap.add_argument("--verifier", choices=["same", "independent"], default="same",
                    help="'same' (default): verifier uses the same model as the "
                         "generator (Experiment A). 'independent': verifier uses "
                         f"{DEFAULT_INDEPENDENT_VERIFIER_MODEL} (Experiment B). "
                         "Overridden by --verifier-llm if that is also given.")
    ap.add_argument("--n", type=int, default=200,
                    help="fever/hotpotqa only: frozen subset size")
    ap.add_argument("--kb-dir", default=str(PROJECT_ROOT / "knowledge_base"),
                    help="custom only: directory of .txt source documents")
    ap.add_argument("--questions-file", default=None,
                    help="custom only: path to the id/claim/label/evidence JSON "
                         "array; defaults to <kb-dir>/questions.json")
    ap.add_argument("--llm", "--generator-llm", dest="llm", default=DEFAULT_GENERATOR_MODEL,
                    help="generator model - never changed silently")
    ap.add_argument("--verifier-llm", default=DEFAULT_VERIFIER_MODEL or None,
                    help="explicit verifier model override; takes precedence over "
                         "--verifier")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--expand-k", type=int, default=3)
    ap.add_argument("--max-iterations", type=int, default=3)
    ap.add_argument("--min-support-ratio", type=float, default=1.0)
    ap.add_argument("--backend", default="auto", choices=["auto", "faiss", "numpy"])
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--name", default=None,
                    help="results/<name> directory; defaults to the subset's own name")
    return ap.parse_args()


def main() -> int:
    import logging

    args = parse_args()
    quiet = not args.debug

    if quiet:
        prev_level = log.level
        log.setLevel(logging.WARNING)
        prev_env = {k: os.environ.get(k) for k in
                   ("TRANSFORMERS_VERBOSITY", "HF_HUB_DISABLE_PROGRESS_BARS")}
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    try:
        if args.dataset == "custom":
            questions_path = Path(args.questions_file or Path(args.kb_dir) / "questions.json")
            subset_dir = build_custom_subset(args.kb_dir, questions_path,
                                             out_dir=SUBSETS_DIR / "custom_knowledge_base")
            index_dir = subset_dir / "index_all_MiniLM_L6_v2"
            _, documents = load_subset(subset_dir)
            retriever = DenseRetriever.build(documents)
            retriever.save(index_dir)
        else:
            subset_dir, index_dir = _resolve_subset(args.dataset, args.n, args.seed)

        verifier_spec = resolve_verifier_spec(args.verifier, args.verifier_llm, args.llm)
        out_dir = run_experiment(
            subset_dir, index_dir, args.llm, verifier_spec, top_k=args.top_k,
            expand_k=args.expand_k, max_iterations=args.max_iterations,
            min_support_ratio=args.min_support_ratio, backend=args.backend,
            seed=args.seed, name=args.name or subset_dir.name, quiet=quiet)
    finally:
        if quiet:
            log.setLevel(prev_level)
            for k, v in prev_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    print_comparison_summary(out_dir / "results.json", args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
