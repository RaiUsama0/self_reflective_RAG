"""Turns one or more run_subset_experiment.py results.json files (schema_version 2+,
three arms: baseline, retrieval_only, self_reflective) into the figures, significance
tests, and standardized results table a dissertation results chapter needs.

Three pairwise comparisons, not one
-------------------------------------
    baseline        vs retrieval_only    -> does more evidence alone help? (ISSUE 2)
    retrieval_only  vs self_reflective   -> does verification help beyond that?
    baseline        vs self_reflective   -> the total effect (sum of the two above)

Each is a paired Wilcoxon signed-rank test + bootstrap 95% CI on the mean difference,
over the same questions, since all three arms answer identically-checksummed
questions - see run_subset_experiment.py's docstring. This script does not decide
whether self-reflective RAG "works" - it reports what the paired tests actually show,
including a null or negative result, and leaves the conclusion to the reader.

Usage
-----
    python scripts/analyze_results.py --results-dir results/fever_n200_seed13
    python scripts/analyze_results.py --results-dir results/fever_n200_seed13 \\
                                       --results-dir results/hotpotqa_n200_seed13
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ARMS = ["baseline", "retrieval_only", "self_reflective"]
ARM_LABEL = {"baseline": "Baseline RAG", "retrieval_only": "Retrieval-Only",
            "self_reflective": "Self-Reflective RAG"}
COLORS = {"baseline": "#7a8699", "retrieval_only": "#c98a3a", "self_reflective": "#10233f"}
COMPARISONS = [("baseline", "retrieval_only"), ("retrieval_only", "self_reflective"),
              ("baseline", "self_reflective")]


def load_results(results_dir: Path) -> dict:
    return json.loads((results_dir / "results.json").read_text(encoding="utf-8"))


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, seed: int = 13) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        boot_means[i] = rng.choice(diffs, size=n, replace=True).mean()
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def paired_significance(pq: list[dict], arm_a: str, arm_b: str, metric: str) -> dict | None:
    pairs = [(q[arm_a].get(metric), q[arm_b].get(metric)) for q in pq]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None and a == a and b == b]
    if len(pairs) < 3:
        return None
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    diffs = b - a
    if np.all(diffs == 0):
        return {"metric": metric, "arm_a": arm_a, "arm_b": arm_b, "n": len(pairs),
                "mean_a": float(a.mean()), "mean_b": float(b.mean()), "mean_diff": 0.0,
                "ci95_low": 0.0, "ci95_high": 0.0, "wilcoxon_p": 1.0}
    lo, hi = bootstrap_ci(diffs)
    try:
        _, p = wilcoxon(b, a)
    except ValueError:
        p = float("nan")
    return {"metric": metric, "arm_a": arm_a, "arm_b": arm_b, "n": len(pairs),
           "mean_a": float(a.mean()), "mean_b": float(b.mean()),
           "mean_diff": float(diffs.mean()), "ci95_low": lo, "ci95_high": hi,
           "wilcoxon_p": float(p)}


def key_metrics_for(pq: list[dict]) -> list[str]:
    task = pq[0].get("task", "qa")
    base = ["confidence", "hallucination", "retrieval_recall", "citation_validity",
           "n_citations", "abstained"]
    base += ["verdict_accuracy"] if task == "fact_verification" else ["em", "f1", "answer_recall"]
    return base


def per_arm_confidence_intervals(pq: list[dict]) -> list[dict]:
    """95% bootstrap CI on each arm's own mean for every key metric - distinct from
    paired_significance's CI on the *difference* between two arms. This answers "how
    precisely do we know self-reflective's mean hallucination rate is 0.19?", not
    "is self-reflective's rate different from baseline's?"."""
    out = []
    for arm in ARMS:
        for m in key_metrics_for(pq):
            vals = np.array([q[arm][m] for q in pq if m in q[arm] and q[arm][m] == q[arm][m]])
            if len(vals) < 3:
                continue
            rng = np.random.default_rng(13)
            boot = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                             for _ in range(10000)])
            out.append({"arm": arm, "metric": m, "n": len(vals), "mean": float(vals.mean()),
                       "ci95_low": float(np.percentile(boot, 2.5)),
                       "ci95_high": float(np.percentile(boot, 97.5))})
    return out


def plot_comparison_bars(pq: list[dict], name: str, out_dir: Path) -> None:
    metrics = [m for m in ["confidence", "hallucination", "retrieval_recall",
                            "verdict_accuracy", "f1", "answer_recall", "abstained"]
               if m in pq[0]["baseline"] and m in pq[0].get("retrieval_only", {})]
    means = {arm: [mean(q[arm][m] for q in pq if m in q[arm]) for m in metrics]
            for arm in ARMS}

    x = np.arange(len(metrics))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, arm in enumerate(ARMS):
        ax.bar(x + (i - 1) * width, means[arm], width, label=ARM_LABEL[arm], color=COLORS[arm])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_ylabel("mean value")
    ax.set_title(f"{name}: baseline vs. retrieval-only vs. self-reflective")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_bars.png", dpi=150)
    plt.close(fig)


def plot_decomposed_gain(pq: list[dict], name: str, out_dir: Path) -> None:
    """The headline chart: for each metric, how much of self-reflective's total gain
    over baseline is attributable to more retrieval alone vs. verification beyond
    that - the direct answer to ISSUE 2."""
    metrics = [m for m in key_metrics_for(pq) if m in pq[0]["baseline"]
              and m in pq[0].get("retrieval_only", {})]
    from_retrieval, from_verification = [], []
    for m in metrics:
        b = mean(q["baseline"][m] for q in pq if m in q["baseline"])
        r = mean(q["retrieval_only"][m] for q in pq if m in q["retrieval_only"])
        s = mean(q["self_reflective"][m] for q in pq if m in q["self_reflective"])
        from_retrieval.append(r - b)
        from_verification.append(s - r)

    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, from_retrieval, 0.5, label="gain from more retrieval\n(retrieval_only - baseline)",
          color=COLORS["retrieval_only"])
    ax.bar(x, from_verification, 0.5, bottom=from_retrieval,
          label="gain from verification beyond that\n(self_reflective - retrieval_only)",
          color=COLORS["self_reflective"])
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_ylabel("mean change vs. baseline")
    ax.set_title(f"{name}: where does self-reflective's gain actually come from?")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "decomposed_gain.png", dpi=150)
    plt.close(fig)


def plot_cost(pq: list[dict], name: str, out_dir: Path) -> None:
    calls = [mean(q[arm]["llm_calls"] for q in pq) for arm in ARMS]
    secs = [mean(q[arm]["seconds"] for q in pq) for arm in ARMS]
    labels = [ARM_LABEL[a] for a in ARMS]
    colors = [COLORS[a] for a in ARMS]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    axes[0].bar(labels, calls, color=colors)
    axes[0].set_ylabel("mean LLM calls / question")
    axes[0].set_title("Cost (calls)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].bar(labels, secs, color=colors)
    axes[1].set_ylabel("mean seconds / question")
    axes[1].set_title("Cost (latency)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{name}: cost across arms")
    fig.tight_layout()
    fig.savefig(out_dir / "cost_comparison.png", dpi=150)
    plt.close(fig)


def standardized_table_rows(data: dict) -> list[dict]:
    """One row per arm, in the exact column set requested for the dissertation's
    results chapter: System / Dataset / N / Answer EM / Answer F1 / Retrieval Recall /
    All Gold Retrieved / Support Ratio / Hallucination Rate / Valid Citations /
    Average Iterations / Average LLM Calls / Average Latency / Estimated Cost."""
    pq = data["per_question"]
    meta = data["meta"]
    rows = []
    for arm in ARMS:
        vals = [q[arm] for q in pq]
        n_iter = [len(q[arm].get("iterations", [1])) or 1 for q in pq] if arm != "baseline" else [1] * len(pq)
        model_key = ("verifier_llm" if arm == "self_reflective" else None)
        cost = None
        if arm == "self_reflective" and meta.get("total_calls", {}).get("generator_llm"):
            cost = None
        rows.append({
            "system": ARM_LABEL[arm], "dataset": meta.get("dataset", ""),
            "n": len(pq),
            "generator_model": meta.get("generator_model", ""),
            "verifier_model": meta.get("verifier_model", "") if arm == "self_reflective" else "",
            "answer_em": mean(v["em"] for v in vals if "em" in v) if any("em" in v for v in vals) else None,
            "answer_f1": mean(v["f1"] for v in vals if "f1" in v) if any("f1" in v for v in vals) else None,
            "verdict_accuracy": (mean(v["verdict_accuracy"] for v in vals if "verdict_accuracy" in v)
                                if any("verdict_accuracy" in v for v in vals) else None),
            "retrieval_recall": mean(v["retrieval_recall"] for v in vals if v.get("retrieval_recall") == v.get("retrieval_recall")),
            "all_gold_retrieved": mean(v["all_gold_retrieved"] for v in vals if v.get("all_gold_retrieved") == v.get("all_gold_retrieved")),
            "support_ratio": (mean(v["confidence"] for v in vals if "confidence" in v)
                             if any("confidence" in v for v in vals) else None),
            "hallucination_rate": (mean(v["hallucination"] for v in vals if "hallucination" in v)
                                   if any("hallucination" in v for v in vals) else None),
            "valid_citations": mean(v["n_citations"] for v in vals),
            "citation_validity": (mean(cv) if (cv := [v["citation_validity"] for v in vals
                                                       if "citation_validity" in v
                                                       and v["citation_validity"] == v["citation_validity"]])
                                  else None),
            "avg_iterations": mean(n_iter),
            "avg_llm_calls": mean(v["llm_calls"] for v in vals),
            "avg_latency_seconds": mean(v["seconds"] for v in vals),
            "estimated_cost_usd": cost,
        })
    return rows


def write_standardized_table(rows: list[dict], out_dir: Path) -> None:
    cols = ["system", "dataset", "n", "generator_model", "verifier_model", "answer_em",
           "answer_f1", "verdict_accuracy", "retrieval_recall", "all_gold_retrieved",
           "support_ratio", "hallucination_rate", "valid_citations", "citation_validity",
           "avg_iterations", "avg_llm_calls", "avg_latency_seconds", "estimated_cost_usd"]
    with open(out_dir / "standardized_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    (out_dir / "standardized_table.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def print_standardized_table(rows: list[dict]) -> None:
    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)
    header = ["System", "N", "EM", "F1", "VerdictAcc", "RetRecall", "AllGold",
             "Support", "Halluc", "Cites", "Iters", "Calls", "Latency(s)"]
    print("  ".join(f"{h:<11}" for h in header))
    for r in rows:
        line = [r["system"], r["n"], r["answer_em"], r["answer_f1"], r["verdict_accuracy"],
               r["retrieval_recall"], r["all_gold_retrieved"], r["support_ratio"],
               r["hallucination_rate"], r["valid_citations"], r["avg_iterations"],
               r["avg_llm_calls"], r["avg_latency_seconds"]]
        print("  ".join(f"{fmt(v):<11}" for v in line))


def analyze_one(results_dir: Path) -> dict:
    data = load_results(results_dir)
    pq = data["per_question"]
    name = data["meta"].get("src_file", results_dir.name)
    out_dir = results_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    plot_comparison_bars(pq, name, out_dir)
    plot_decomposed_gain(pq, name, out_dir)
    plot_cost(pq, name, out_dir)

    all_sig = []
    for arm_a, arm_b in COMPARISONS:
        for m in key_metrics_for(pq):
            s = paired_significance(pq, arm_a, arm_b, m)
            if s:
                all_sig.append(s)
    (results_dir / "significance.json").write_text(
        json.dumps({"name": name, "n_questions": len(pq), "comparisons": all_sig}, indent=2),
        encoding="utf-8")

    print(f"\n=== {name} (n={len(pq)}) - significance ===")
    for arm_a, arm_b in COMPARISONS:
        print(f"\n  {ARM_LABEL[arm_a]}  vs  {ARM_LABEL[arm_b]}")
        for s in all_sig:
            if s["arm_a"] != arm_a or s["arm_b"] != arm_b:
                continue
            flag = "  *p<0.05*" if s["wilcoxon_p"] == s["wilcoxon_p"] and s["wilcoxon_p"] < 0.05 else ""
            print(f"    {s['metric']:<18} {s['mean_a']:.3f} -> {s['mean_b']:.3f}  "
                 f"diff={s['mean_diff']:+.3f}  95% CI=[{s['ci95_low']:+.3f}, {s['ci95_high']:+.3f}]  "
                 f"p={s['wilcoxon_p']:.4f}{flag}")

    cis = per_arm_confidence_intervals(pq)
    (results_dir / "confidence_intervals.json").write_text(
        json.dumps({"name": name, "n_questions": len(pq), "per_arm_ci95": cis}, indent=2),
        encoding="utf-8")
    print(f"\n=== {name} - per-arm 95% confidence intervals ===")
    for arm in ARMS:
        print(f"\n  {ARM_LABEL[arm]}")
        for c in cis:
            if c["arm"] != arm:
                continue
            print(f"    {c['metric']:<18} mean={c['mean']:.3f}  "
                 f"95% CI=[{c['ci95_low']:.3f}, {c['ci95_high']:.3f}]  (n={c['n']})")

    rows = standardized_table_rows(data)
    write_standardized_table(rows, results_dir)
    print(f"\n=== {name} - standardized results table ===")
    print_standardized_table(rows)
    print(f"\n  wrote {out_dir}, {results_dir / 'significance.json'}, "
         f"{results_dir / 'confidence_intervals.json'}, "
         f"{results_dir / 'standardized_table.csv'}")
    return {"name": name, "results_dir": str(results_dir), "table_rows": rows}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", action="append", required=True,
                    help="results/<name> directory; repeat for multiple datasets")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    summaries = [analyze_one(Path(d)) for d in args.results_dir]
    all_rows = [row for s in summaries for row in s["table_rows"]]
    write_standardized_table(all_rows, PROJECT_ROOT / "results")
    (PROJECT_ROOT / "results" / "analysis_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nwrote combined table to {PROJECT_ROOT / 'results' / 'standardized_table.csv'}")
    print(f"wrote {PROJECT_ROOT / 'results' / 'analysis_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
