# Self-Reflective RAG: Improving Attribution via Iterative Verification

MSc dissertation project — Rai Usama, Keele University (CSC-44120).
Supervisor: Marco Ortolani.

**Current status:** environment, data pipeline, three experimental arms (baseline,
retrieval-only, self-reflective), independent generator/verifier configuration, the
N-iteration ablation, and human validation of the verifier are implemented. Still to
do: the full-scale experimental comparison over HotpotQA and FEVER at n=200.

## Experimental arms

Three arms isolate what actually causes any measured difference (see
`scripts/run_subset_experiment.py`'s docstring for the full reasoning):

| Arm | Mechanism | Answers |
|---|---|---|
| `BaselineRAG` | retrieve once, generate once | the control condition |
| `RetrievalOnlyRAG` | same retrieval-expansion schedule as self-reflective, **no verifier** | does more evidence alone help? |
| `SelfReflectiveRAG` | retrieve → generate → verify → reformulate → retry | does verification help beyond that? |

The generator and verifier are independently configurable — `GENERATOR_MODEL`/
`VERIFIER_MODEL` env vars, or `--llm`/`--verifier-llm` flags — so the same script runs
both **Experiment A** (same model for both roles) and **Experiment B** (an independent
verifier model), addressing the self-verification circularity risk of using one model
to judge its own output. Leaving `VERIFIER_MODEL`/`--verifier-llm` unset means
same-model (Experiment A, the default); every run records which model played which
role in `config.json` and `results.json`'s `meta` block, never silently.

```bash
# Experiment A + B + the standardized comparison table, one dataset:
python main.py subset  --dataset fever --n 200 --seed 13
python main.py index   --subset data/subsets/fever_n200_seed13
python scripts/run_subset_experiment.py --subset data/subsets/fever_n200_seed13 \
    --index data/subsets/fever_n200_seed13/index_all_MiniLM_L6_v2
python scripts/ablation_n_iterations.py --results results/fever_n200_seed13/results.json
python scripts/analyze_results.py --results-dir results/fever_n200_seed13

# Independent-verifier comparison (Experiment A vs B) + claim-level agreement:
python scripts/run_verifier_independence.py \
    --subset data/subsets/fever_n200_seed13 \
    --index data/subsets/fever_n200_seed13/index_all_MiniLM_L6_v2 \
    --independent-verifier-llm openai:gpt-4.1-mini
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py check               # versions, GPU, FAISS
```

`torch` and FAISS are imported lazily and never required — `numpy`/`scikit-learn`
cover retrieval and embedding fallbacks. The `openai` package and network access are
required, though: the test suite makes one real call to the OpenAI API.

## Verify it works

```bash
python tests/test_pipeline.py      # needs OPENAI_API_KEY and network for one test
python tests/test_arms.py          # fully offline: all 3 arms, stopping rules, gold-leakage proof
```

## Why the pipeline is split into these stages

**Download → preprocess** decouples the raw dumps from the HuggingFace cache, so
later stages don't break when the cache is cleared.

**Preprocess → subset** freezes the evaluation set. Sampling once and writing the
questions to disk with checksums is what makes the eventual baseline-vs-proposed
comparison controlled: both arms read byte-identical questions and corpus, provably.

**Subset → index** encodes the corpus once. Encoding is minutes of GPU time; both
arms then load the same vectors in seconds and retrieve over identical embeddings.

## What gets measured

| Metric | Meaning |
|---|---|
| `em`, `f1`, `answer_recall` | answer quality (SQuAD normalisation; `answer_recall` = gold span appears in the free-form answer) |
| `retrieval_recall`, `retrieval_precision` | against gold supporting passages |
| `all_gold_retrieved` | fraction of questions where every gold passage was retrieved — the ceiling on what any generator can do |
| `n_citations`, `abstained` | citation behaviour |
| `support_ratio` (`confidence`), `hallucination_rate` | attribution — share of claims a passage actually backs, from `Verifier`; baseline/retrieval-only get an *evaluation-only* verification pass (not part of their own mechanism) so all three arms are comparable on this metric |
| `llm_calls`, `calls_by_purpose`, `seconds`, `estimated_cost_usd` | cost, broken down by generation/decompose/verify/reformulate — report alongside accuracy, never claim improvement without it |

`all_gold_retrieved` is the number to watch first. It caps everything downstream:
if the retriever misses the evidence, no amount of verification can recover it.

## Layout

```
data/raw/          downloaded dumps (JSONL)
data/processed/    normalised corpus.jsonl + questions.jsonl per dataset
data/subsets/      frozen evaluation subsets, with manifest and prebuilt index
results/           one directory per run: config.json, results.jsonl, summary.json

src/config/config.py         paths, defaults, RunConfig
src/data/download.py         HotpotQA + FEVER acquisition
src/data/preprocess.py       raw -> corpus + questions
src/data/subset.py           frozen subsets, checksums
src/embeddings/embedder.py   sentence-transformers and TF-IDF backends
src/retrieval/faiss_index.py flat / IVF / HNSW, exact numpy fallback
src/retrieval/retriever.py   dense retrieval, batching, persistence
src/generator/llm.py         HuggingFace / OpenAI-compatible backends; prompts; per-purpose call/token/cost tracking
src/pipeline/baseline.py     BaselineRAG (Arm 1) + evaluation metrics
src/pipeline/retrieval_only.py  RetrievalOnlyRAG (Arm 2) - retrieval-expansion ablation, no verifier
src/pipeline/verifier.py     claim decomposition + claim-level verification
src/pipeline/refine.py       query reformulation from unsupported claims
src/pipeline/self_reflective.py  SelfReflectiveRAG (Arm 3) - the retrieve-generate-verify-refine loop
src/pipeline/interactive.py  side-by-side comparison for one question
src/utils/                   logging, seeding, IO, checksums, schema, env check
main.py                      CLI entry point for every stage

scripts/run_subset_experiment.py  the 3-arm comparison over a frozen subset
scripts/ablation_n_iterations.py  N-iteration cost/benefit sweep (0/1/2/3 rounds)
scripts/analyze_results.py        significance tests + plots + standardized results table
scripts/verifier_agreement.py     claim-level agreement between two verifier models
scripts/run_verifier_independence.py  Experiment A vs B, end to end

tests/test_pipeline.py       preprocessing, retrieval, metrics, verifier defensive parsing
tests/test_arms.py           all 3 arms, stopping rules, independent verifier config, gold-leakage proof, cost tracking
```

## Environment variables

| Variable | Meaning |
|---|---|
| `GENERATOR_MODEL` | default generator model spec (e.g. `openai:gpt-4o-mini`); never overridden silently — CLI flags still win |
| `VERIFIER_MODEL` | default verifier model spec; unset/blank = same model as the generator (Experiment A) |
| `OPENAI_API_KEY` | required for any `openai:<model>` spec |

