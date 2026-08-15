# Self-Reflective RAG: Improving Attribution via Iterative Verification

MSc dissertation project — Rai Usama, Keele University (CSC-44120).
Supervisor: Marco Ortolani.

**Current status:** environment, data pipeline, single-pass baseline and the
verification loop, the N-iteration ablation, and human validation of the verifier are
implemented. Still to do: the full experimental comparison over HotpotQA and FEVER.

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
| `llm_calls`, `seconds` | cost — report alongside accuracy |

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
src/generator/llm.py         HuggingFace / OpenAI-compatible backends; prompts
src/pipeline/baseline.py     single-pass RAG + evaluation metrics
src/pipeline/verifier.py     claim decomposition + claim-level verification
src/pipeline/refine.py       query reformulation from unsupported claims
src/pipeline/self_reflective.py  the retrieve-generate-verify-refine loop
src/pipeline/interactive.py  side-by-side comparison for one question
src/utils/                   logging, seeding, IO, checksums, schema, env check
main.py                      CLI entry point for every stage
```

