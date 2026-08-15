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

## Talk to it

`ask` runs both systems on one question and prints them side by side: the baseline's
answer with claim-level verdicts, then each iteration of the loop, then the
attribution and hallucination rates and what the extra LLM calls cost.

```bash
python main.py ask --subset data/subsets/hotpotqa_n200_seed13 \
                   --index  data/subsets/hotpotqa_n200_seed13/index_all_MiniLM_L6_v2 \
                   --llm hf:Qwen/Qwen2.5-7B-Instruct
```

`ask` requires either `--subset` or `--file`. Only a real `--llm` on a real subset
tells you whether verification reduces hallucination.

## Ask questions of your own file

`--file` points `ask` at one local `.txt`, `.md`, or `.pdf` file instead of a dataset
subset. It is chunked into passage-sized documents (paragraphs packed to
`--chunk-words`, default 120) and indexed like any other corpus - same retrieve,
generate, verify, refine loop, same attribution/hallucination numbers.

```bash
python main.py ask --file notes.txt --llm openai:gpt-4o-mini \
                   --question "What was the budget?"
```

`--llm` takes `openai:<model>` or `hf:<model_id>` (default `openai:gpt-4o-mini`).

The index is built once and cached next to `data/subsets/file_<name>/`, keyed on the
file's checksum - asking a second question re-embeds nothing, and editing the file
triggers an automatic rebuild. Pass `--force-reindex` to rebuild anyway (e.g. after
changing `--chunk-words`).

Passages that split a sentence referring back to an earlier passage (e.g. "the
project" resolving to a name introduced in the previous passage) can come back
`INSUFFICIENT EVIDENCE` even when the fact is technically present two passages away -
the generator is deliberately strict about not bridging pronouns across passage
boundaries. A larger `--chunk-words` keeps more of that context together.

## Run on real data

```bash
python main.py download   --dataset hotpotqa            # -> data/raw/
python main.py preprocess --dataset hotpotqa            # -> data/processed/
python main.py subset     --dataset hotpotqa --n 200 --seed 13
python main.py index      --subset data/subsets/hotpotqa_n200_seed13 \
                          --embedder sentence-transformers/all-MiniLM-L6-v2 \
                          --smoke-test
python main.py baseline   --subset data/subsets/hotpotqa_n200_seed13 \
                          --index  data/subsets/hotpotqa_n200_seed13/index_all_MiniLM_L6_v2 \
                          --llm hf:Qwen/Qwen2.5-7B-Instruct --name hotpot_baseline
```

FEVER needs one extra download, because it ships claims but not page text. By default
the page download is streamed and targeted at just the evidence titles referenced by
the claims just downloaded, instead of pulling the full ~5GB dump; pass `--full-pages`
to get the untargeted full dump instead.

```bash
python main.py download   --dataset fever --limit 1000 --max-pages 2000
python main.py preprocess --dataset fever
python main.py subset     --dataset fever --n 30 --seed 13
```

Both `download` and `preprocess` skip work and reuse what is already on disk if their
output file already exists - pass `--force` to redo them anyway (e.g. after changing
`--limit`).

**Known environment issue:** `hotpot_qa` and `fever` on the Hub are still
loading-script datasets. Newer `huggingface_hub` (>=1.0) cannot resolve their
non-namespaced repo id, and newer `datasets` (>=4.0) refuses loading-script datasets
outright - `requirements.txt` pins `datasets==2.21.0` for that reason, but a
`huggingface_hub` new enough for `transformers`/`sentence-transformers` (>=1.5) is too
new for these dataset downloads. If `download` fails with an `HfUriError` or a
`trust_remote_code` error, temporarily run
`pip install "huggingface_hub==0.25.2"`, do the download, then
`pip install -r requirements.txt` again to restore the version the rest of the
pipeline needs. This only has to happen once per dataset, since `download` caches its
output and will not hit the network again.

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

