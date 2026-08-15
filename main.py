#!/usr/bin/env python3
"""Self-Reflective RAG - command line entry point.

Pipeline order (each step consumes the previous step's output):

    python main.py check                                   toolchain and GPU
    python main.py download  --dataset hotpotqa            -> data/raw/
    python main.py preprocess --dataset hotpotqa           -> data/processed/
    python main.py subset    --dataset hotpotqa --n 200    -> data/subsets/
    python main.py index     --subset <path>               -> <subset>/index_<tag>/
    python main.py baseline  --subset <path> --index <path> -> results/<name>/

FEVER needs its page dump as well, since it ships claims but not page text:

    python main.py download --dataset fever --max-pages 200000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv(Path(__file__).resolve().parent / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.config.config import (
    DEFAULT_EMBEDDER, DEFAULT_SEED, DEFAULT_TOP_K,
    EmbeddingConfig, GenerationConfig, RetrievalConfig, RunConfig,
)
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

log = get_logger()


def cmd_check(args: argparse.Namespace) -> int:
    from src.utils.env_check import check_environment

    return check_environment()


def cmd_download(args: argparse.Namespace) -> int:
    from src.data.download import (
        download_fever_claims, download_fever_pages, download_hotpotqa,
    )

    if args.dataset == "hotpotqa":
        download_hotpotqa(split=args.split, cache_dir=args.cache_dir, limit=args.limit,
                          force=args.force)
    else:
        claims_path = download_fever_claims(split="labelled_dev", cache_dir=args.cache_dir,
                                            limit=args.limit, force=args.force)
        wanted = None
        if not args.full_pages:
            from src.utils.io import read_jsonl

            wanted = {r["evidence_wiki_url"] for r in read_jsonl(claims_path)
                     if r.get("evidence_wiki_url")}
            log.info("targeting the %d evidence titles referenced by %s",
                     len(wanted), claims_path)
        download_fever_pages(cache_dir=args.cache_dir, max_pages=args.max_pages,
                             wanted_titles=wanted, scan_limit=args.scan_limit,
                             distractor_pages=args.distractor_pages,
                             force=args.force)
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    from src.data.preprocess import preprocess_fever, preprocess_hotpotqa

    if args.dataset == "hotpotqa":
        preprocess_hotpotqa(split=args.split, force=args.force)
    else:
        preprocess_fever(split="labelled_dev", force=args.force)
    return 0


def cmd_subset(args: argparse.Namespace) -> int:
    from src.data.subset import build_subset

    set_seed(args.seed)
    out = build_subset(args.dataset, n=args.n, seed=args.seed, name=args.name)
    print(json.dumps(json.loads((out / "manifest.json").read_text()), indent=2))
    print(f"\nnext:  python main.py index --subset {out}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from src.data.subset import load_subset
    from src.retrieval.retriever import DenseRetriever

    set_seed(args.seed)
    subset = Path(args.subset)
    questions, documents = load_subset(subset)
    retriever = DenseRetriever.build(
        documents, embedder_name=args.embedder, index_type=args.index_type,
        backend=args.backend, seed=args.seed, batch_size=args.batch_size,
        device=args.device, nlist=args.nlist, nprobe=args.nprobe)

    tag = args.embedder.split("/")[-1].replace("-", "_")
    out = Path(args.out) if args.out else subset / f"index_{tag}"
    retriever.save(out)

    if args.smoke_test and questions:
        q = questions[0]
        print(f"\nsmoke test - {q.question}")
        print(f"  gold: {q.gold_doc_ids}")
        for d in retriever.retrieve(q.question, k=5):
            mark = "*" if d.doc_id in q.gold_doc_ids else " "
            print(f"  {mark} {d.score:.3f}  {d.doc_id}")
        print("  (* = gold supporting passage)")

    print(f"\nnext:  python main.py baseline --subset {subset} --index {out}")
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    from src.data.subset import load_subset
    from src.generator.llm import build_llm
    from src.pipeline.baseline import BaselineRAG, run_baseline
    from src.retrieval.retriever import DenseRetriever

    set_seed(args.seed)
    questions, documents = load_subset(args.subset)

    if args.index:
        retriever = DenseRetriever.load(args.index, backend=args.backend,
                                        device=args.device, nprobe=args.nprobe)
        if len(retriever.documents) != len(documents):
            log.error("index has %d passages but the subset has %d - rebuild the index",
                      len(retriever.documents), len(documents))
            return 1
        embedder_name = retriever.embedder.name
    else:
        retriever = DenseRetriever.build(documents, embedder_name=args.embedder,
                                         backend=args.backend, seed=args.seed,
                                         device=args.device)
        embedder_name = args.embedder

    llm = build_llm(args.llm) if args.llm != "hf" else build_llm(args.llm)
    cfg = RunConfig(
        name=args.name, subset=str(args.subset), index_dir=str(args.index or ""),
        seed=args.seed,
        embedding=EmbeddingConfig(model=embedder_name, device=args.device),
        retrieval=RetrievalConfig(top_k=args.top_k, backend=retriever.index.backend,
                                  index_type=retriever.index.index_type),
        generation=GenerationConfig(model=args.llm, max_new_tokens=args.max_new_tokens),
    )
    system = BaselineRAG(retriever, llm, top_k=args.top_k,
                         max_new_tokens=args.max_new_tokens)
    summary = run_baseline(system, questions, cfg)
    print(json.dumps(summary, indent=2))
    return 0


def _build_systems(args, questions, documents, retriever=None):
    """Assemble both arms over the same retriever, LLM and prompt.

    Sharing every component is the point: the only difference between the two is the
    verification loop, so any measured change is attributable to it.

    `retriever`, if already built (e.g. by `--file` ingestion), is used as-is instead
    of building or loading one from `documents`/`args.index`.
    """
    from src.generator.llm import build_llm
    from src.pipeline.baseline import BaselineRAG
    from src.pipeline.refine import QueryReformulator
    from src.pipeline.self_reflective import SelfReflectiveRAG
    from src.pipeline.verifier import Verifier
    from src.retrieval.retriever import DenseRetriever

    if retriever is None:
        if getattr(args, "index", None):
            retriever = DenseRetriever.load(args.index, backend=args.backend,
                                            device=args.device)
            if len(retriever.documents) != len(documents):
                raise SystemExit(
                    f"index has {len(retriever.documents)} passages but the subset has "
                    f"{len(documents)} - rebuild the index for this subset")
        else:
            retriever = DenseRetriever.build(documents, embedder_name=args.embedder,
                                             backend=args.backend, seed=args.seed,
                                             device=args.device)

    llm = build_llm(args.llm)
    verifier = Verifier(llm)
    baseline = BaselineRAG(retriever, llm, top_k=args.top_k,
                           max_new_tokens=args.max_new_tokens)
    proposed = SelfReflectiveRAG(
        retriever, llm, verifier, QueryReformulator(llm), top_k=args.top_k,
        max_iterations=args.max_iterations,
        min_support_ratio=args.min_support_ratio,
        stop_on_no_improvement=not args.no_early_stop,
        max_new_tokens=args.max_new_tokens)
    return retriever, llm, verifier, baseline, proposed


def cmd_ask(args: argparse.Namespace) -> int:
    """Ask a question against --file or --subset; prints one tabular report per
    question (initial answer, reflection stats, final answer, call counts), plus a
    mean confidence/hallucination summary for more than one question. Same report
    whether the corpus is a HotpotQA/FEVER subset or an ingested file like
    knowledge_base, and whether questions come from --question, --questions-file, or
    the live REPL."""
    from src.data.subset import load_subset
    from src.pipeline.interactive import batch_table_ask, print_table_report, repl_table

    set_seed(args.seed)

    retriever = None
    if args.file:
        from src.data.ingest import build_or_load_dir_index, build_or_load_file_index

        if Path(args.file).is_dir():
            retriever = build_or_load_dir_index(
                args.file, chunk_words=args.chunk_words, embedder_name=args.embedder,
                backend=args.backend, device=args.device, seed=args.seed,
                force=args.force_reindex)
        else:
            retriever = build_or_load_file_index(
                args.file, chunk_words=args.chunk_words, embedder_name=args.embedder,
                backend=args.backend, device=args.device, seed=args.seed,
                force=args.force_reindex)
        questions, documents = [], retriever.documents
        log.info("using uploaded file %s (%d passages)", args.file, len(documents))
    elif args.subset:
        subset = Path(args.subset)
        questions, documents = load_subset(subset)
    else:
        raise SystemExit("ask requires --file or --subset")

    *_, baseline, proposed = _build_systems(args, questions, documents,
                                            retriever=retriever)

    if args.questions_file:
        qs = [line.strip() for line in
              Path(args.questions_file).read_text(encoding="utf-8").splitlines()
              if line.strip()]
        if not qs:
            raise SystemExit(f"{args.questions_file} has no questions (one per line)")
        batch_table_ask(qs, baseline, proposed)
    elif args.question:
        print_table_report(args.question, baseline, proposed)
    else:
        intro = (f"File loaded: {args.file} ({len(documents)} passages). "
                 "What would you like to ask about it? (or 'quit' to exit)"
                 if args.file else None)
        repl_table(baseline, proposed, intro=intro)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="main.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="verify the toolchain and GPU").set_defaults(func=cmd_check)

    p = sub.add_parser("ask", help="ask a question; tabular baseline/self-reflective report")
    p.add_argument("--question", default=None,
                   help="one question; omitted -> interactive prompt")
    p.add_argument("--questions-file", default=None,
                   help="file with one question per line; asks each in turn against "
                        "the same --file/--subset, printing one table per question "
                        "plus a mean confidence/hallucination summary")
    p.add_argument("--file", default=None,
                   help="ask questions of one local file (.txt/.md/.pdf), or every "
                        "such file in a folder (e.g. --file knowledge_base), instead "
                        "of a dataset subset; the index is built once and cached, and "
                        "rebuilt automatically if the file(s) change")
    p.add_argument("--chunk-words", type=int, default=120,
                   help="--file only: target passage size in words")
    p.add_argument("--force-reindex", action="store_true",
                   help="--file only: rebuild the cached index even if unchanged")
    p.add_argument("--subset", default=None,
                   help="frozen subset to retrieve over; --file or --subset is required")
    p.add_argument("--index", default=None, help="prebuilt index for that subset")
    p.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    p.add_argument("--llm", default="openai:gpt-4o-mini",
                   help="hf:<model_id> | openai:<model>")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--min-support-ratio", type=float, default=1.0)
    p.add_argument("--no-early-stop", action="store_true",
                   help="disable the no-improvement stopping rule")
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--backend", default="auto", choices=["auto", "faiss", "numpy"])
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("download", help="download raw datasets")
    p.add_argument("--dataset", required=True, choices=["hotpotqa", "fever"])
    p.add_argument("--split", default="validation")
    p.add_argument("--limit", type=int, default=0, help="0 = all rows")
    p.add_argument("--max-pages", type=int, default=0, help="FEVER wiki pages cap")
    p.add_argument("--scan-limit", type=int, default=0,
                   help="FEVER: rows to scan for targeted pages before giving up "
                        "(0 = scan the whole dump - it is roughly alphabetical, so "
                        "wanted titles are spread across it, not front-loaded); "
                        "ignored with --full-pages")
    p.add_argument("--distractor-pages", type=int, default=1000,
                   help="FEVER: non-gold pages to keep incidentally while scanning "
                        "for targeted pages, so retrieval is not trivially easy")
    p.add_argument("--full-pages", action="store_true",
                   help="FEVER: download the whole wiki_pages dump (~5GB) instead of "
                        "streaming just the pages referenced by the downloaded claims")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--force", action="store_true", help="redownload even if cached")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("preprocess", help="raw -> corpus.jsonl + questions.jsonl")
    p.add_argument("--dataset", required=True, choices=["hotpotqa", "fever"])
    p.add_argument("--split", default="validation")
    p.add_argument("--force", action="store_true", help="reprocess even if cached")
    p.set_defaults(func=cmd_preprocess)

    p = sub.add_parser("subset", help="freeze an evaluation subset")
    p.add_argument("--dataset", required=True, choices=["hotpotqa", "fever"])
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--name", default=None)
    p.set_defaults(func=cmd_subset)

    p = sub.add_parser("index", help="encode a subset's corpus and save the index")
    p.add_argument("--subset", required=True)
    p.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    p.add_argument("--index-type", default="flat", choices=["flat", "ivf", "hnsw"])
    p.add_argument("--backend", default="auto", choices=["auto", "faiss", "numpy"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--nlist", type=int, default=100)
    p.add_argument("--nprobe", type=int, default=10)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out", default=None)
    p.add_argument("--smoke-test", action="store_true")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("baseline", help="run single-pass RAG over a subset")
    p.add_argument("--subset", required=True)
    p.add_argument("--index", default=None, help="prebuilt index; otherwise built now")
    p.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    p.add_argument("--llm", default="openai:gpt-4o-mini",
                   help="hf:<model_id> | openai:<model>")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--backend", default="auto", choices=["auto", "faiss", "numpy"])
    p.add_argument("--device", default=None)
    p.add_argument("--nprobe", type=int, default=10)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--name", default="baseline")
    p.set_defaults(func=cmd_baseline)

    return ap


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
