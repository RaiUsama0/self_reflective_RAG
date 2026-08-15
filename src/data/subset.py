"""Define and freeze evaluation subsets.

Why freeze rather than sample at run time: every experimental arm must see exactly
the same questions and the same corpus, or the comparison is not controlled. This
samples once, writes the questions and corpus to disk with checksums, and every
later run loads that fixed set. Re-running with the same seed reproduces it exactly.

For HotpotQA the corpus is restricted to the paragraphs belonging to the sampled
questions, so distractors from other sampled questions act as extra negatives -
harder, and more realistic, than retrieving within one question's own context.
"""
from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path

from ..config.config import PROCESSED_DIR, SUBSETS_DIR
from ..utils.io import file_checksum, read_jsonl, write_json, write_jsonl
from ..utils.logging_utils import get_logger
from ..utils.schema import Document, Question

log = get_logger()


def build_subset(dataset: str, n: int, seed: int = 13, name: str | None = None,
                 processed_dir: Path | None = None, out_root: Path | None = None,
                 restrict_corpus: bool = True) -> Path:
    """Sample n questions and write a self-contained subset directory."""
    processed = Path(processed_dir or PROCESSED_DIR / dataset)
    q_path, c_path = processed / "questions.jsonl", processed / "corpus.jsonl"
    for p in (q_path, c_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found - run the preprocess step first")

    questions = list(read_jsonl(q_path))
    rng = random.Random(seed)
    rng.shuffle(questions)
    sampled = questions[:n] if n > 0 else questions

    corpus = list(read_jsonl(c_path))
    if restrict_corpus and dataset == "hotpotqa":
        keep: set[str] = set()
        for q in sampled:
            keep.update(q["gold_doc_ids"])
        by_id = {d["doc_id"]: d for d in corpus}
        extra = [d for d in corpus if d["doc_id"] not in keep]
        rng.shuffle(extra)
        target_extra = min(len(extra), max(0, len(keep) * 4))
        keep.update(d["doc_id"] for d in extra[:target_extra])
        corpus = [by_id[i] for i in sorted(keep) if i in by_id]

    name = name or f"{dataset}_n{len(sampled)}_seed{seed}"
    out = Path(out_root or SUBSETS_DIR) / name
    out.mkdir(parents=True, exist_ok=True)

    write_jsonl(out / "questions.jsonl", sampled)
    write_jsonl(out / "corpus.jsonl", corpus)

    corpus_ids = {d["doc_id"] for d in corpus}
    missing = sum(1 for q in sampled if not set(q["gold_doc_ids"]) <= corpus_ids)
    lengths = [len(d["text"].split()) for d in corpus]

    manifest = {
        "name": name,
        "dataset": dataset,
        "seed": seed,
        "n_questions": len(sampled),
        "n_documents": len(corpus),
        "mean_passage_words": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "questions_checksum": file_checksum(out / "questions.jsonl"),
        "corpus_checksum": file_checksum(out / "corpus.jsonl"),
        "questions_with_gold_missing_from_corpus": missing,
    }
    write_json(out / "manifest.json", manifest)

    log.info("subset %s: %d questions, %d passages", name, len(sampled), len(corpus))
    if missing:
        log.warning("%d questions have gold passages absent from the corpus - "
                    "retrieval recall is capped below 1.0 for those", missing)
    return out


def load_subset(path: str | Path) -> tuple[list[Question], list[Document]]:
    """Load a frozen subset into the dataclasses the pipeline expects."""
    path = Path(path)
    questions = [
        Question(qid=r["qid"], question=r["question"], answer=r["answer"],
                 gold_doc_ids=r.get("gold_doc_ids", []), meta=r.get("meta", {}))
        for r in read_jsonl(path / "questions.jsonl")
    ]
    documents = [
        Document(doc_id=r["doc_id"], title=r.get("title", ""), text=r["text"])
        for r in read_jsonl(path / "corpus.jsonl")
    ]
    return questions, documents
