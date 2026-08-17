"""Turn raw dumps into the two files every later stage consumes.

    data/processed/<dataset>/corpus.jsonl      doc_id, title, text
    data/processed/<dataset>/questions.jsonl   qid, question, answer, gold_doc_ids

Normalising both datasets to this shape is what lets one retriever, one baseline and
one metric suite serve both without special cases.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..config.config import PROCESSED_DIR, RAW_DIR
from ..utils.io import read_jsonl, write_json, write_jsonl
from ..utils.logging_utils import get_logger

log = get_logger()

_WS = re.compile(r"\s+")


def normalise_ws(text: str) -> str:
    return _WS.sub(" ", str(text)).strip()


def make_doc_id(title: str) -> str:
    """Stable, filesystem-safe id derived from a page title."""
    return _WS.sub("_", str(title).strip())[:120]


def clean_fever_text(text: str) -> str:
    """FEVER's dump escapes brackets and colons; restore them for readability."""
    text = (str(text).replace("-LRB-", "(").replace("-RRB-", ")")
            .replace("-LSB-", "[").replace("-RSB-", "]")
            .replace("-COLON-", ":"))
    return normalise_ws(text)


def preprocess_hotpotqa(split: str = "validation", raw_dir: Path | None = None,
                        out_dir: Path | None = None, force: bool = False) -> Path:
    """Flatten HotpotQA: paragraphs become the corpus, supporting titles the gold ids.

    Skips reprocessing if corpus.jsonl and questions.jsonl already exist, unless
    `force=True`.
    """
    raw_path = Path(raw_dir or RAW_DIR / "hotpotqa") / f"{split}.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found - run the download step first")
    out_dir = Path(out_dir or PROCESSED_DIR / "hotpotqa")
    if not force and (out_dir / "corpus.jsonl").exists() and (out_dir / "questions.jsonl").exists():
        log.info("using cached %s (pass force=True to reprocess)", out_dir)
        return out_dir

    docs: dict[str, dict] = {}
    questions: list[dict] = []

    for row in read_jsonl(raw_path):
        ctx = row["context"]
        titles, sentence_lists = ctx["title"], ctx["sentences"]
        for title, sents in zip(titles, sentence_lists):
            doc_id = make_doc_id(title)
            if doc_id not in docs:
                docs[doc_id] = {"doc_id": doc_id, "title": title,
                                "text": normalise_ws(" ".join(sents))}
        gold = sorted({make_doc_id(t) for t in row["supporting_facts"]["title"]})
        questions.append({
            "qid": str(row["id"]),
            "question": normalise_ws(row["question"]),
            "answer": normalise_ws(row["answer"]),
            "gold_doc_ids": gold,
            "meta": {"type": row.get("type", ""), "level": row.get("level", ""),
                     "dataset": "hotpotqa"},
        })

    n_docs = write_jsonl(out_dir / "corpus.jsonl", docs.values())
    n_q = write_jsonl(out_dir / "questions.jsonl", questions)
    write_json(out_dir / "meta.json",
               {"dataset": "hotpotqa", "split": split, "n_documents": n_docs,
                "n_questions": n_q})
    log.info("hotpotqa -> %s  (%d passages, %d questions)", out_dir, n_docs, n_q)
    return out_dir


def preprocess_fever(split: str = "labelled_dev", raw_dir: Path | None = None,
                     out_dir: Path | None = None, force: bool = False) -> Path:
    """Flatten FEVER. NOT ENOUGH INFO claims are dropped - they carry no gold
    evidence, so retrieval recall is undefined for them. Report that filter in the
    methodology.

    FEVER's raw claims dump has one row per (claim, valid-evidence-set) pair, not one
    row per claim: a claim with several independently-sufficient evidence sets
    appears as several rows sharing the same `id`, same `claim` text, and same
    `label`, differing only in `evidence_wiki_url` (observed up to 59x for one claim
    in labelled_dev). Treating each row as its own question - the original version of
    this function did - inflates some claims' sampling weight by up to 59x relative
    to a claim with a single annotated evidence set, silently violating the
    independent-observations assumption behind every paired significance test run
    downstream. Rows are grouped by id here and collapsed into one question per
    unique claim, with `gold_doc_ids` as the union of every valid evidence document
    across the group - consistent with how HotpotQA's multi-hop gold sets are already
    handled, meaning `all_gold_retrieved` requires every valid evidence document to
    be retrieved, not just one complete evidence set (a deliberately stricter
    standard than FEVER's own official "any one complete set" scoring, chosen for
    consistency with the rest of this codebase rather than to flatter the numbers).

    Skips reprocessing if corpus.jsonl and questions.jsonl already exist, unless
    `force=True`.
    """
    raw_dir = Path(raw_dir or RAW_DIR / "fever")
    claims_path = raw_dir / f"claims_{split}.jsonl"
    pages_path = raw_dir / "wiki_pages.jsonl"
    for p in (claims_path, pages_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found - run the download step first")
    out_dir = Path(out_dir or PROCESSED_DIR / "fever")
    if not force and (out_dir / "corpus.jsonl").exists() and (out_dir / "questions.jsonl").exists():
        log.info("using cached %s (pass force=True to reprocess)", out_dir)
        return out_dir

    docs = ({"doc_id": make_doc_id(row["title"]),
             "title": str(row["title"]).replace("_", " "),
             "text": row["text"]}
            for row in read_jsonl(pages_path))
    n_docs = write_jsonl(out_dir / "corpus.jsonl", docs)

    groups: dict[str, dict] = {}
    n_dropped = 0
    n_rows = 0
    for row in read_jsonl(claims_path):
        n_rows += 1
        if row["label"] not in ("SUPPORTS", "REFUTES"):
            n_dropped += 1
            continue
        qid = str(row["id"])
        group = groups.setdefault(qid, {
            "qid": qid, "question": normalise_ws(row["claim"]), "answer": row["label"],
            "gold_doc_ids": [], "meta": {"task": "fact_verification", "dataset": "fever"},
        })
        if row["label"] != group["answer"]:
            raise ValueError(
                f"claim id {qid} has inconsistent labels across rows "
                f"({group['answer']!r} vs {row['label']!r}) - the group-by-id "
                "collapse assumes one label per claim id; investigate before "
                "reprocessing.")
        if row.get("evidence_wiki_url"):
            doc_id = make_doc_id(row["evidence_wiki_url"])
            if doc_id not in group["gold_doc_ids"]:
                group["gold_doc_ids"].append(doc_id)

    questions = sorted(groups.values(), key=lambda q: q["qid"])
    n_q = write_jsonl(out_dir / "questions.jsonl", questions)
    write_json(out_dir / "meta.json",
               {"dataset": "fever", "split": split, "n_documents": n_docs,
                "n_raw_rows": n_rows, "n_questions": n_q,
                "n_dropped_not_enough_info_rows": n_dropped})
    log.info("fever -> %s  (%d passages, %d raw rows collapsed to %d unique claims, "
             "%d NEI rows dropped)", out_dir, n_docs, n_rows, n_q, n_dropped)
    return out_dir
