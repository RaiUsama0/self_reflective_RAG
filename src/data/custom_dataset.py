"""Custom FEVER-like knowledge-base dataset: claim/label/evidence questions.json
paired with plain-text documents in a directory (knowledge_base/ by default),
adapted into the exact frozen-subset shape (questions.jsonl + corpus.jsonl +
manifest.json) that HotpotQA and FEVER subsets already use - so the existing 3-arm
pipeline (run_experiment() in scripts/run_subset_experiment.py) runs it completely
unchanged, with zero new evaluation code.

Reuses rather than reimplements:
  - src.data.ingest.load_dir_as_documents for chunking - the project's one chunking
    strategy, the same one `main.py ask --file` already uses.
  - the existing Question/Document schema - no second dataset class.
  - DenseRetriever.build/.save for indexing, called by the caller exactly like any
    other subset's index - no second retrieval or FAISS implementation.

Gold evidence resolution happens only here, at load time, and never touches
retrieval, generation, or verification: a claim's "evidence": ["hospital.txt"] is
resolved to every chunk doc_id that hospital.txt was split into, so `gold_doc_ids`
ends up holding real, retrievable ids - exactly what `all_gold_retrieved` /
`retrieval_recall` / `retrieval_precision` already expect, unmodified.
"""
from __future__ import annotations

from pathlib import Path

from ..config.config import SUBSETS_DIR
from ..utils.io import file_checksum, read_json, write_json, write_jsonl
from ..utils.logging_utils import get_logger
from .ingest import load_dir_as_documents

log = get_logger()


def build_custom_subset(kb_dir: str | Path, questions_path: str | Path,
                        out_dir: str | Path | None = None, chunk_words: int = 120,
                        name: str = "custom_knowledge_base") -> Path:
    """Reads `kb_dir`/*.txt + `questions_path` (the id/claim/label/evidence JSON
    array), and writes a frozen subset directory in the shape load_subset() already
    reads. Only `question` (== claim text) ever reaches the retriever/generator/
    verifier downstream - `answer` (the gold label) and `gold_doc_ids` (resolved
    evidence) are read only by evaluate_one(), after inference, same as every other
    dataset in this project.
    """
    kb_dir = Path(kb_dir)
    questions_path = Path(questions_path)
    if not kb_dir.is_dir():
        raise NotADirectoryError(kb_dir)
    if not questions_path.exists():
        raise FileNotFoundError(questions_path)

    documents = load_dir_as_documents(kb_dir, chunk_words=chunk_words)
    if not documents:
        raise ValueError(f"{kb_dir} produced no usable passages")

    chunks_by_file: dict[str, list[str]] = {}
    for d in documents:
        chunks_by_file.setdefault(d.title, []).append(d.doc_id)

    raw_questions = read_json(questions_path)
    questions: list[dict] = []
    for row in raw_questions:
        evidence_files = [Path(e).stem for e in row.get("evidence", [])]
        gold_doc_ids: list[str] = []
        for f in evidence_files:
            if f not in chunks_by_file:
                raise KeyError(
                    f"question {row['id']!r} references evidence file '{f}.txt', "
                    f"which is not present in {kb_dir} "
                    f"(available: {sorted(chunks_by_file)})")
            for cid in chunks_by_file[f]:
                if cid not in gold_doc_ids:
                    gold_doc_ids.append(cid)
        questions.append({
            "qid": str(row["id"]),
            "question": row["claim"],
            "answer": row["label"],
            "gold_doc_ids": gold_doc_ids,
            "meta": {"task": "fact_verification", "dataset": "custom"},
        })

    out_dir = Path(out_dir or SUBSETS_DIR / name)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "questions.jsonl", questions)
    write_jsonl(out_dir / "corpus.jsonl", (d.to_dict() for d in documents))

    manifest = {
        "name": name, "dataset": "custom", "n_questions": len(questions),
        "n_documents": len(documents), "n_source_files": len(chunks_by_file),
        "questions_checksum": file_checksum(out_dir / "questions.jsonl"),
        "corpus_checksum": file_checksum(out_dir / "corpus.jsonl"),
        "source_kb_dir": str(kb_dir), "source_questions_path": str(questions_path),
    }
    write_json(out_dir / "manifest.json", manifest)
    log.info("custom subset %s: %d questions over %d passages from %d source files",
             name, len(questions), len(documents), len(chunks_by_file))
    return out_dir
