"""Build a searchable corpus from one local file, for ad hoc question answering.

Same target shape as the dataset loaders (Document: doc_id, title, text), but the
source is a single file's own paragraphs rather than a benchmark dump - there is no
supporting-facts structure to define passages for us, so packing paragraphs to a
target size is the whole strategy.

Unlike the dataset pipeline there are no gold answers, so only the `ask` command
makes sense here - not `baseline`, which needs a `Question.answer` to score against.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..config.config import DEFAULT_EMBEDDER, SUBSETS_DIR
from ..utils.io import file_checksum, read_json, write_json
from ..utils.logging_utils import get_logger
from ..utils.schema import Document
from .preprocess import make_doc_id, normalise_ws

log = get_logger()

_PARA = re.compile(r"\n\s*\n+")
_SENT = re.compile(r"(?<=[.!?])\s+")


def read_file_text(path: Path) -> str:
    """Extract plain text from a .txt, .md, or .pdf file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("pip install pypdf to ingest PDF files") from exc
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"unsupported file type: {suffix!r} (supported: .txt, .md, .pdf)")


def chunk_text(text: str, chunk_words: int = 120) -> list[str]:
    """Pack paragraphs into passages of roughly `chunk_words`, splitting any single
    paragraph that alone exceeds the target at sentence boundaries.

    Paragraph-first (rather than a fixed sliding window) keeps a passage's sentences
    topically related, which matters for the verifier: it judges each claim against
    whole passages, so a passage that splices unrelated paragraphs together makes
    entailment judgements noisier for no benefit.
    """
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        if buf:
            chunks.append(normalise_ws(" ".join(buf)))
            buf.clear()

    for para in _PARA.split(text):
        para = normalise_ws(para)
        if not para:
            continue
        words = para.split()
        if len(words) > chunk_words * 1.5:
            for sent in _SENT.split(para):
                sent = sent.strip()
                if not sent:
                    continue
                buf.append(sent)
                buf_len += len(sent.split())
                if buf_len >= chunk_words:
                    flush()
                    buf_len = 0
            continue
        if buf and buf_len + len(words) > chunk_words:
            flush()
            buf_len = 0
        buf.append(para)
        buf_len += len(words)
    flush()
    return chunks


def load_file_as_documents(path: str | Path, chunk_words: int = 120) -> list[Document]:
    """Read a local file and split it into passage-sized Documents."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    chunks = chunk_text(read_file_text(path), chunk_words=chunk_words)
    if not chunks:
        raise ValueError(f"{path} produced no usable text")
    title = path.stem
    doc_id = make_doc_id(title)
    return [Document(doc_id=f"{doc_id}_{i:04d}", title=title, text=c)
            for i, c in enumerate(chunks)]


_SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def load_dir_as_documents(dir_path: str | Path, chunk_words: int = 120) -> list[Document]:
    """Read every supported file directly under `dir_path` and split each into
    passage-sized Documents, same as `load_file_as_documents` but for a whole folder
    (e.g. a knowledge_base directory) instead of one file.

    Not recursive - only top-level files are read. Each file's chunks keep their own
    doc_id prefix (the file's stem), so citations still say which source file a claim
    came from.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(dir_path)
    paths = sorted(p for p in dir_path.iterdir()
                   if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES)
    if not paths:
        raise ValueError(f"{dir_path} has no supported files (.txt, .md, .pdf)")
    documents: list[Document] = []
    for p in paths:
        documents.extend(load_file_as_documents(p, chunk_words=chunk_words))
    return documents


def _dir_checksum(dir_path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    for p in sorted(dir_path.iterdir()):
        if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES:
            h.update(p.name.encode("utf-8"))
            h.update(file_checksum(p).encode("utf-8"))
    return h.hexdigest()[:16]


def build_or_load_dir_index(dir_path: str | Path, chunk_words: int = 120,
                            embedder_name: str = DEFAULT_EMBEDDER,
                            out_dir: Path | None = None, force: bool = False,
                            backend: str = "auto", device: str | None = None,
                            seed: int = 13):
    """Directory counterpart to `build_or_load_file_index`: builds one retriever over
    every supported file in `dir_path`, reusing the cached index if none of those
    files' contents (or the set of files) have changed.
    """
    from ..retrieval.retriever import DenseRetriever

    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(dir_path)
    checksum = _dir_checksum(dir_path)
    out_dir = Path(out_dir or SUBSETS_DIR / f"file_{make_doc_id(dir_path.name)}")
    tag = embedder_name.split("/")[-1].replace("-", "_")
    index_dir = out_dir / f"index_{tag}"
    source_meta = index_dir / "source.json"

    if not force and source_meta.exists():
        prev = read_json(source_meta)
        if prev.get("checksum") == checksum and prev.get("chunk_words") == chunk_words:
            log.info("reusing cached index for %s (unchanged since last build)", dir_path)
            return DenseRetriever.load(index_dir, backend=backend, device=device)
        log.info("%s changed since the cached index was built - rebuilding", dir_path)

    documents = load_dir_as_documents(dir_path, chunk_words=chunk_words)
    retriever = DenseRetriever.build(documents, embedder_name=embedder_name,
                                     backend=backend, seed=seed, device=device)
    retriever.save(index_dir)
    write_json(source_meta, {"source_path": str(dir_path), "checksum": checksum,
                             "chunk_words": chunk_words, "n_passages": len(documents)})
    log.info("built and cached index for %s -> %s (%d passages)",
             dir_path, index_dir, len(documents))
    return retriever


def build_or_load_file_index(path: str | Path, chunk_words: int = 120,
                             embedder_name: str = DEFAULT_EMBEDDER,
                             out_dir: Path | None = None, force: bool = False,
                             backend: str = "auto", device: str | None = None,
                             seed: int = 13):
    """Build a DenseRetriever over one file, reusing the index already on disk if the
    file and settings have not changed since it was last built.

    Same principle as the dataset download/preprocess caching: embedding is the
    expensive step, so a second `ask --file` on the same document should not redo it.
    """
    from ..retrieval.retriever import DenseRetriever

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    checksum = file_checksum(path)
    out_dir = Path(out_dir or SUBSETS_DIR / f"file_{make_doc_id(path.stem)}")
    tag = embedder_name.split("/")[-1].replace("-", "_")
    index_dir = out_dir / f"index_{tag}"
    source_meta = index_dir / "source.json"

    if not force and source_meta.exists():
        prev = read_json(source_meta)
        if prev.get("checksum") == checksum and prev.get("chunk_words") == chunk_words:
            log.info("reusing cached index for %s (unchanged since last build)", path)
            return DenseRetriever.load(index_dir, backend=backend, device=device)
        log.info("%s changed since the cached index was built - rebuilding", path)

    documents = load_file_as_documents(path, chunk_words=chunk_words)
    retriever = DenseRetriever.build(documents, embedder_name=embedder_name,
                                     backend=backend, seed=seed, device=device)
    retriever.save(index_dir)
    write_json(source_meta, {"source_path": str(path), "checksum": checksum,
                             "chunk_words": chunk_words, "n_passages": len(documents)})
    log.info("built and cached index for %s -> %s (%d passages)",
             path, index_dir, len(documents))
    return retriever
