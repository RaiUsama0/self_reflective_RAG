"""Dense retriever: encodes a corpus once, then answers top-k queries against it.

Building the index is the expensive step - minutes of GPU time for a large corpus.
Build once with `main.py index`, save, and every experiment loads it in seconds.
Both arms of the comparison then retrieve over byte-identical vectors, which is what
makes the experiment controlled.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..embeddings.embedder import BaseEmbedder, build_embedder
from ..utils.io import read_jsonl, write_json, write_jsonl
from ..utils.logging_utils import get_logger
from ..utils.schema import Document, RetrievedDoc
from .faiss_index import VectorIndex

log = get_logger()


class DenseRetriever:
    def __init__(self, documents: Sequence[Document], embedder: BaseEmbedder,
                 index: VectorIndex, seed: int = 13) -> None:
        self.documents = list(documents)
        self.embedder = embedder
        self.index = index
        self.seed = seed

    @classmethod
    def build(cls, documents: Sequence[Document], embedder_name: str = "tfidf",
              index_type: str = "flat", backend: str = "auto", seed: int = 13,
              batch_size: int = 64, device: str | None = None,
              nlist: int = 100, nprobe: int = 10) -> "DenseRetriever":
        texts = [d.text for d in documents]
        embedder = build_embedder(embedder_name, corpus=texts, device=device,
                                  batch_size=batch_size, seed=seed)
        log.info("encoding %d passages with %s (dim=%d)",
                 len(texts), embedder.name, embedder.dim)
        vectors = embedder.encode_passages(texts)
        index = VectorIndex(vectors, index_type=index_type, backend=backend,
                            nlist=nlist, nprobe=nprobe)
        log.info("index ready: backend=%s type=%s", index.backend, index.index_type)
        return cls(documents, embedder, index, seed=seed)

    def retrieve(self, query: str, k: int = 5,
                 exclude: Iterable[str] = ()) -> list[RetrievedDoc]:
        exclude = set(exclude)
        raw_k = min(self.index.size, k + len(exclude))
        scores, idx = self.index.search(self.embedder.encode_queries([query]), raw_k)
        out: list[RetrievedDoc] = []
        for score, i in zip(scores[0], idx[0]):
            doc = self.documents[int(i)]
            if doc.doc_id in exclude:
                continue
            out.append(RetrievedDoc(doc_id=doc.doc_id, text=doc.text, title=doc.title,
                                    score=float(score), rank=len(out)))
            if len(out) == k:
                break
        return out

    def retrieve_batch(self, queries: Sequence[str], k: int = 5) -> list[list[RetrievedDoc]]:
        """Batched retrieval - materially faster than looping when encoding on GPU."""
        scores, idx = self.index.search(self.embedder.encode_queries(list(queries)), k)
        results = []
        for row_scores, row_idx in zip(scores, idx):
            docs = []
            for rank, (score, i) in enumerate(zip(row_scores, row_idx)):
                d = self.documents[int(i)]
                docs.append(RetrievedDoc(doc_id=d.doc_id, text=d.text, title=d.title,
                                         score=float(score), rank=rank))
            results.append(docs)
        return results

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.index.save(path)
        write_jsonl(path / "documents.jsonl", (d.to_dict() for d in self.documents))
        write_json(path / "meta.json", {
            "embedder": self.embedder.name,
            "dim": self.embedder.dim,
            "n_documents": len(self.documents),
            "index_type": self.index.index_type,
            "index_backend": self.index.backend,
            "seed": self.seed,
        })
        log.info("saved index to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path, backend: str = "auto",
             device: str | None = None, nprobe: int = 10) -> "DenseRetriever":
        path = Path(path)
        meta = __import__("json").loads((path / "meta.json").read_text())
        documents = [Document(**row) for row in read_jsonl(path / "documents.jsonl")]
        index = VectorIndex.load(path, backend=backend, nprobe=nprobe)
        embedder = build_embedder(meta["embedder"], corpus=[d.text for d in documents],
                                  device=device, seed=meta.get("seed", 13))
        log.info("loaded index from %s (embedder=%s, %d passages)",
                 path, meta["embedder"], len(documents))
        return cls(documents, embedder, index, seed=meta.get("seed", 13))
