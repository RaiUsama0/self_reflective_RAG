"""FAISS vector index with an exact numpy fallback.

Index types:
  flat - exact inner-product search. Correct by construction; the right default at
         this corpus size, and the reference every approximate index is judged against.
  ivf  - inverted file, trained. Faster on large corpora, approximate: it can miss
         gold passages, which shows up as lower retrieval recall.
  hnsw - graph index, no training step, also approximate.

Vectors are L2-normalised on the way in, so inner product equals cosine similarity.

If FAISS is not installed, an exact numpy index is used instead. That is not a
degradation in correctness - only in speed above roughly 1e5 passages.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..embeddings.embedder import l2_normalize
from ..utils.logging_utils import get_logger

log = get_logger()


def faiss_available() -> bool:
    try:
        import faiss
        return True
    except ImportError:
        return False


class VectorIndex:
    """Cosine-similarity index over passage vectors."""

    def __init__(self, vectors: np.ndarray, index_type: str = "flat",
                 backend: str = "auto", nlist: int = 100, nprobe: int = 10) -> None:
        self.vectors = l2_normalize(np.asarray(vectors, dtype="float32"))
        self.dim = int(self.vectors.shape[1])
        self.index_type = index_type
        self.nprobe = nprobe
        self._faiss = None
        self.backend = "numpy"

        if backend in ("auto", "faiss"):
            if faiss_available():
                self._build_faiss(index_type, nlist, nprobe)
            elif backend == "faiss":
                raise ImportError("faiss requested but not installed: pip install faiss-cpu")
            else:
                log.info("faiss not installed - using the exact numpy index")

    def _build_faiss(self, index_type: str, nlist: int, nprobe: int) -> None:
        import faiss

        n = self.vectors.shape[0]
        if index_type == "ivf":
            nlist = max(1, min(nlist, n // 39)) or 1
            if n < 39 * nlist or nlist < 2:
                log.warning("corpus too small to train IVF (%d vectors) - using flat", n)
                index_type = "flat"
            else:
                quantizer = faiss.IndexFlatIP(self.dim)
                index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
                index.train(self.vectors)
                index.add(self.vectors)
                index.nprobe = nprobe
                self._faiss, self.backend, self.index_type = index, "faiss", "ivf"
                return
        if index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.add(self.vectors)
            self._faiss, self.backend, self.index_type = index, "faiss", "hnsw"
            return

        index = faiss.IndexFlatIP(self.dim)
        index.add(self.vectors)
        self._faiss, self.backend, self.index_type = index, "faiss", "flat"

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices), highest similarity first."""
        q = l2_normalize(np.asarray(queries, dtype="float32"))
        k = min(k, self.size)
        if self._faiss is not None:
            return self._faiss.search(q, k)
        sims = q @ self.vectors.T
        idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(sims.shape[0])[:, None]
        order = np.argsort(-sims[rows, idx], axis=1)
        idx = idx[rows, order]
        return sims[rows, idx], idx

    def save(self, path: str | Path) -> None:
        """Persist the vectors, and the trained FAISS structure when there is one.

        A flat index rebuilds from vectors instantly, so only trained structures
        (ivf, hnsw) are worth writing to disk separately.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        if self._faiss is not None and self.index_type in ("ivf", "hnsw"):
            import faiss

            faiss.write_index(self._faiss, str(path / "faiss.index"))

    @classmethod
    def load(cls, path: str | Path, backend: str = "auto", nprobe: int = 10) -> "VectorIndex":
        path = Path(path)
        vectors = np.load(path / "vectors.npy")
        obj = cls.__new__(cls)
        obj.vectors = vectors
        obj.dim = int(vectors.shape[1])
        obj.nprobe = nprobe
        obj._faiss = None
        obj.backend = "numpy"
        obj.index_type = "flat"
        trained = path / "faiss.index"
        if backend in ("auto", "faiss") and faiss_available():
            import faiss

            if trained.exists():
                index = faiss.read_index(str(trained))
                if hasattr(index, "nprobe"):
                    index.nprobe = nprobe
                obj._faiss, obj.backend = index, "faiss"
                obj.index_type = "ivf" if "IVF" in type(index).__name__ else "hnsw"
            else:
                index = faiss.IndexFlatIP(obj.dim)
                index.add(vectors)
                obj._faiss, obj.backend = index, "faiss"
        return obj
