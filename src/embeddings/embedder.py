"""Text embedding.

Two backends behind one interface:

  SentenceTransformerEmbedder - the dense retriever used for real experiments.
  TfidfEmbedder               - deterministic sklearn fallback. No GPU, no download,
                                no network. Used by the tests, useful for debugging
                                the pipeline, and reportable as a sparse baseline.

Queries and passages go through the same encoder here. If you later switch to an
asymmetric model (E5, BGE, GTR), those need "query: " / "passage: " prefixes -
`is_query` is threaded through for exactly that reason.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Unit-normalise rows so inner product equals cosine similarity."""
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class BaseEmbedder:
    name: str = "base"
    dim: int = 0

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        raise NotImplementedError

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts, is_query=True)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts, is_query=False)


class TfidfEmbedder(BaseEmbedder):
    """TF-IDF followed by truncated SVD (latent semantic indexing).

    Must be fitted on the corpus before use; the fitted vocabulary is what places
    queries in the same space as passages.
    """

    name = "tfidf"

    def __init__(self, dim: int = 256, seed: int = 13) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.dim = dim
        self.seed = seed
        self._vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self._svd = TruncatedSVD(n_components=dim, random_state=seed)
        self._fitted = False

    def fit(self, corpus: Sequence[str]) -> "TfidfEmbedder":
        X = self._vec.fit_transform(corpus)
        n_comp = min(self.dim, max(2, min(X.shape) - 1))
        self._svd.set_params(n_components=n_comp)
        self._svd.fit(X)
        self.dim = n_comp
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit(corpus) must be called first")
        return self._svd.transform(self._vec.transform(list(texts))).astype("float32")


class SentenceTransformerEmbedder(BaseEmbedder):
    """Dense embeddings from sentence-transformers. Imported lazily so the package
    works without torch installed."""

    def __init__(self, model_name: str, device: str | None = None,
                 batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for dense retrieval:\n"
                "    pip install sentence-transformers torch"
            ) from exc

        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        self.name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        return np.asarray(
            self.model.encode(list(texts), batch_size=self.batch_size,
                              convert_to_numpy=True, show_progress_bar=False,
                              normalize_embeddings=False),
            dtype="float32",
        )


def build_embedder(model: str, corpus: Sequence[str] | None = None,
                   device: str | None = None, batch_size: int = 64,
                   seed: int = 13) -> BaseEmbedder:
    """`tfidf` (needs the corpus to fit on) or any sentence-transformers model id."""
    if model == "tfidf":
        if corpus is None:
            raise ValueError("the tfidf embedder must be fitted on the corpus")
        return TfidfEmbedder(seed=seed).fit(corpus)
    return SentenceTransformerEmbedder(model, device=device, batch_size=batch_size)
