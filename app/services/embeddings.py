"""Embedding providers behind a single interface.

The rest of the app depends only on ``EmbeddingProvider.embed``; the concrete
implementation is chosen at startup from ``EMBEDDING_MODE``:

  * ``mock``  -> deterministic hash->vector. No downloads, reproducible in CI.
  * ``local`` -> sentence-transformers all-MiniLM-L6-v2 (extra: ``uv sync --extra local``).

A hosted provider (e.g. OpenAI) would be one more subclass implementing the
same two methods. All providers return L2-normalized float lists so cosine
similarity reduces to a dot product and Chroma's cosine space behaves
consistently.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np

from app.config import Settings


class EmbeddingProvider(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one normalized vector per input text."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic embeddings: hash(text) seeds an RNG -> fixed vector.

    Same text always maps to the same vector, so semantically identical inputs
    collide and search/similar-users are reproducible without a real model.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _seed(self, text: str) -> int:
        digest = hashlib.sha256(text.strip().lower().encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            rng = np.random.default_rng(self._seed(text))
            out[i] = rng.standard_normal(self.dim, dtype=np.float32)
        return _normalize(out).tolist()


class LocalEmbeddingProvider(EmbeddingProvider):
    """sentence-transformers model, loaded once (called from lifespan)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vecs, dtype=np.float32).tolist()


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    mode = settings.embedding_mode
    if mode == "mock":
        return MockEmbeddingProvider(dim=settings.embedding_dim)
    if mode == "local":
        return LocalEmbeddingProvider(model_name=settings.local_model_name)
    raise ValueError(f"Unknown EMBEDDING_MODE: {mode!r}")
