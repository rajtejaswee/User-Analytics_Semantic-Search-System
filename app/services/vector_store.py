"""ChromaDB wrapper — the one place that knows about the vector backend.

Kept deliberately small (upsert / query / fetch-by-user) so the store is
swappable: a Pinecone or FAISS implementation would expose the same methods
and nothing else in the app would change.

Chroma is configured with cosine space, so a returned ``distance`` maps to a
similarity ``score = 1 - distance`` (clamped to [0, 1]).
"""
from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import Settings


def _score_from_distance(distance: float) -> float:
    return round(max(0.0, min(1.0, 1.0 - distance)), 4)


class VectorStore:
    def __init__(self, settings: Settings):
        self._client = chromadb.PersistentClient(
            path=settings.chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        *,
        id: str,
        embedding: list[float],
        document: str,
        metadata: dict[str, Any],
    ) -> None:
        self._collection.upsert(
            ids=[id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata],
        )

    def query(
        self, *, embedding: list[float], limit: int
    ) -> list[dict[str, Any]]:
        """Nearest events to a query embedding, ranked by cosine similarity."""
        if self.count() == 0:
            return []
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            include=["metadatas", "distances", "documents"],
        )
        ids = res["ids"][0]
        metadatas = res["metadatas"][0]
        distances = res["distances"][0]
        documents = res["documents"][0]
        results: list[dict[str, Any]] = []
        for _id, meta, dist, doc in zip(ids, metadatas, distances, documents):
            results.append(
                {
                    "id": _id,
                    "metadata": meta or {},
                    "document": doc,
                    "score": _score_from_distance(dist),
                }
            )
        return results

    def all_embeddings(self) -> list[dict[str, Any]]:
        """Every stored vector with its metadata (for user-centroid math)."""
        res = self._collection.get(include=["embeddings", "metadatas"])
        out: list[dict[str, Any]] = []
        for _id, emb, meta in zip(
            res["ids"], res["embeddings"], res["metadatas"]
        ):
            out.append({"id": _id, "embedding": emb, "metadata": meta or {}})
        return out

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Drop all vectors (used by the test suite between runs)."""
        self._client.reset()
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
