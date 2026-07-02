"""Semantic search + similar-users logic.

Both operate over the ChromaDB index:
  * search        -> embed the query, cosine-nearest events.
  * similar-users -> build a per-user behavior vector as the *centroid* (mean)
                     of that user's event embeddings, then rank other users by
                     cosine similarity of centroids.

The centroid approach is a deliberate, documented approximation: a user is
represented by the average "shape" of their behavior. It's cheap, needs no
training, and degrades gracefully with sparse data. (README discusses
alternatives such as weighting by recency or event type.)
"""
from __future__ import annotations

import numpy as np

from app.services.embeddings import EmbeddingProvider
from app.services.vector_store import VectorStore


def semantic_search(
    *,
    query: str,
    limit: int,
    embedder: EmbeddingProvider,
    store: VectorStore,
) -> list[dict]:
    embedding = embedder.embed_one(query)
    hits = store.query(embedding=embedding, limit=limit)
    results = []
    for hit in hits:
        meta = hit["metadata"]
        results.append(
            {
                "id": hit["id"],
                "user_id": meta.get("user_id", ""),
                "event": meta.get("event", hit.get("document", "")),
                "timestamp": meta.get("timestamp"),
                "score": hit["score"],
            }
        )
    return results


def _user_centroids(store: VectorStore) -> dict[str, np.ndarray]:
    """Mean (L2-normalized) embedding per user across all their events."""
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for item in store.all_embeddings():
        user_id = item["metadata"].get("user_id")
        if not user_id:
            continue
        vec = np.asarray(item["embedding"], dtype=np.float32)
        if user_id in sums:
            sums[user_id] += vec
            counts[user_id] += 1
        else:
            sums[user_id] = vec.copy()
            counts[user_id] = 1

    centroids: dict[str, np.ndarray] = {}
    for user_id, total in sums.items():
        mean = total / counts[user_id]
        norm = np.linalg.norm(mean)
        centroids[user_id] = mean / norm if norm else mean
    return centroids


def similar_users(
    *, user_id: str, limit: int, store: VectorStore
) -> list[dict] | None:
    """Top-K users by centroid cosine similarity. None if user is unknown."""
    centroids = _user_centroids(store)
    if user_id not in centroids:
        return None

    target = centroids[user_id]
    scored = []
    for other_id, vec in centroids.items():
        if other_id == user_id:
            continue
        # centroids are normalized -> dot product is cosine similarity
        score = float(np.dot(target, vec))
        scored.append({"user_id": other_id, "score": round(score, 4)})

    scored.sort(key=lambda r: (-r["score"], r["user_id"]))
    return scored[:limit]
