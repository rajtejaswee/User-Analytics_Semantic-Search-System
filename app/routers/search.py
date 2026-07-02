"""GET /search — semantic (vector) search over tracked events."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.deps import get_embedder, get_vector_store
from app.schemas.events import SearchResponse, SearchResult
from app.services.embeddings import EmbeddingProvider
from app.services.search import semantic_search
from app.services.vector_store import VectorStore

router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResponse)
async def search(
    query: str = Query(min_length=1, description="Natural-language search query"),
    limit: int = Query(default=5, ge=1, le=50),
    embedder: EmbeddingProvider = Depends(get_embedder),
    store: VectorStore = Depends(get_vector_store),
) -> SearchResponse:
    hits = semantic_search(
        query=query, limit=limit, embedder=embedder, store=store
    )
    return SearchResponse(
        query=query, results=[SearchResult(**h) for h in hits]
    )
