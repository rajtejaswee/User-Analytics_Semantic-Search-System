"""Shared FastAPI dependencies.

The embedding provider and vector store are created once during lifespan and
stashed on ``app.state``; these accessors pull them off the current request's
app so routers stay decoupled from startup wiring.
"""
from __future__ import annotations

from fastapi import Request

from app.services.embeddings import EmbeddingProvider
from app.services.vector_store import VectorStore

# Re-export the DB session dependency for a single import site in routers.
from app.db.session import get_session  # noqa: F401


def get_embedder(request: Request) -> EmbeddingProvider:
    return request.app.state.embedder


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store
