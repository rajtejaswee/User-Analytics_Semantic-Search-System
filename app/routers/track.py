"""POST /track — record an event (dual-write: Postgres + Chroma)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Event
from app.deps import get_embedder, get_session, get_vector_store
from app.schemas.events import TrackRequest, TrackResponse
from app.services.embeddings import EmbeddingProvider
from app.services.vector_store import VectorStore

router = APIRouter(tags=["track"])


def _embed_text(event: str, metadata: dict) -> str:
    """Text handed to the embedder: event plus light metadata context."""
    if not metadata:
        return event
    context = " ".join(f"{k}={v}" for k, v in metadata.items())
    return f"{event} | {context}"


@router.post("/track", response_model=TrackResponse, status_code=status.HTTP_201_CREATED)
async def track(
    payload: TrackRequest,
    session: AsyncSession = Depends(get_session),
    embedder: EmbeddingProvider = Depends(get_embedder),
    store: VectorStore = Depends(get_vector_store),
) -> TrackResponse:
    ts = payload.timestamp or datetime.now(timezone.utc)
    event_id = uuid.uuid4()

    # 1) Source of truth: Postgres.
    row = Event(
        id=event_id,
        user_id=payload.user_id,
        event=payload.event,
        event_metadata=payload.metadata,
        timestamp=ts,
    )
    session.add(row)
    await session.flush()  # surface DB errors before we touch the index

    # 2) Derived index: Chroma. Done in-request (best-effort dual write). If this
    #    raises, get_session rolls back Postgres so the two stores don't diverge.
    #    Production would decouple this via an async queue (see README).
    document = _embed_text(payload.event, payload.metadata)
    embedding = embedder.embed_one(document)
    store.upsert(
        id=str(event_id),
        embedding=embedding,
        document=document,
        metadata={
            "user_id": payload.user_id,
            "event": payload.event,
            "timestamp": ts.isoformat(),
        },
    )

    return TrackResponse(id=str(event_id), status="tracked")
