"""SQLAlchemy ORM models.

Postgres is the source of truth for analytics. ChromaDB is a derived index
(see services/vector_store.py) that mirrors these rows for semantic ops.

Column types are declared dialect-agnostically so the same model runs on
Postgres (production / docker-compose) and SQLite (the test suite):
  * ``Uuid``  -> native UUID on Postgres, CHAR(32) elsewhere.
  * ``JSON().with_variant(JSONB, "postgresql")`` -> JSONB on Postgres, JSON on SQLite.

UUIDs are generated application-side (``default=uuid.uuid4``) rather than via
``gen_random_uuid()``, so ``/track`` can return the id without a DB round-trip
and the same code path works on every dialect.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Index, String, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# JSONB on Postgres, plain JSON on SQLite (tests).
JsonType = JSON().with_variant(JSONB(), "postgresql")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    event: Mapped[str] = mapped_column(String, nullable=False, index=True)
    # Mapped attribute is `event_metadata`; DB column is `metadata`
    # (`metadata` is reserved on the Declarative class).
    event_metadata: Mapped[dict] = mapped_column(
        "metadata", JsonType, nullable=False, default=dict
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        # Composite index for filtered analytics (event + time-range scans).
        Index("ix_events_event_timestamp", "event", "timestamp"),
    )
