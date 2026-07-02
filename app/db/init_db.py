"""Schema bootstrap.

Tables are created directly from the SQLAlchemy metadata on startup
(``create_all``) instead of via Alembic migrations, which keeps setup to a
single ``docker compose up``. A production system would use versioned
migrations (noted in the README).
"""
from __future__ import annotations

from app.db.models import Base
from app.db.session import get_engine


async def init_db() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
