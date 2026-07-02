"""Analytics aggregations — computed in SQL, never by loading rows into Python.

All counts come from ``GROUP BY`` queries against indexed columns so the work
stays in Postgres and scales with the data, not the process memory.
"""
from __future__ import annotations

from datetime import date, datetime, time, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Event


def _apply_filters(stmt, *, event, user_id, date_from, date_to):
    if event is not None:
        stmt = stmt.where(Event.event == event)
    if user_id is not None:
        stmt = stmt.where(Event.user_id == user_id)
    if date_from is not None:
        start = datetime.combine(date_from, time.min, tzinfo=timezone.utc)
        stmt = stmt.where(Event.timestamp >= start)
    if date_to is not None:
        # inclusive `to`: everything strictly before the next day's midnight
        end = datetime.combine(date_to, time.max, tzinfo=timezone.utc)
        stmt = stmt.where(Event.timestamp <= end)
    return stmt


async def compute_analytics(
    session: AsyncSession,
    *,
    event: str | None = None,
    user_id: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    top_n: int = 10,
) -> dict:
    filters = dict(event=event, user_id=user_id, date_from=date_from, date_to=date_to)

    # Per-user counts (one GROUP BY covers total, per-user, and most-active).
    per_user_stmt = _apply_filters(
        select(Event.user_id, func.count().label("cnt")), **filters
    ).group_by(Event.user_id)
    rows = (await session.execute(per_user_stmt)).all()

    events_per_user = {user: cnt for user, cnt in rows}
    total_events = sum(events_per_user.values())
    most_active = sorted(
        ({"user_id": u, "count": c} for u, c in events_per_user.items()),
        key=lambda r: (-r["count"], r["user_id"]),
    )[:top_n]

    return {
        "total_events": total_events,
        "events_per_user": events_per_user,
        "most_active_users": most_active,
    }
