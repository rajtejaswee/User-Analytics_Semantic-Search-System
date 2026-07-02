"""GET /analytics — SQL aggregations with optional, combinable filters."""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.deps import get_session
from app.schemas.events import (
    AnalyticsResponse,
    FiltersApplied,
    UserCount,
)
from app.services.analytics import compute_analytics

router = APIRouter(tags=["analytics"])


@router.get("/analytics", response_model=AnalyticsResponse)
async def analytics(
    event: str | None = Query(default=None),
    userId: str | None = Query(default=None),
    from_: date | None = Query(default=None, alias="from"),
    to: date | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
) -> AnalyticsResponse:
    if from_ and to and from_ > to:
        raise HTTPException(
            status_code=422,
            detail="'from' must be on or before 'to'.",
        )

    result = await compute_analytics(
        session,
        event=event,
        user_id=userId,
        date_from=from_,
        date_to=to,
        top_n=get_settings().default_top_users,
    )

    return AnalyticsResponse(
        total_events=result["total_events"],
        events_per_user=result["events_per_user"],
        most_active_users=[UserCount(**u) for u in result["most_active_users"]],
        filters_applied=FiltersApplied(
            event=event, user_id=userId, from_=from_, to=to
        ),
    )
