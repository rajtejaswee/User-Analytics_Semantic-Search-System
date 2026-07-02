"""Pydantic request/response models.

External JSON uses camelCase (``userId``); internal Python stays snake_case.
``populate_by_name`` + ``alias`` bridge the two.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------- POST /track ----------
class TrackRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_id: str = Field(alias="userId", min_length=1, max_length=128, description="Actor id")
    event: str = Field(
        min_length=1, max_length=5000, description="Free-text event description"
    )
    metadata: dict = Field(default_factory=dict)
    # Optional: server sets now() when omitted.
    timestamp: datetime | None = None

    @field_validator("user_id", "event")
    @classmethod
    def _strip_and_require_content(cls, v: str) -> str:
        # min_length alone would accept "   " and create a whitespace "user".
        v = v.strip()
        if not v:
            raise ValueError("must not be empty or whitespace-only")
        return v

    @field_validator("timestamp")
    @classmethod
    def _naive_means_utc(cls, v: datetime | None) -> datetime | None:
        # Naive timestamps are assumed UTC so stored data is uniformly tz-aware.
        if v is not None and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class TrackResponse(BaseModel):
    id: str
    status: str = "tracked"


# ---------- GET /analytics ----------
class UserCount(BaseModel):
    user_id: str = Field(serialization_alias="userId")
    count: int


class FiltersApplied(BaseModel):
    event: str | None = None
    user_id: str | None = Field(default=None, serialization_alias="userId")
    from_: date | None = Field(default=None, serialization_alias="from")
    to: date | None = None


class AnalyticsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    total_events: int = Field(serialization_alias="totalEvents")
    events_per_user: dict[str, int] = Field(serialization_alias="eventsPerUser")
    most_active_users: list[UserCount] = Field(serialization_alias="mostActiveUsers")
    filters_applied: FiltersApplied = Field(serialization_alias="filtersApplied")


# ---------- GET /search ----------
class SearchResult(BaseModel):
    id: str
    user_id: str = Field(serialization_alias="userId")
    event: str
    timestamp: str | None = None
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


# ---------- GET /similar-users ----------
class SimilarUser(BaseModel):
    user_id: str = Field(serialization_alias="userId")
    score: float


class SimilarUsersResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_id: str = Field(serialization_alias="userId")
    similar_users: list[SimilarUser] = Field(serialization_alias="similarUsers")
