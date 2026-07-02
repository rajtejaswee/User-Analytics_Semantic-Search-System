"""GET /similar-users — rank users by behavior-centroid similarity."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.deps import get_vector_store
from app.schemas.events import SimilarUser, SimilarUsersResponse
from app.services.search import similar_users as compute_similar_users
from app.services.vector_store import VectorStore

router = APIRouter(tags=["similar-users"])


@router.get("/similar-users", response_model=SimilarUsersResponse)
async def similar_users(
    userId: str = Query(min_length=1),
    limit: int = Query(default=5, ge=1, le=50),
    store: VectorStore = Depends(get_vector_store),
) -> SimilarUsersResponse:
    results = compute_similar_users(user_id=userId, limit=limit, store=store)
    if results is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No tracked events for userId {userId!r}.",
        )
    return SimilarUsersResponse(
        user_id=userId,
        similar_users=[SimilarUser(**r) for r in results],
    )
