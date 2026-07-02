"""FastAPI application: lifespan wiring, error envelope, router registration."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_settings
from app.db.init_db import init_db
from app.db.session import dispose_engine
from app.routers import analytics, search, similar_users, track
from app.services.embeddings import build_embedding_provider
from app.services.vector_store import VectorStore

logger = logging.getLogger("analytics")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    # Create the schema on startup (see db/init_db.py for the migration tradeoff).
    await init_db()

    # Heavy objects created ONCE and shared across requests.
    app.state.embedder = build_embedding_provider(settings)
    app.state.vector_store = VectorStore(settings)
    logger.info(
        "Startup complete: embedding_mode=%s dim=%s",
        settings.embedding_mode,
        app.state.embedder.dim,
    )

    yield

    await dispose_engine()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        summary="User analytics + semantic search backend.",
        lifespan=lifespan,
    )

    # ---- Consistent error envelope ----
    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        # jsonable_encoder: validator errors carry a raw exception in `ctx`,
        # which JSONResponse alone can't serialize.
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "type": "validation_error",
                    "detail": jsonable_encoder(exc.errors()),
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"type": "http_error", "detail": exc.detail}},
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": {"type": "internal_error", "detail": "Internal server error."}},
        )

    @app.get("/health", tags=["meta"])
    async def health():
        return {"status": "ok"}

    app.include_router(track.router)
    app.include_router(analytics.router)
    app.include_router(search.router)
    app.include_router(similar_users.router)
    return app


app = create_app()
