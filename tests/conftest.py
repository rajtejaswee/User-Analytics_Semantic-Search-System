"""Test fixtures.

Tests run fully offline and deterministically:
  * EMBEDDING_MODE=mock  -> hash-based vectors, no model download.
  * SQLite (aiosqlite)   -> the async ORM works unchanged (see models.py variants).
  * A temp ChromaDB dir  -> isolated from any real ./chroma_data.

Each test gets a *fresh* SQLite file and ChromaDB dir. An async engine is bound
to the event loop that created it, so we reset the engine/settings singletons
per test and let the app's lifespan (run by TestClient) build a new engine in
its own loop — then dispose it on shutdown. This avoids cross-loop reuse.
"""
from __future__ import annotations

import os
import tempfile
import uuid

import pytest

# Base env set before app import so the cached Settings pick it up.
os.environ.setdefault("EMBEDDING_MODE", "mock")
os.environ.setdefault("EMBEDDING_DIM", "384")

from fastapi.testclient import TestClient  # noqa: E402

import app.db.session as db_session  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.main import create_app  # noqa: E402


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="analytics-test-")
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp}/test-{uuid.uuid4().hex}.db"
    os.environ["CHROMA_DIR"] = f"{tmp}/chroma"

    # Force fresh singletons for this test's event loop.
    get_settings.cache_clear()
    db_session._engine = None
    db_session._sessionmaker = None

    app = create_app()
    # raise_server_exceptions=False: unhandled errors come back as the app's
    # 500 envelope (like production) instead of re-raising into the test.
    with TestClient(app, raise_server_exceptions=False) as c:  # runs lifespan
        yield c


@pytest.fixture
def sample_events():
    """A small, varied set covering pricing / checkout / signup themes."""
    return [
        {"userId": "u1", "event": "user viewed pricing page", "metadata": {"page": "/pricing"}},
        {"userId": "u1", "event": "user viewed enterprise pricing", "metadata": {"page": "/pricing/enterprise"}},
        {"userId": "u1", "event": "user started checkout", "metadata": {"page": "/checkout"}},
        {"userId": "u2", "event": "user viewed pricing page", "metadata": {"page": "/pricing"}},
        {"userId": "u2", "event": "user compared plans", "metadata": {"page": "/pricing"}},
        {"userId": "u3", "event": "user clicked signup button", "metadata": {"cta": "hero"}},
        {"userId": "u3", "event": "user completed signup", "metadata": {}},
    ]


def seed(client: TestClient, events: list[dict]) -> list[str]:
    """POST each event through the API; return created ids."""
    ids = []
    for ev in events:
        resp = client.post("/track", json=ev)
        assert resp.status_code == 201, resp.text
        ids.append(resp.json()["id"])
    return ids
