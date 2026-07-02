"""Tests for POST /track (validation + dual-write)."""
from __future__ import annotations

from tests.conftest import seed


def test_track_returns_201_and_id(client):
    resp = client.post(
        "/track",
        json={"userId": "u1", "event": "user viewed pricing page", "metadata": {"page": "/pricing"}},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "tracked"
    assert body["id"]  # non-empty uuid string


def test_track_missing_user_id_is_422(client):
    resp = client.post("/track", json={"event": "no user"})
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "validation_error"


def test_track_missing_event_is_422(client):
    resp = client.post("/track", json={"userId": "u1"})
    assert resp.status_code == 422


def test_track_malformed_timestamp_is_422(client):
    resp = client.post(
        "/track",
        json={"userId": "u1", "event": "e", "timestamp": "not-a-date"},
    )
    assert resp.status_code == 422


def test_track_missing_timestamp_defaults_to_now(client):
    resp = client.post("/track", json={"userId": "u1", "event": "server sets time"})
    assert resp.status_code == 201
    # Event is immediately visible in analytics (row persisted).
    a = client.get("/analytics").json()
    assert a["totalEvents"] == 1


def test_track_dual_writes_to_vector_store(client, sample_events):
    seed(client, sample_events)
    # If Chroma received the writes, search returns hits.
    resp = client.get("/search", params={"query": "pricing", "limit": 3})
    assert resp.status_code == 200
    assert len(resp.json()["results"]) > 0


def test_track_whitespace_only_fields_are_422(client):
    resp = client.post("/track", json={"userId": "   ", "event": "e"})
    assert resp.status_code == 422
    resp = client.post("/track", json={"userId": "u1", "event": "   "})
    assert resp.status_code == 422


def test_track_overlong_fields_are_422(client):
    resp = client.post("/track", json={"userId": "u" * 129, "event": "e"})
    assert resp.status_code == 422
    resp = client.post("/track", json={"userId": "u1", "event": "x" * 5001})
    assert resp.status_code == 422


def test_track_naive_timestamp_assumed_utc(client):
    resp = client.post(
        "/track",
        json={"userId": "u1", "event": "naive time", "timestamp": "2026-03-01T10:00:00"},
    )
    assert resp.status_code == 201
    # Round-trips as tz-aware UTC (mock embeddings: exact text matches itself).
    hit = client.get("/search", params={"query": "naive time", "limit": 1}).json()["results"][0]
    assert hit["timestamp"] == "2026-03-01T10:00:00+00:00"


def test_track_vector_store_failure_rolls_back_postgres(client):
    store = client.app.state.vector_store
    original_upsert = store.upsert

    def boom(**kwargs):
        raise RuntimeError("vector store down")

    store.upsert = boom
    try:
        resp = client.post("/track", json={"userId": "u1", "event": "must not persist"})
    finally:
        store.upsert = original_upsert

    assert resp.status_code == 500
    assert resp.json()["error"]["type"] == "internal_error"
    # The Postgres row was rolled back: the two stores did not diverge.
    assert client.get("/analytics").json()["totalEvents"] == 0
