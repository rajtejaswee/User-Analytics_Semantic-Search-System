"""Tests for GET /analytics (SQL aggregation + filter combinations)."""
from __future__ import annotations

from tests.conftest import seed


def test_analytics_empty(client):
    body = client.get("/analytics").json()
    assert body["totalEvents"] == 0
    assert body["eventsPerUser"] == {}
    assert body["mostActiveUsers"] == []


def test_analytics_totals_and_per_user(client, sample_events):
    seed(client, sample_events)
    body = client.get("/analytics").json()
    assert body["totalEvents"] == 7
    assert body["eventsPerUser"] == {"u1": 3, "u2": 2, "u3": 2}
    # Most active is u1 (3), ordered by count desc.
    assert body["mostActiveUsers"][0] == {"userId": "u1", "count": 3}


def test_analytics_filter_by_user(client, sample_events):
    seed(client, sample_events)
    body = client.get("/analytics", params={"userId": "u1"}).json()
    assert body["totalEvents"] == 3
    assert body["eventsPerUser"] == {"u1": 3}
    assert body["filtersApplied"]["userId"] == "u1"


def test_analytics_filter_by_event(client, sample_events):
    seed(client, sample_events)
    body = client.get("/analytics", params={"event": "user viewed pricing page"}).json()
    assert body["totalEvents"] == 2  # u1 + u2
    assert body["eventsPerUser"] == {"u1": 1, "u2": 1}


def test_analytics_date_range_filter(client):
    client.post("/track", json={"userId": "u1", "event": "old", "timestamp": "2026-01-01T10:00:00Z"})
    client.post("/track", json={"userId": "u1", "event": "mid", "timestamp": "2026-03-15T10:00:00Z"})
    client.post("/track", json={"userId": "u1", "event": "new", "timestamp": "2026-06-01T10:00:00Z"})

    body = client.get("/analytics", params={"from": "2026-02-01", "to": "2026-04-01"}).json()
    assert body["totalEvents"] == 1  # only "mid"


def test_analytics_from_after_to_is_422(client):
    resp = client.get("/analytics", params={"from": "2026-05-01", "to": "2026-01-01"})
    assert resp.status_code == 422


def test_analytics_combined_filters(client, sample_events):
    seed(client, sample_events)
    body = client.get(
        "/analytics",
        params={"userId": "u1", "event": "user started checkout"},
    ).json()
    assert body["totalEvents"] == 1
    assert body["filtersApplied"]["userId"] == "u1"
    assert body["filtersApplied"]["event"] == "user started checkout"
