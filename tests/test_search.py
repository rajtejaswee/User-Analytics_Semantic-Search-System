"""Tests for GET /search and GET /similar-users (vector-backed)."""
from __future__ import annotations

from tests.conftest import seed


def test_search_empty_index_returns_no_results(client):
    body = client.get("/search", params={"query": "pricing"}).json()
    assert body["query"] == "pricing"
    assert body["results"] == []


def test_search_returns_ranked_results(client, sample_events):
    seed(client, sample_events)
    body = client.get("/search", params={"query": "user viewed pricing page", "limit": 5}).json()
    assert len(body["results"]) > 0
    # Scores are sorted descending.
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_search_exact_match_tops_ranking(client, sample_events):
    # "user completed signup" is stored with empty metadata, so its embedded
    # document equals the event text. Querying that exact text must score ~1.0
    # and rank first (mock vectors only "match" identical strings).
    seed(client, sample_events)
    body = client.get(
        "/search", params={"query": "user completed signup", "limit": 5}
    ).json()
    assert body["results"][0]["event"] == "user completed signup"
    assert body["results"][0]["score"] >= 0.99


def test_search_respects_limit(client, sample_events):
    seed(client, sample_events)
    body = client.get("/search", params={"query": "pricing", "limit": 2}).json()
    assert len(body["results"]) <= 2


def test_search_result_shape(client, sample_events):
    seed(client, sample_events)
    r = client.get("/search", params={"query": "signup", "limit": 1}).json()["results"][0]
    assert set(r.keys()) == {"id", "userId", "event", "timestamp", "score"}


def test_similar_users_returns_ranked_users(client, sample_events):
    seed(client, sample_events)
    body = client.get("/similar-users", params={"userId": "u1", "limit": 5}).json()
    assert body["userId"] == "u1"
    others = {u["userId"] for u in body["similarUsers"]}
    assert "u1" not in others  # never compares to self
    assert others <= {"u2", "u3"}
    scores = [u["score"] for u in body["similarUsers"]]
    assert scores == sorted(scores, reverse=True)


def test_similar_users_unknown_user_is_404(client, sample_events):
    seed(client, sample_events)
    resp = client.get("/similar-users", params={"userId": "nope"})
    assert resp.status_code == 404
