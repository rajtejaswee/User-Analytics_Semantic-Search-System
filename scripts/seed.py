"""Seed 52 varied events across 5 users via the live API.

Run against a running server (local or docker-compose):

    uv run python scripts/seed.py                 # -> http://localhost:8000
    BASE_URL=http://localhost:8000 uv run python scripts/seed.py

Uses only the standard library so it needs no extra deps. Each user has a
distinct behavior profile so /search and /similar-users return meaningful
results immediately (e.g. u_buyer and u_shopper cluster together; u_dev stands
apart).
"""
from __future__ import annotations

import json
import os
import random
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")

# user_id -> list of (event text, metadata) templates describing their "theme".
PROFILES: dict[str, list[tuple[str, dict]]] = {
    "u_buyer": [
        ("user viewed pricing page", {"page": "/pricing"}),
        ("user viewed enterprise pricing", {"page": "/pricing/enterprise"}),
        ("user compared subscription plans", {"page": "/pricing"}),
        ("user added plan to cart", {"plan": "pro"}),
        ("user started checkout", {"page": "/checkout"}),
        ("user completed purchase", {"amount": 49}),
    ],
    "u_shopper": [
        ("user viewed pricing page", {"page": "/pricing"}),
        ("user compared subscription plans", {"page": "/pricing"}),
        ("user viewed product features", {"page": "/features"}),
        ("user added plan to cart", {"plan": "basic"}),
        ("user abandoned checkout", {"page": "/checkout"}),
    ],
    "u_reader": [
        ("user read blog post about analytics", {"slug": "intro-analytics"}),
        ("user read documentation on search", {"page": "/docs/search"}),
        ("user viewed tutorial video", {"video": "getting-started"}),
        ("user subscribed to newsletter", {}),
        ("user read case study", {"slug": "customer-x"}),
    ],
    "u_dev": [
        ("user opened api documentation", {"page": "/docs/api"}),
        ("user generated an api key", {}),
        ("user made first api request", {"endpoint": "/track"}),
        ("user viewed webhook settings", {"page": "/settings/webhooks"}),
        ("user read rate limit docs", {"page": "/docs/limits"}),
    ],
    "u_newbie": [
        ("user signed up for account", {"cta": "hero"}),
        ("user completed onboarding step", {"step": 1}),
        ("user completed onboarding step", {"step": 2}),
        ("user invited a teammate", {}),
        ("user viewed dashboard", {"page": "/dashboard"}),
    ],
}


def post_track(payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/track",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def main() -> None:
    random.seed(42)
    now = datetime.now(timezone.utc)
    count = 0
    # Two passes over the profiles -> 52 events with natural repetition per user.
    for _pass in range(2):
        for user_id, templates in PROFILES.items():
            for event, metadata in templates:
                ts = now - timedelta(
                    days=random.randint(0, 30), minutes=random.randint(0, 1440)
                )
                payload = {
                    "userId": user_id,
                    "event": event,
                    "metadata": metadata,
                    "timestamp": ts.isoformat(),
                }
                try:
                    post_track(payload)
                    count += 1
                except urllib.error.URLError as exc:  # pragma: no cover
                    raise SystemExit(
                        f"Could not reach {BASE_URL}. Is the server running? ({exc})"
                    )

    print(f"Seeded {count} events across {len(PROFILES)} users at {BASE_URL}.")
    print("Try:  curl 'http://localhost:8000/analytics'")
    print("      curl 'http://localhost:8000/search?query=pricing&limit=5'")
    print("      curl 'http://localhost:8000/similar-users?userId=u_buyer'")


if __name__ == "__main__":
    main()
