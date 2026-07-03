# Analytics Search Backend

A backend service that ingests user-activity events, exposes **SQL-based
analytics**, and provides **semantic search** and **similar-user** discovery
over those events using vector embeddings.

**Stack:** FastAPI · PostgreSQL (SQLAlchemy 2.0 async / asyncpg) · ChromaDB ·
sentence-transformers (with a deterministic mock fallback) · uv · Docker.

Every event is **dual-written**: Postgres is the source of truth for analytics;
ChromaDB is a derived vector index for the semantic endpoints.

---

## Contents
1. [Setup instructions](#1-setup-instructions)
2. [API documentation](#2-api-documentation) (sample requests / responses)
3. [Design decisions](#3-design-decisions)
4. [Scalability & edge cases](#4-scalability--edge-cases)
5. [Testing & correctness](#5-testing--correctness) 
6. [Project layout](#6-project-layout) 
7.[What's mocked](#7-whats-mocked--not-productionized)

---

## 1. Setup instructions

### Option A — Docker (recommended, one command)

```bash
docker compose up --build
```

This starts **Postgres** and the **API**. Tables are created automatically on
startup. Then, in a second terminal, seed demo data and open the docs:

```bash
python3 scripts/seed.py                 # 52 events across 5 users (stdlib only — no uv needed)
open http://localhost:8000/docs         # interactive Swagger UI
```

> The API runs with `EMBEDDING_MODE=mock` by default — **no model download, no
> API keys, fully deterministic**. Great for a zero-friction review. The mock
> has no semantic structure, so for real semantic search quality build with the
> local model instead (one flag, see below):
>
> ```bash
> docker compose build --build-arg EXTRAS="--extra local"
> EMBEDDING_MODE=local docker compose up
> ```

To stop and wipe volumes: `docker compose down -v`.

### Option B — Local with `uv`

Requires a reachable PostgreSQL. Easiest path: run just the DB from compose and
the app on the host.

```bash
docker compose up -d db                 # Postgres on localhost:5432
uv sync --extra dev                     # install deps (incl. test tools)
cp .env.example .env                    # DATABASE_URL already points at localhost

uv run uvicorn app.main:app --reload    # http://localhost:8000
uv run python scripts/seed.py           # in another terminal
```

### Using real (non-mock) embeddings

The mock provider is deterministic but has **no semantic structure** (it exists
to make setup/CI trivial). For real semantic relevance, switch to the local
sentence-transformers model:

```bash
uv sync --extra local                            # installs sentence-transformers
EMBEDDING_MODE=local uv run uvicorn app.main:app --reload
```

### Configuration (env vars — see `.env.example`)

| Var | Default | Purpose |
|-----|---------|---------|
| `DATABASE_URL` | `postgresql+asyncpg://analytics:analytics@localhost:5432/analytics` | Async DB URL |
| `EMBEDDING_MODE` | `mock` | `mock` \| `local` |
| `EMBEDDING_DIM` | `384` | Mock vector dimension |
| `LOCAL_MODEL_NAME` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `CHROMA_DIR` | `./chroma_data` | ChromaDB persistence path |
| `CHROMA_COLLECTION` | `events` | Chroma collection name |
| `DEFAULT_TOP_USERS` | `10` | `mostActiveUsers` top-N |

---

## 2. API documentation

Interactive, always-in-sync OpenAPI docs: **http://localhost:8000/docs**
(ReDoc at `/redoc`, raw schema at `/openapi.json`).

**Conventions**
- External JSON is **camelCase** (`userId`); internal code is snake_case.
- Every error shares one envelope: `{"error": {"type": ..., "detail": ...}}`.

### `POST /track` — record an event

Validates → inserts into Postgres → embeds the event text → upserts into
ChromaDB → returns `201`.

**Request**
```bash
curl -s -X POST http://localhost:8000/track \
  -H 'Content-Type: application/json' \
  -d '{
        "userId": "u_buyer",
        "event": "user viewed pricing page",
        "metadata": {"page": "/pricing"},
        "timestamp": "2026-03-01T10:00:00Z"
      }'
```

**Response** `201 Created`
```json
{ "id": "3f2a5b7c-...-uuid", "status": "tracked" }
```

| Field | Required | Notes |
|-------|----------|-------|
| `userId` | yes | trimmed; 1–128 chars; missing/whitespace-only/overlong → `422` |
| `event` | yes | free-text, trimmed; 1–5000 chars; missing/whitespace-only/overlong → `422` |
| `metadata` | no | arbitrary JSON object; default `{}` |
| `timestamp` | no | ISO-8601; omitted → server `now()`; naive (no offset) → assumed UTC; malformed → `422` |

**Validation error** `422`
```json
{ "error": { "type": "validation_error", "detail": [ /* field errors */ ] } }
```

### `GET /analytics` — aggregated metrics

Counts are aggregated in SQL (one `GROUP BY user_id` — one row per distinct
user, never one per event); the total and top-N are derived from those grouped
counts. All filters are optional and combinable.

| Query param | Example | Meaning |
|-------------|---------|---------|
| `event` | `?event=user+viewed+pricing+page` | exact event match |
| `userId` | `?userId=u_buyer` | single user |
| `from` | `?from=2026-01-01` | inclusive start date (UTC) |
| `to` | `?to=2026-12-31` | inclusive end date (UTC) |

`from > to` → `422`.

**Request**
```bash
curl -s 'http://localhost:8000/analytics?from=2026-01-01&to=2026-12-31'
```

**Response** `200 OK`
```json
{
  "totalEvents": 52,
  "eventsPerUser": { "u_buyer": 12, "u_shopper": 10, "u_reader": 10, "u_dev": 10, "u_newbie": 10 },
  "mostActiveUsers": [
    { "userId": "u_buyer",  "count": 12 },
    { "userId": "u_dev",    "count": 10 }
  ],
  "filtersApplied": { "event": null, "userId": null, "from": "2026-01-01", "to": "2026-12-31" }
}
```

`mostActiveUsers` = top **N** (default 10) by count desc.

### `GET /search` — semantic search

Embeds the query, runs cosine similarity over ChromaDB, returns ranked events
with scores in `[0, 1]`.

**Request**
```bash
curl -s 'http://localhost:8000/search?query=pricing%20page&limit=5'
```

**Response** `200 OK`
```json
{
  "query": "pricing page",
  "results": [
    { "id": "3f2a5b7c-...", "userId": "u_buyer",
      "event": "user viewed pricing page",
      "timestamp": "2026-03-01T10:00:00+00:00", "score": 0.87 }
  ]
}
```

`limit` defaults to 5 (range 1–50). Empty index → `200` with `"results": []`.

> **Note:** the scores above assume `EMBEDDING_MODE=local`. Under the default
> mock, vectors have no semantic structure, so scores are low/noisy unless the
> query matches a stored document exactly (see [§7](#7-whats-mocked--not-productionized)).

### `GET /similar-users` — behaviourally similar users

Each user is represented by the **centroid (mean) of their event embeddings**;
users are ranked by cosine similarity of centroids.

**Request**
```bash
curl -s 'http://localhost:8000/similar-users?userId=u_buyer&limit=5'
```

**Response** `200 OK`
```json
{
  "userId": "u_buyer",
  "similarUsers": [
    { "userId": "u_shopper", "score": 0.79 },
    { "userId": "u_newbie",  "score": 0.41 }
  ]
}
```

Unknown `userId` → `404`:
```json
{ "error": { "type": "http_error", "detail": "No tracked events for userId 'nope'." } }
```

### `GET /health`
```json
{ "status": "ok" }
```

---

## 3. Design decisions

### Stack rationale
| Layer | Choice | Why |
|-------|--------|-----|
| API | FastAPI (async) | auto OpenAPI docs, Pydantic validation, async I/O |
| DB | PostgreSQL + SQLAlchemy 2.0 (asyncpg) | relational analytics, JSONB metadata, indexing |
| Vectors | ChromaDB (persistent, local) | no external keys, embeddable, cosine search |
| Embeddings | `all-MiniLM-L6-v2` (384-d) + deterministic mock | offline-capable, swappable, reproducible tests |
| Packaging | uv | fast, reproducible installs |

### API design
The API is four intention-revealing endpoints — `POST /track`, `GET /analytics`,
`GET /search`, `GET /similar-users` — plus `GET /health`. I kept a few
conventions consistent across all of them:

- **Status codes carry meaning.** `201` for a created event, `422` for anything
  invalid, `404` only for "this user has no data", and `200` with an empty list
  when a query is valid but simply has no results — an empty index is not an
  error.
- **One error envelope everywhere.** Every failure — validation, HTTP, or
  unexpected — returns the same `{"error": {"type": ..., "detail": ...}}` shape
  (exception handlers in `app/main.py`), so a client parses one format.
- **Typed contracts.** Every request and response is a Pydantic model
  (`app/schemas/events.py`); the OpenAPI docs at `/docs` are generated from
  them, so they can't drift from the code. External JSON is camelCase, internal
  Python is snake_case, bridged by field aliases.
- **Thin routers.** Routers only validate and delegate; the logic lives in
  `services/`, and shared resources (DB session, embedder, vector store) are
  injected via FastAPI `Depends` (`app/deps.py`), which is also what makes the
  test suite able to swap them out.

### Dual-write consistency tradeoff
`/track` writes to **two** stores. Postgres is the **source of truth**; Chroma
is a **derived index**. Order: persist the Postgres row (flush to surface DB
errors) → embed → upsert into Chroma — all inside the request. The DB session
commits only if both succeed; if the Chroma upsert raises, the request fails and
Postgres is rolled back.

One failure window remains open by design: if the Postgres **commit itself**
fails *after* the vector write succeeded, Chroma keeps an orphan vector. I
accepted that asymmetry deliberately — an orphan vector can at worst surface a
stale search hit; it can never corrupt analytics, because those only read
Postgres. The production design (async indexing via a queue + a reconciliation
job that diffs the two stores) closes this window and is described in
[§4](#4-scalability--edge-cases).

### Why a centroid for similar-users
A user's behaviour is summarized as the **mean of their event embeddings** (then
L2-normalized), and users are compared by cosine similarity of centroids:
- **cheap** — one vector per user, no pairwise event comparison;
- **training-free** — works from the first event, nothing to fit;
- **graceful** — degrades sensibly with sparse data.

It's an explicit approximation (ignores event frequency, recency, and sequence).
Documented alternatives: TF-IDF-style weighting of rare events, recency-decayed
centroids, or a learned user embedding (matrix factorization / two-tower model).

### Embedding abstraction
All embedding code sits behind one `EmbeddingProvider` interface, selected by
`EMBEDDING_MODE`, so the model is a swappable detail:

| Mode | Behavior | Use for |
|------|----------|---------|
| `mock` *(default)* | hash(text) → seeded RNG → normalized vector | CI / offline — deterministic |
| `local` | sentence-transformers `all-MiniLM-L6-v2` | real semantic quality |

A hosted provider (OpenAI, Cohere, …) would be one more subclass implementing
the same two-method interface. The model is loaded **once at startup** (FastAPI
lifespan), never per request.

### Database schema
Everything lives in one `events` table — events are immutable facts, so there's
nothing to normalize away, and analytics stay single-table aggregations with no
joins:

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID PK | app-generated (`uuid4`) — lets `/track` return the id without a round-trip and keeps the code dialect-agnostic |
| `user_id` | TEXT NOT NULL | **indexed** |
| `event` | TEXT NOT NULL | **indexed** |
| `metadata` | JSONB | flexible per-event attributes, default `{}` |
| `timestamp` | TIMESTAMPTZ NOT NULL | **indexed**; server default `now()` |
| `created_at` | TIMESTAMPTZ | server default `now()` |

The split between typed columns and JSONB is intentional: everything the API
filters or aggregates on (`user_id`, `event`, `timestamp`) is a strongly typed,
indexed column, while open-ended per-event attributes go into JSONB — schemaless
where flexibility matters, structured where queries run. `timestamp` (when the
event happened, client-suppliable for backfills) is kept separate from
`created_at` (when we recorded it).

**Indexing follows the query patterns:** a single-column index for each filter
(`user_id`, `event`, `timestamp`), plus a composite `(event, timestamp)` for
the most common analytics shape — "one event type over a date range". Column
types are declared dialect-agnostically (`Uuid`,
`JSON().with_variant(JSONB, "postgresql")`), so the identical model runs on
Postgres in docker-compose and on SQLite in the test suite.

ChromaDB's `events` collection mirrors each row by the **same UUID**; document =
event text (+ light metadata context); metadata = `{user_id, event, timestamp}`
so search results return without a second DB hit. Cosine space, so
`score = 1 − distance`.

### Code organization
The codebase is layered `routers → services → db`, with `schemas`, `config`,
and `deps` as cross-cutting modules — each file has one responsibility (see
[project layout](#6-project-layout)). The two most volatile choices — the
embedding model and the vector backend — are each isolated behind a single
interface, so replacing either is a one-class change that doesn't ripple.
Docstrings document the *why* of each non-obvious decision (dual-write
ordering, the centroid approximation, the dialect shims), not just the what.

### Vector store is swappable
Everything Chroma-specific lives in `app/services/vector_store.py` behind a small
`upsert / query / all_embeddings` surface. Swapping to **FAISS** (in-proc) or
**Pinecone/Weaviate** (hosted) means one class with the same methods — nothing
else changes.

---

## 4. Scalability & edge cases

**Scalability**
- **Analytics** counting happens in Postgres (`GROUP BY` on indexed columns),
  so the app only ever handles one row per distinct user — never one per event —
  and the work scales with data volume, not process memory. `mostActiveUsers` is
  a top-N slice. `eventsPerUser` returns one entry per distinct user; for very
  high-cardinality user bases this map should be paginated or dropped in favour
  of the top-N list (a noted production change).
- **Embedding model** loaded once at startup, not per request; the provider
  interface takes a batch (`embed(list[str])`) for bulk ingest.
- **Vector search** uses Chroma's **HNSW cosine index** (approximate NN), which
  scales sub-linearly.
- **Similar-users** recomputes centroids from all vectors per call — fine for
  thousands of events; at scale, precompute per-user centroids and update them
  incrementally.

**Edge cases handled (each with a test)**
- Missing `userId` / `event` → `422`; malformed `timestamp` → `422`.
- Whitespace-only `userId` / `event` → `422` (values are trimmed, so `"  "`
  can't become a phantom user).
- Length caps (`userId` ≤ 128, `event` ≤ 5000 chars) → `422` on abusive payloads.
- Omitted `timestamp` → server `now()`; naive timestamp (no offset) → assumed
  UTC, so stored data is uniformly tz-aware.
- `from > to` → `422`.
- `/search` on an empty index → `200` with `[]` (no crash).
- `/similar-users` for an unknown user → `404`.
- Vector-store write failure → request fails and Postgres rolls back (tested
  by forcing the failure; the reverse commit-failure window is documented in
  [§3](#3-design-decisions)).
- One consistent error envelope for all failures.

**What I'd change for production**
- **Async indexing:** `/track` writes Postgres synchronously and enqueues the
  embed/index step (worker + queue) with a reconciliation job that backfills
  Chroma from Postgres — removing model latency from the write path.
- **Move blocking work off the event loop:** the embedding and vector-store
  calls are synchronous inside async handlers. Negligible with the mock
  (microseconds), but the local model would block the loop under load — the
  queue above fixes it structurally; a threadpool (`anyio.to_thread`) is the
  lighter interim fix.
- **Precomputed user vectors** updated incrementally.
- **Dedicated vector DB** (Pinecone/Weaviate/pgvector) with replication.
- **Batching** embeddings on ingest.
- **AuthN/AuthZ**, per-tenant isolation, **rate limiting**, quotas.
- **Alembic migrations** instead of `create_all` on startup.
- Observability: structured logs, metrics, tracing.
- *(A cache such as Redis could front hot analytics queries — a future option,
  not part of this build.)*

---

## 5. Testing & correctness

Tests run fully offline and deterministically (`EMBEDDING_MODE=mock`, SQLite via
aiosqlite, temp ChromaDB dir):

```bash
uv sync --extra dev
uv run pytest -q          # 24 passed
```

The 24 tests cover `/track` validation and the dual write — including a forced
vector-store failure that asserts the `500` response *and* the Postgres
rollback — every `/analytics` filter combination, `/search`
ranking/limit/response shape, and `/similar-users` ranking + `404`.

The same ORM models run on SQLite (tests) and Postgres (prod) via
dialect-agnostic column types, so tests exercise the real code paths. Beyond
the unit suite, the whole stack was verified end-to-end against real Postgres +
ChromaDB via `docker compose up`: seeded through the live API and exercised
through every endpoint, not just against the SQLite test shim.

---

## 6. Project layout

```
app/
  main.py            FastAPI app, lifespan, error handlers, router registration
  config.py          pydantic-settings (env-driven)
  deps.py            shared dependencies (session, embedder, vector store)
  db/
    models.py        SQLAlchemy Event model (dialect-agnostic types)
    session.py       async engine + session factory
    init_db.py       create tables on startup
  schemas/events.py  Pydantic request/response models (camelCase aliases)
  services/
    embeddings.py    EmbeddingProvider: mock | local
    vector_store.py  ChromaDB wrapper (swappable)
    analytics.py     SQL GROUP BY aggregations
    search.py        semantic search + user-centroid similarity
  routers/           track, analytics, search, similar_users
tests/               conftest + endpoint tests (24)
scripts/seed.py      52 demo events across 5 users (via the live API)
docker-compose.yml   api + postgres (healthcheck, auto table creation)
Dockerfile           uv-based image (Python 3.12)
```

---

## 7. What's mocked / not productionized

Being upfront about the shortcuts I took deliberately, and what the production
version would do instead:
- **Embeddings default to a mock** (hash-based) so the system runs with zero
  downloads/keys. Real semantics require `EMBEDDING_MODE=local`. Under the mock,
  vectors have no semantic structure: only an *identical embedded document*
  (event text **plus its metadata context**, e.g. `"user viewed pricing page |
  page=/pricing"`) reproduces the same vector, so `/search` scores are noise
  unless the stored document matches the query exactly. `/similar-users` still
  clusters users who share identical events.
- **Schema is created via `create_all`** on startup, not Alembic migrations.
- **Dual write is synchronous in-request**, not queue-backed.
- **No auth / rate limiting** — deliberately omitted here; in production I'd
  add API-key or JWT auth and per-client rate limits before exposing `/track`
  publicly.
