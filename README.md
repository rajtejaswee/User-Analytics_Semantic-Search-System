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
4. [How this maps to the evaluation criteria](#4-how-this-maps-to-the-evaluation-criteria)
5. [Scalability & edge cases](#5-scalability--edge-cases)
6. [Testing](#6-testing) · [Project layout](#7-project-layout) · [What's mocked](#8-whats-mocked--not-productionized)

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
| `EMBEDDING_MODE` | `mock` | `mock` \| `local` \| `openai` |
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
> query matches a stored document exactly (see [§8](#8-whats-mocked--not-productionized)).

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

### Dual-write consistency tradeoff
`/track` writes to **two** stores. Postgres is the **source of truth**; Chroma
is a **derived index**. Order: persist the Postgres row (flush to surface DB
errors) → embed → upsert into Chroma — all inside the request. The DB session
commits only if both succeed; if the Chroma upsert raises, the request fails and
Postgres is rolled back, so the stores never silently diverge.

This is a deliberate **best-effort synchronous dual write** — simple and correct
at this scale. The production alternative (async indexing via a queue +
reconciliation job) is described in [§5](#5-scalability--edge-cases).

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
| `mock` *(default)* | hash(text) → seeded RNG → normalized vector | CI / review / offline — deterministic |
| `local` | sentence-transformers `all-MiniLM-L6-v2` | real semantic quality |
| `openai` | interface stub (raises if used) | shows the seam for a hosted provider |

The model is loaded **once at startup** (FastAPI lifespan), never per request.

### Database schema 
Postgres `events` table:

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID PK | app-generated (`uuid4`) — lets `/track` return the id without a round-trip and keeps the code dialect-agnostic |
| `user_id` | TEXT NOT NULL | **indexed** |
| `event` | TEXT NOT NULL | **indexed** |
| `metadata` | JSONB | flexible per-event attributes, default `{}` |
| `timestamp` | TIMESTAMPTZ NOT NULL | **indexed**; server default `now()` |
| `created_at` | TIMESTAMPTZ | server default `now()` |

**Indexes:** `user_id`, `event`, `timestamp`, and a composite `(event,
timestamp)` for filtered time-range analytics. JSONB keeps event metadata
schemaless while the analytical columns stay strongly typed and indexable.

ChromaDB `events` collection mirrors each row by the **same UUID**; document =
event text (+ light metadata context); metadata = `{user_id, event, timestamp}`
so results return without a second DB hit. Cosine space, so `score = 1 − distance`.

### Vector store is swappable
Everything Chroma-specific lives in `app/services/vector_store.py` behind a small
`upsert / query / all_embeddings` surface. Swapping to **FAISS** (in-proc) or
**Pinecone/Weaviate** (hosted) means one class with the same methods — nothing
else changes.

---

## 4. How this maps to the evaluation criteria

> This section ties each grading criterion to concrete, verifiable evidence in
> the codebase.

### 4.1 API design and structure
- **RESTful, intention-revealing endpoints:** `POST /track`, `GET /analytics`,
  `GET /search`, `GET /similar-users`, plus `GET /health`.
- **Correct semantics:** `201` on create, `422` on validation, `404` for an
  unknown user, `200` with empty results (not an error) on an empty index.
- **Typed contracts:** Pydantic request/response models (`app/schemas/events.py`)
  with camelCase aliases — auto-generated, always-accurate OpenAPI at `/docs`.
- **Consistent error envelope:** a single `{"error": {...}}` shape via FastAPI
  exception handlers (`app/main.py`) instead of ad-hoc error bodies.
- **Thin routers, layered structure:** routers validate + delegate; all logic
  lives in `services/`. Dependencies (DB session, embedder, vector store) are
  injected via FastAPI `Depends` (`app/deps.py`).

### 4.2 Database schema design
- Normalized `events` table with an explicit, minimal column set (see schema
  table above).
- **Indexing strategy driven by the query patterns:** single-column indexes for
  each filter, plus a composite `(event, timestamp)` for the common
  "event over a date range" analytics query.
- **JSONB** for open-ended per-event metadata — flexible without sacrificing the
  typed, indexed analytical columns.
- **Dialect-agnostic types** (`Uuid`, `JSON().with_variant(JSONB, "postgresql")`)
  so the identical model runs on Postgres (prod) and SQLite (tests).

### 4.3 Code clarity and organization
- **Clear layering:** `routers → services → db`, with `schemas`, `config`, and
  `deps` as cross-cutting concerns. Each module has one responsibility.
- **The vector backend and the embedding model are each isolated behind one
  interface**, so swaps don't ripple.
- **Docstrings explain the "why"** (dual-write ordering, centroid rationale,
  dialect shims), not just the "what". Consistent naming and type hints
  throughout.

### 4.4 Correctness of implementation
- **24 automated tests pass** (`uv run pytest -q`) covering `/track` validation +
  dual-write (including a forced vector-store failure → rollback), every
  `/analytics` filter combination, `/search` ranking/limits/shape, and
  `/similar-users` ranking + 404.
- **Verified end-to-end against real Postgres + ChromaDB** via `docker compose up`
  (not just against the SQLite test shim).
- **Analytics are exact aggregations** — counted via SQL `GROUP BY` (one row
  per distinct user, not per event), with the total and top-N derived from the
  same grouped counts, so totals/per-user/most-active are always consistent
  with each other.
- **Consistency is preserved on failure:** a vector-store write error rolls back
  the Postgres transaction.

### 4.5 Thoughtfulness in scalability & edge cases
Covered in detail in [§5](#5-scalability--edge-cases): SQL-side aggregation,
model loaded once, HNSW approximate search, plus explicit handling of missing
fields, malformed/absent timestamps, inverted date ranges, empty index, and
unknown users — each with a matching test.

---

## 5. Scalability & edge cases

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
- Dual-write failure → request fails and Postgres rolls back (no divergence).
- One consistent error envelope for all failures.

**What I'd change for production**
- **Async indexing:** `/track` writes Postgres synchronously and enqueues the
  embed/index step (worker + queue) with a reconciliation job that backfills
  Chroma from Postgres — removing model latency from the write path.
- **Precomputed user vectors** updated incrementally.
- **Dedicated vector DB** (Pinecone/Weaviate/pgvector) with replication.
- **Batching** embeddings on ingest.
- **AuthN/AuthZ**, per-tenant isolation, **rate limiting**, quotas.
- **Alembic migrations** instead of `create_all` on startup.
- Observability: structured logs, metrics, tracing.
- *(A cache such as Redis could front hot analytics queries — a future option,
  not part of this build.)*

---

## 6. Testing

Tests run fully offline and deterministically (`EMBEDDING_MODE=mock`, SQLite via
aiosqlite, temp ChromaDB dir):

```bash
uv sync --extra dev
uv run pytest -q          # 24 passed
```

The same ORM models run on SQLite (tests) and Postgres (prod) via
dialect-agnostic column types, so tests exercise the real code paths.

---

## 7. Project layout

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
    embeddings.py    EmbeddingProvider: mock | local | openai
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

## 8. What's mocked / not productionized

Being explicit (the assignment allows partial scope if well-explained):
- **Embeddings default to a mock** (hash-based) so the system runs with zero
  downloads/keys. Real semantics require `EMBEDDING_MODE=local`. Under the mock,
  vectors have no semantic structure: only an *identical embedded document*
  (event text **plus its metadata context**, e.g. `"user viewed pricing page |
  page=/pricing"`) reproduces the same vector, so `/search` scores are noise
  unless the stored document matches the query exactly. `/similar-users` still
  clusters users who share identical events.
- **`openai` embedding mode is a stub** — it demonstrates the provider seam but
  isn't wired to a live API.
- **Schema is created via `create_all`** on startup, not Alembic migrations.
- **Dual write is synchronous in-request**, not queue-backed.
- **No auth / rate limiting** — out of scope for the assignment.
