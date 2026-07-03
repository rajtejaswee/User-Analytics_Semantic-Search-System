# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Faster, quieter Python; no .pyc clutter.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# uv (pinned) copied from the official distroless image.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

WORKDIR /app

# Install deps first (cached layer) from the manifest + lockfile; --frozen
# guarantees the image gets exactly the locked, tested versions.
# (README is copied because pyproject's metadata references it.)
COPY pyproject.toml uv.lock README.md ./
# Base deps only -> mock embedding mode, small image, no model download.
# For real embeddings, build with:  --build-arg EXTRAS="--extra local"
ARG EXTRAS=""
RUN uv sync --frozen ${EXTRAS} --no-dev --no-install-project

# App source.
COPY app ./app
COPY scripts ./scripts

# Now install the project itself into the environment.
RUN uv sync --frozen ${EXTRAS} --no-dev

EXPOSE 8000
CMD ["uv", "run", "--no-dev", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
