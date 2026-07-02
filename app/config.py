"""Application settings, loaded from environment / .env via pydantic-settings."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Application
    app_name: str = "Analytics Search Backend"
    log_level: str = "INFO"

    # Database (async SQLAlchemy URL)
    database_url: str = (
        "postgresql+asyncpg://analytics:analytics@localhost:5432/analytics"
    )

    # Embeddings
    embedding_mode: Literal["mock", "local", "openai"] = "mock"
    embedding_dim: int = 384
    local_model_name: str = "all-MiniLM-L6-v2"

    # Vector store
    chroma_dir: str = "./chroma_data"
    chroma_collection: str = "events"

    # Analytics
    default_top_users: int = 10


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton (env is read once)."""
    return Settings()
