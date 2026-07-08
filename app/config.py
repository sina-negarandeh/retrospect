from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application identity
    app_name: str = Field(default="Retrospect", description="Human-readable application name.")
    app_version: str = Field(default="0.1.0", description="Semantic version of the application.")

    # Ollama
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the Ollama inference server.",
    )
    ollama_model: str = Field(
        default="gemma4:26b-mlx",
        description="Default Ollama model tag to use for chat completions.",
    )
    ollama_embedding_model: str = Field(
        default="embeddinggemma:300m",
        description="Ollama model tag to use for embeddings.",
    )

    # Qdrant
    qdrant_url: str = Field(
        default="http://qdrant:6333",
        description="URL for the Qdrant vector database.",
    )
    qdrant_collection_name: str = Field(
        default="retrospect_docs",
        description="Name of the Qdrant collection.",
    )

    # Document Processing
    chunk_size: int = Field(
        default=512,
        description="Maximum tokens per document chunk.",
    )
    chunk_overlap: int = Field(
        default=64,
        description="Token overlap between adjacent chunks.",
    )
    vector_store_batch_size: int = Field(
        default=20,
        description="Batch size for embedding documents and upserting points.",
    )
    qdrant_scroll_batch_size: int = Field(
        default=100,
        description="Batch size for checking existing document IDs via Qdrant scroll.",
    )
    hf_token: str | None = Field(
        default=None,
        description="Hugging Face API token for tokenizer.",
    )
    tokenizer_model: str = Field(
        default="google/gemma-4-26B-A4B-it",
        description="Hugging Face model ID for the tokenizer.",
    )

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://mlflow:5000",
        description="URI of the MLflow tracking server.",
    )
    mlflow_experiment_name: str = Field(
        default="retrospect",
        description="MLflow experiment name under which runs are logged.",
    )

    # FastAPI
    api_prefix: str = Field(
        default="/api/v1",
        description="URL prefix for all v1 API routes.",
    )

    # Observability
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Python logging level for the application.",
    )
    debug: bool = Field(
        default=False,
        description="Enable FastAPI debug mode.",
    )

    # Security
    cors_allowed_origins: list[str] = Field(
        default=["*"],
        description=(
            "List of origins allowed to make cross-origin requests. "
            "Defaults to ['*'] for development."
        ),
    )
    admin_token: str = Field(
        default="",
        description="Bearer token required for admin endpoints.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
