from __future__ import annotations

import logging
import logging.config
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import v1_router
from app.config import get_settings


def _configure_logging(log_level: str) -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%dT%H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": log_level,
                "handlers": ["console"],
            },
        }
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    # Logging
    _configure_logging(settings.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    # MLflow
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        logger.info(
            "MLflow configured: uri=%s experiment=%s",
            settings.mlflow_tracking_uri,
            settings.mlflow_experiment_name,
        )
    except Exception as exc:
        logger.warning("Failed to configure MLflow tracking server during startup: %s", exc)

    # Warmup
    try:
        from app.graph.nodes import (
            _get_chat_llm,
            _get_reranker,
            _get_rewrite_llm,
            _get_vector_store,
        )

        logger.info("Warming up singletons...")
        await _get_vector_store()
        await _get_reranker()
        _get_rewrite_llm()
        _get_chat_llm()
        logger.info("Singletons warmed up successfully.")
    except Exception:
        logger.exception("Failed to warm up singletons during startup.")

    # RAG Ingestion
    # Auto-ingestion is DISABLED. Trigger it explicitly via the admin API:
    #   POST /api/v1/admin/ingest        
    #   POST /api/v1/admin/ingest?limit=5
    #   DELETE /api/v1/admin/vector-store
    logger.info("Ingestion not started automatically. Use POST /api/v1/admin/ingest to trigger it.")

    yield

    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """Creates and fully configures the FastAPI app."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Retrospect — an LLM-powered conversation service built with "
            "LangGraph, Ollama, and MLflow."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
        debug=settings.debug,
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(v1_router, prefix=settings.api_prefix)

    return app


app = create_app()
