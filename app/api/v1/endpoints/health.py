from __future__ import annotations

import asyncio
import logging

import httpx
from fastapi import APIRouter

from app.config import get_settings
from app.domain.models import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()

_PROBE_TIMEOUT = 3.0  # seconds


async def _probe(client: httpx.AsyncClient, url: str, name: str) -> bool:
    """Probe a single URL; return True if HTTP 2xx, False on any failure."""
    try:
        resp = await client.get(url)
        return resp.is_success
    except Exception:
        logger.debug("%s health probe failed", name, exc_info=True)
        return False


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Application health check",
    description=(
        "Returns the operational status of the API and its upstream dependencies "
        "(Ollama inference server, MLflow tracking server, Qdrant vector store)."
    ),
)
async def health_check() -> HealthResponse:
    """Probe all dependencies concurrently and return a structured status."""
    settings = get_settings()

    async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
        ollama_ok, mlflow_ok, qdrant_ok = await asyncio.gather(
            _probe(client, f"{settings.ollama_base_url}/api/tags", "Ollama"),
            _probe(client, f"{settings.mlflow_tracking_uri}/health", "MLflow"),
            _probe(client, f"{settings.qdrant_url}/healthz", "Qdrant"),
        )

    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        app_version=settings.app_version,
        ollama_reachable=ollama_ok,
        mlflow_reachable=mlflow_ok,
        qdrant_reachable=qdrant_ok,
    )
