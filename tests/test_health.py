"""Tests for GET /api/v1/health."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _make_mock_response(status_code: int) -> AsyncMock:
    """Return an AsyncMock that behaves like an httpx.Response."""
    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.is_success = status_code < 400
    mock_response.status_code = status_code
    return mock_response

@pytest.mark.asyncio
async def test_health_returns_200_when_all_deps_healthy() -> None:
    """Health endpoint returns 200 with status=ok when all probes succeed."""
    mock_response = _make_mock_response(200)

    with patch(
        "app.api.v1.endpoints.health.httpx.AsyncClient.get",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "app_name" in body
    assert "app_version" in body
    assert body["ollama_reachable"] is True
    assert body["mlflow_reachable"] is True
    assert body["qdrant_reachable"] is True


@pytest.mark.asyncio
async def test_health_returns_200_when_ollama_unreachable() -> None:
    """Health endpoint still returns 200 even when Ollama is down."""

    async def _failing_get(url: str, **kwargs: object) -> httpx.Response:
        if "11434" in url or "api/tags" in url:
            raise httpx.ConnectError("connection refused")
        mock = _make_mock_response(200)
        return mock

    with patch(
        "app.api.v1.endpoints.health.httpx.AsyncClient.get",
        new_callable=AsyncMock,
        side_effect=_failing_get,
    ):
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ollama_reachable"] is False


@pytest.mark.asyncio
async def test_health_returns_200_when_mlflow_unreachable() -> None:
    """Health endpoint still returns 200 even when MLflow is down."""

    async def _failing_get(url: str, **kwargs: object) -> httpx.Response:
        if "5000" in url or "mlflow" in url:
            raise httpx.ConnectError("connection refused")
        mock = _make_mock_response(200)
        return mock

    with patch(
        "app.api.v1.endpoints.health.httpx.AsyncClient.get",
        new_callable=AsyncMock,
        side_effect=_failing_get,
    ):
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mlflow_reachable"] is False


@pytest.mark.asyncio
async def test_health_returns_200_when_qdrant_unreachable() -> None:
    """Health endpoint still returns 200 even when Qdrant is down."""

    async def _failing_get(url: str, **kwargs: object) -> httpx.Response:
        if "6333" in url or "qdrant" in url:
            raise httpx.ConnectError("connection refused")
        mock = _make_mock_response(200)
        return mock

    with patch(
        "app.api.v1.endpoints.health.httpx.AsyncClient.get",
        new_callable=AsyncMock,
        side_effect=_failing_get,
    ):
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["qdrant_reachable"] is False
