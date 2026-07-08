"""Tests for POST /api/v1/chat."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.main import app

client = TestClient(app)


# Fixtures


def _make_graph_result(content: str = "Hello! How can I help you?") -> dict:  # type: ignore[type-arg]
    """Build a fake LangGraph ainvoke result dict."""
    return {
        "messages": [AIMessage(content=content)],
        "session_id": "test-session-123",
        "system_prompt": None,
        "token_usage": {"input_tokens": 12, "output_tokens": 34},
        "latency_ms": {"llm_ms": 150.0},
        "error": None,
    }


# Tests


@pytest.mark.asyncio
async def test_chat_returns_200_with_valid_request() -> None:
    """POST /chat returns 200 and a well-formed ChatResponse."""
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=_make_graph_result())

    with patch("app.services.llm_service.build_graph", return_value=mock_graph):
        response = client.post(
            "/api/v1/chat",
            json={"message": "What is LangGraph?"},
        )

    assert response.status_code == 200
    body = response.json()

    # Verify required response fields
    assert "session_id" in body
    assert "message" in body
    assert "content" in body["message"]
    assert len(body["message"]["content"]) > 0

    # Token usage shape
    assert "token_usage" in body
    assert "input_tokens" in body["token_usage"]
    assert "output_tokens" in body["token_usage"]

    # Latency shape
    assert "latency" in body
    assert "total_ms" in body["latency"]


@pytest.mark.asyncio
async def test_chat_returns_502_when_graph_raises() -> None:
    """POST /chat returns 502 when the LangGraph invocation fails."""
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("Ollama unreachable"))

    with patch("app.services.llm_service.build_graph", return_value=mock_graph):
        response = client.post(
            "/api/v1/chat",
            json={"message": "Trigger failure"},
        )

    assert response.status_code == 502
    body = response.json()
    assert "detail" in body


@pytest.mark.asyncio
async def test_chat_propagates_session_id() -> None:
    """POST /chat echoes back the provided session_id."""
    session_id = "my-custom-session-abc"
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=_make_graph_result())

    with patch("app.services.llm_service.build_graph", return_value=mock_graph):
        response = client.post(
            "/api/v1/chat",
            json={"session_id": session_id, "message": "Hello"},
        )

    assert response.status_code == 200
    assert response.json()["session_id"] == session_id


@pytest.mark.asyncio
async def test_chat_returns_422_for_empty_message() -> None:
    """POST /chat returns 422 Unprocessable Entity when message is empty."""
    response = client.post(
        "/api/v1/chat",
        json={"message": ""},
    )
    assert response.status_code == 422
