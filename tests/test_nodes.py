"""Unit tests for LangGraph nodes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.domain.models import Chunk


def _make_state(
    query: str = "What did I do in Paris?",
    search_query: str | None = None,
    filters: dict | None = None,
    context: list | None = None,
) -> dict:
    return {
        "messages": [HumanMessage(content=query)],
        "session_id": "test-session",
        "system_prompt": None,
        "search_query": search_query,
        "filters": filters,
        "context": context or [],
        "token_usage": {},
        "latency_ms": {},
        "error": None,
    }


def _make_chunk(
    chunk_id: str = "chunk-1",
    doc_id: str = "doc-1",
    content: str = "child content",
    is_child: bool = True,
    parent_content: str = "full parent content",
) -> Chunk:
    metadata: dict = {"cross_score": 0.9}
    if is_child:
        metadata["is_child"] = True
        metadata["parent_content"] = parent_content
    return Chunk(id=chunk_id, document_id=doc_id, content=content, metadata=metadata)


# rewrite_query


class TestRewriteQuery:
    @pytest.mark.asyncio
    async def test_returns_fallback_on_empty_messages(self) -> None:
        from app.graph.nodes import rewrite_query

        state_no_msg = {**_make_state(), "messages": []}
        result = await rewrite_query(state_no_msg)
        assert result["search_query"] == ""

    @pytest.mark.asyncio
    async def test_parses_valid_json_response(self) -> None:
        from app.graph.nodes import rewrite_query

        mock_response = MagicMock()
        mock_response.content = '{"search_query": "Paris Charlotte visit", "filters": {"places": ["Paris"]}}'
        mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("app.graph.nodes._get_rewrite_llm", return_value=mock_llm):
            result = await rewrite_query(_make_state("What did I do in Paris with Charlotte?"))

        assert result["search_query"] == "Paris Charlotte visit"
        assert result["filters"] == {"places": ["Paris"]}

    @pytest.mark.asyncio
    async def test_falls_back_to_original_query_on_invalid_json(self) -> None:
        from app.graph.nodes import rewrite_query

        mock_response = MagicMock()
        mock_response.content = "NOT VALID JSON"
        mock_response.usage_metadata = None

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("app.graph.nodes._get_rewrite_llm", return_value=mock_llm):
            result = await rewrite_query(_make_state("What did I do in Paris?"))

        assert result["search_query"] == "What did I do in Paris?"
        assert result["filters"] is None

    @pytest.mark.asyncio
    async def test_strips_empty_filter_values(self) -> None:
        from app.graph.nodes import rewrite_query

        mock_response = MagicMock()
        mock_response.content = '{"search_query": "travel", "filters": {"places": [], "people": ["Marco"]}}'
        mock_response.usage_metadata = {"input_tokens": 8, "output_tokens": 4}

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("app.graph.nodes._get_rewrite_llm", return_value=mock_llm):
            result = await rewrite_query(_make_state("Marco"))

        # Empty "places" list should be stripped; only "people" should remain
        assert result["filters"] == {"people": ["Marco"]}


# retrieve_documents


class TestRetrieveDocuments:
    @pytest.mark.asyncio
    async def test_expands_child_chunk_to_parent_content(self) -> None:
        from app.graph.nodes import retrieve_documents

        child = _make_chunk(is_child=True, content="child text", parent_content="FULL PARENT")

        mock_vs = AsyncMock()
        mock_vs.similarity_search = AsyncMock(return_value=[child])

        mock_reranker = MagicMock()
        mock_reranker.predict = MagicMock(return_value=[0.95])

        with (
            patch("app.graph.nodes._get_vector_store", return_value=mock_vs),
            patch("app.graph.nodes._get_reranker", return_value=mock_reranker),
        ):
            result = await retrieve_documents(_make_state(search_query="travel"))

        assert len(result["context"]) == 1
        assert result["context"][0].content == "FULL PARENT"
        assert child.content == "child text"

    @pytest.mark.asyncio
    async def test_deduplicates_by_document_id(self) -> None:
        from app.graph.nodes import retrieve_documents

        chunk_a = _make_chunk(chunk_id="c1", doc_id="doc-1", content="chunk a")
        chunk_b = _make_chunk(chunk_id="c2", doc_id="doc-1", content="chunk b")  # same doc

        mock_vs = AsyncMock()
        mock_vs.similarity_search = AsyncMock(return_value=[chunk_a, chunk_b])

        mock_reranker = MagicMock()
        mock_reranker.predict = MagicMock(return_value=[0.9, 0.8])

        with (
            patch("app.graph.nodes._get_vector_store", return_value=mock_vs),
            patch("app.graph.nodes._get_reranker", return_value=mock_reranker),
        ):
            result = await retrieve_documents(_make_state())

        assert len(result["context"]) == 1

    @pytest.mark.asyncio
    async def test_returns_empty_context_on_no_results(self) -> None:
        from app.graph.nodes import retrieve_documents

        mock_vs = AsyncMock()
        mock_vs.similarity_search = AsyncMock(return_value=[])

        with patch("app.graph.nodes._get_vector_store", return_value=mock_vs):
            result = await retrieve_documents(_make_state())

        assert result["context"] == []
