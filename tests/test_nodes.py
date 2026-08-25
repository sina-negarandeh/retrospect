"""Unit tests for LangGraph nodes."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from qdrant_client.http import models as rest

from app.config import get_settings
from app.domain.models import Chunk


def _make_state(
    query: str = "What did I do in Paris?",
    search_query: str | None = None,
    filters: dict | None = None,
    context: list | None = None,
    strategy: str = "filtered",
    attempts: int = 0,
    top_score: float | None = None,
) -> dict:
    return {
        "messages": [HumanMessage(content=query)],
        "session_id": "test-session",
        "system_prompt": None,
        "search_query": search_query,
        "filters": filters,
        "context": context or [],
        "original_query": query,
        "retrieval_strategy": strategy,
        "retrieval_attempts": attempts,
        "retrieval_top_k": None,
        "top_score": top_score,
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


class TestTemporalFilters:
    """Date bounds survive the LLM -> Pydantic -> Qdrant hand-off."""

    @pytest.mark.asyncio
    async def test_parses_date_bounds_from_llm_output(self) -> None:
        from app.graph.nodes import rewrite_query

        mock_response = MagicMock()
        mock_response.content = (
            '{"search_query": "garden walk", '
            '"filters": {"date_from": "1771-05-10", "date_to": "1771-05-20"}}'
        )
        mock_response.usage_metadata = {"input_tokens": 12, "output_tokens": 6}

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("app.graph.nodes._get_rewrite_llm", return_value=mock_llm):
            result = await rewrite_query(_make_state("What did I write in mid May 1771?"))

        assert result["filters"] == {"date_from": "1771-05-10", "date_to": "1771-05-20"}

    @pytest.mark.asyncio
    async def test_drops_unparseable_date_but_keeps_other_filters(self) -> None:
        from app.graph.nodes import rewrite_query

        mock_response = MagicMock()
        mock_response.content = (
            '{"search_query": "aunt", '
            '"filters": {"people": ["Charlotte"], "date_from": "last spring"}}'
        )
        mock_response.usage_metadata = None

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("app.graph.nodes._get_rewrite_llm", return_value=mock_llm):
            result = await rewrite_query(_make_state("What did I say about Charlotte?"))

        # The bad bound is dropped; the usable filter survives.
        assert result["filters"] == {"people": ["Charlotte"]}

    def test_builds_inclusive_epoch_range(self) -> None:
        from app.date_utils import to_epoch_seconds
        from app.services.vector_store import _build_date_range

        rng = _build_date_range({"date_from": "1771-05-10", "date_to": "1771-05-20"})

        assert rng is not None
        # An entry dated exactly on the upper bound encodes to that bound, so
        # ``lte`` includes the final day.
        assert rng.gte == to_epoch_seconds(date(1771, 5, 10))
        assert rng.lte == to_epoch_seconds(date(1771, 5, 20))

    def test_open_ended_range_leaves_one_bound_unset(self) -> None:
        from app.services.vector_store import _build_date_range

        rng = _build_date_range({"date_to": "2019-12-31"})

        assert rng is not None
        assert rng.gte is None
        assert rng.lte is not None

    def test_no_date_keys_yields_no_range(self) -> None:
        from app.services.vector_store import _build_date_range

        assert _build_date_range({"people": ["Charlotte"]}) is None

    def test_inverted_bounds_are_swapped(self) -> None:
        from app.domain.models import MetadataFilters

        filters = MetadataFilters.model_validate(
            {"date_from": "1771-05-20", "date_to": "1771-05-10"}
        )

        assert filters.date_from == date(1771, 5, 10)
        assert filters.date_to == date(1771, 5, 20)

    def test_date_filter_reaches_qdrant_as_a_range_condition(self) -> None:
        from app.date_utils import DATE_PAYLOAD_FIELD
        from app.services.vector_store import _build_date_range

        rng = _build_date_range({"date_from": "1771-05-01"})
        assert rng is not None

        condition = rest.FieldCondition(key=DATE_PAYLOAD_FIELD, range=rng)
        assert condition.key == DATE_PAYLOAD_FIELD
        assert condition.range is not None


class TestRetrievalLoop:
    """The retrieve -> relax -> retrieve cycle and the guards that stop it."""

    def test_relevant_context_goes_straight_to_generation(self) -> None:
        from app.graph.nodes import should_retry_retrieval

        state = _make_state(context=[_make_chunk()], attempts=1, top_score=4.2)
        assert should_retry_retrieval(state) == "call_model"

    def test_empty_context_triggers_a_retry(self) -> None:
        from app.graph.nodes import should_retry_retrieval

        state = _make_state(context=[], attempts=1, top_score=None)
        assert should_retry_retrieval(state) == "relax_retrieval"

    def test_below_threshold_context_triggers_a_retry(self) -> None:
        from app.graph.nodes import should_retry_retrieval

        state = _make_state(context=[_make_chunk()], attempts=1, top_score=-6.0)
        assert should_retry_retrieval(state) == "relax_retrieval"

    def test_attempt_budget_stops_the_cycle(self) -> None:
        from app.config import get_settings
        from app.graph.nodes import should_retry_retrieval

        exhausted = get_settings().retrieval_max_attempts
        state = _make_state(context=[], attempts=exhausted, top_score=None)
        assert should_retry_retrieval(state) == "call_model"

    def test_exhausted_ladder_stops_the_cycle(self) -> None:
        from app.graph.nodes import should_retry_retrieval

        # Already fully broadened — another pass would repeat the same search.
        state = _make_state(context=[], attempts=1, strategy="broad")
        assert should_retry_retrieval(state) == "call_model"

    def test_empty_messages_stop_the_cycle(self) -> None:
        from app.graph.nodes import should_retry_retrieval

        state = {**_make_state(context=[]), "messages": []}
        assert should_retry_retrieval(state) == "call_model"

    @pytest.mark.asyncio
    async def test_first_relaxation_drops_the_filters(self) -> None:
        from app.graph.nodes import relax_retrieval

        state = _make_state(search_query="paris trip", filters={"places": ["Paris"]})
        result = await relax_retrieval(state)

        assert result["retrieval_strategy"] == "unfiltered"
        assert result["filters"] is None
        # The rewritten query is untouched on this rung.
        assert "search_query" not in result

    @pytest.mark.asyncio
    async def test_second_relaxation_restores_the_verbatim_query(self) -> None:
        from app.config import get_settings
        from app.graph.nodes import relax_retrieval

        state = _make_state(
            "What did I do in Paris?", search_query="paris trip", strategy="unfiltered"
        )
        result = await relax_retrieval(state)

        assert result["retrieval_strategy"] == "broad"
        assert result["search_query"] == "What did I do in Paris?"
        assert result["retrieval_top_k"] == get_settings().retrieval_broad_top_k

    @pytest.mark.asyncio
    async def test_unfiltered_first_pass_skips_straight_to_broad(self) -> None:
        from app.graph.nodes import relax_retrieval

        # Nothing to unfilter, so the no-op rung is skipped rather than burning
        # an attempt on an identical search.
        state = _make_state(search_query="paris trip", filters=None)
        result = await relax_retrieval(state)

        assert result["retrieval_strategy"] == "broad"

    @pytest.mark.asyncio
    async def test_retrieve_documents_counts_attempts_and_reports_top_score(self) -> None:
        from app.graph.nodes import retrieve_documents

        chunk = _make_chunk()
        mock_vs = AsyncMock()
        mock_vs.similarity_search = AsyncMock(return_value=[chunk])

        mock_reranker = MagicMock()
        mock_reranker.predict = MagicMock(return_value=[3.5])

        with (
            patch("app.graph.nodes._get_vector_store", return_value=mock_vs),
            patch("app.graph.nodes._get_reranker", return_value=mock_reranker),
        ):
            result = await retrieve_documents(_make_state(attempts=1))

        assert result["retrieval_attempts"] == 2
        assert result["top_score"] == 3.5

    @pytest.mark.asyncio
    async def test_empty_pass_still_counts_as_an_attempt(self) -> None:
        from app.graph.nodes import retrieve_documents

        mock_vs = AsyncMock()
        mock_vs.similarity_search = AsyncMock(return_value=[])

        with patch("app.graph.nodes._get_vector_store", return_value=mock_vs):
            result = await retrieve_documents(_make_state(attempts=0))

        # Without this the loop could spin forever on a permanently empty store.
        assert result["retrieval_attempts"] == 1
        assert result["top_score"] is None

    @pytest.mark.asyncio
    async def test_loop_recovers_when_relaxing_finds_results(self) -> None:
        """End-to-end: an over-eager filter returns nothing, the retry succeeds."""
        from app.graph.nodes import relax_retrieval, retrieve_documents, should_retry_retrieval

        calls: list[dict | None] = []

        async def _search(query: str, filters: dict | None = None, top_k: int = 20) -> list:
            calls.append(filters)
            return [] if filters else [_make_chunk()]

        mock_vs = AsyncMock()
        mock_vs.similarity_search = _search

        mock_reranker = MagicMock()
        mock_reranker.predict = MagicMock(return_value=[2.0])

        state = _make_state(search_query="garden", filters={"places": ["Atlantis"]})

        with (
            patch("app.graph.nodes._get_vector_store", return_value=mock_vs),
            patch("app.graph.nodes._get_reranker", return_value=mock_reranker),
        ):
            state.update(await retrieve_documents(state))
            assert should_retry_retrieval(state) == "relax_retrieval"

            state.update(await relax_retrieval(state))
            state.update(await retrieve_documents(state))
            assert should_retry_retrieval(state) == "call_model"

        assert calls == [{"places": ["Atlantis"]}, None]
        assert len(state["context"]) == 1
        assert state["retrieval_attempts"] == 2


class TestGraphTopology:
    def test_graph_contains_the_retrieval_cycle(self) -> None:
        from app.graph.builder import build_graph

        edges = {(e.source, e.target) for e in build_graph().get_graph().edges}

        assert ("retrieve_documents", "relax_retrieval") in edges
        assert ("relax_retrieval", "retrieve_documents") in edges
        assert ("retrieve_documents", "call_model") in edges

    @pytest.mark.asyncio
    async def test_compiled_graph_cycles_recovers_and_terminates(self) -> None:
        """Drive the whole graph: a dead-end filter must not produce a dead-end answer.

        Walks every rung of the ladder — filtered, unfiltered, broad — and
        asserts the run still ends with context and a reply rather than
        looping or bottoming out empty.
        """
        from app.graph.builder import build_graph

        searches: list[dict] = []

        async def _search(query: str, filters: dict | None = None, top_k: int = 20) -> list:
            searches.append({"query": query, "filters": filters, "top_k": top_k})
            if filters or "Atlantis" in query:
                return []
            return [_make_chunk(parent_content="FULL ENTRY TEXT")]

        rewrite = MagicMock()
        rewrite.content = (
            '{"search_query": "Atlantis garden", '
            '"filters": {"places": ["Atlantis"], "date_from": "1771-05-01"}}'
        )
        rewrite.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
        rewrite_llm = MagicMock()
        rewrite_llm.ainvoke = AsyncMock(return_value=rewrite)

        answer = AIMessage(content="You wrote about the garden.")
        answer.usage_metadata = {"input_tokens": 100, "output_tokens": 20}
        chat_llm = MagicMock()
        chat_llm.ainvoke = AsyncMock(return_value=answer)

        mock_vs = AsyncMock()
        mock_vs.similarity_search = _search
        mock_reranker = MagicMock()
        mock_reranker.predict = MagicMock(return_value=[5.0])

        with (
            patch("app.graph.nodes._get_rewrite_llm", return_value=rewrite_llm),
            patch("app.graph.nodes._get_chat_llm", return_value=chat_llm),
            patch("app.graph.nodes._get_vector_store", return_value=mock_vs),
            patch("app.graph.nodes._get_reranker", return_value=mock_reranker),
        ):
            result = await build_graph().ainvoke(
                {
                    "messages": [HumanMessage(content="Who laid out the garden in May 1771?")],
                    "session_id": "test-session",
                    "system_prompt": None,
                    "token_usage": {},
                    "latency_ms": {},
                    "error": None,
                }
            )

        assert len(searches) == 3, "expected three passes around the cycle"
        assert searches[0]["filters"] == {"places": ["Atlantis"], "date_from": "1771-05-01"}
        assert searches[1]["filters"] is None  # rung 1: filters dropped
        assert searches[2]["query"] == "Who laid out the garden in May 1771?"  # rung 2: verbatim
        assert searches[2]["top_k"] == get_settings().retrieval_broad_top_k  # ...wider pool

        assert result["retrieval_attempts"] == 3
        assert result["retrieval_strategy"] == "broad"
        assert len(result["context"]) == 1
        assert result["messages"][-1].content == "You wrote about the garden."
