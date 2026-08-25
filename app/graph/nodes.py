from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

import mlflow
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.domain.models import TranslatedQuery
from app.graph.state import RetrospectState
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Rungs of the retrieval relaxation ladder, in the order the loop walks them.
# Each failed pass gives up one constraint, so a retry is never a verbatim
# repeat of the attempt that just came back empty.
STRATEGY_FILTERED = "filtered"  # rewritten query + self-query metadata filters
STRATEGY_UNFILTERED = "unfiltered"  # rewritten query, filters dropped
STRATEGY_BROAD = "broad"  # verbatim user question, no filters, wider pool

_vector_store_instance: VectorStore | None = None
_vector_store_lock = asyncio.Lock()

_reranker_instance: Any = None
_reranker_lock = asyncio.Lock()


async def _get_vector_store() -> VectorStore:
    """Return the process-level singleton VectorStore (thread-safe init)."""
    global _vector_store_instance
    if _vector_store_instance is None:
        async with _vector_store_lock:
            if _vector_store_instance is None:  # double-checked locking
                instance = VectorStore()
                await instance.initialize()
                _vector_store_instance = instance
    return _vector_store_instance


async def _get_reranker() -> Any:
    """Return the process-level singleton CrossEncoder (thread-safe init)."""
    global _reranker_instance
    if _reranker_instance is None:
        async with _reranker_lock:
            if _reranker_instance is None:
                from sentence_transformers import CrossEncoder

                # CrossEncoder loads a PyTorch model — run in a thread to avoid
                # blocking the event loop during the first request.
                _reranker_instance = await asyncio.to_thread(
                    CrossEncoder,
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    max_length=512,
                )
    return _reranker_instance


def _build_rewrite_llm() -> ChatOllama:
    """Build the ChatOllama client for query rewriting (JSON mode, small ctx)."""
    settings = get_settings()
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        num_ctx=2048,
        temperature=0.0,
        format="json",
    )


def _build_chat_llm() -> ChatOllama:
    """Build the ChatOllama client for answer generation (large ctx)."""
    settings = get_settings()
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        num_ctx=16384,
    )


_rewrite_llm: ChatOllama | None = None
_chat_llm: ChatOllama | None = None


def _get_rewrite_llm() -> ChatOllama:
    global _rewrite_llm
    if _rewrite_llm is None:
        _rewrite_llm = _build_rewrite_llm()
    return _rewrite_llm


def _get_chat_llm() -> ChatOllama:
    global _chat_llm
    if _chat_llm is None:
        _chat_llm = _build_chat_llm()
    return _chat_llm



@mlflow.trace(name="Rewrite_Query", span_type="TOOL")
async def rewrite_query(state: RetrospectState) -> dict[str, Any]:
    """Rewrite the user's query into keywords optimised for vector search.

    Also performs self-querying: the model emits deterministic metadata and
    date-range filters alongside the semantic query, which are validated
    through ``TranslatedQuery`` before they are ever handed to Qdrant.

    Returns a partial state dict with ``search_query``, ``filters``, the
    initial retrieval-loop state, ``token_usage``, and ``latency_ms``.
    """
    settings = get_settings()
    messages = state.get("messages", [])
    if not messages:
        return {"search_query": ""}

    query = str(messages[-1].content)

    if span := mlflow.get_current_active_span():
        span.set_inputs({"user_query": query})

    llm = _get_rewrite_llm()

    from app.prompts import QUERY_REWRITE_PROMPT

    # Relative expressions ("last month") can only be resolved against a
    # concrete reference date, so today's date is pinned into the prompt.
    prompt = QUERY_REWRITE_PROMPT.format(
        query=query,
        today=datetime.now(UTC).date().isoformat(),
    )

    t0 = time.perf_counter()
    async for attempt in AsyncRetrying(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    ):
        with attempt:
            response = await llm.ainvoke(prompt)
    elapsed_ms = (time.perf_counter() - t0) * 1_000

    rewritten: str = query
    filters: dict[str, Any] | None = None

    try:
        content = json.loads(str(response.content).strip())

        # Validate before use: values arrive straight from an LLM, and an
        # off-vocabulary or malformed filter silently matches zero points.
        translated = TranslatedQuery.model_validate(content)
        rewritten = translated.search_query.strip() or query
        filters = translated.filters.to_query_filters() or None

    except Exception:
        # Log the failure so degraded retrieval is visible in traces.
        logger.warning(
            "rewrite_query: failed to parse LLM JSON output; falling back to raw query.",
            exc_info=True,
        )

    _meta = response.usage_metadata
    token_usage: dict[str, int] = {
        "input_tokens": int(_meta.get("input_tokens", 0)) if _meta is not None else 0,
        "output_tokens": int(_meta.get("output_tokens", 0)) if _meta is not None else 0,
    }

    if span := mlflow.get_current_active_span():
        span.set_outputs(
            {
                "search_query": rewritten,
                "filters": filters,
                "input_tokens": token_usage["input_tokens"],
                "output_tokens": token_usage["output_tokens"],
                "latency_ms": round(elapsed_ms, 3),
            }
        )

    return {
        "search_query": rewritten,
        "filters": filters,
        "original_query": query,
        "retrieval_strategy": STRATEGY_FILTERED,
        "retrieval_attempts": 0,
        "retrieval_top_k": settings.retrieval_top_k,
        "top_score": None,
        "token_usage": token_usage,
        "latency_ms": {"rewrite_ms": round(elapsed_ms, 3)},
    }


@mlflow.trace(name="Retrieve_Documents", span_type="RETRIEVER")
async def retrieve_documents(state: RetrospectState) -> dict[str, Any]:
    """Retrieve and re-rank documents relevant to the latest user message.

    Pipeline:
        1. Hybrid search (dense + sparse + RRF) via Qdrant, under whichever
           strategy the retrieval loop is currently on.
        2. Cross-encoder re-ranking of the top-k pool.
        3. Small-to-big expansion for re-ranked survivors.

    This node may run several times per request — see ``should_retry_retrieval``.
    """
    settings = get_settings()
    attempt_no = int(state.get("retrieval_attempts", 0)) + 1
    messages = state.get("messages", [])
    if not messages:
        # Still counts as an attempt so the loop cannot spin on an empty state.
        return {"context": [], "top_score": None, "retrieval_attempts": attempt_no}

    query = state.get("search_query") or str(messages[-1].content)
    filters = state.get("filters")
    strategy = state.get("retrieval_strategy") or STRATEGY_FILTERED
    top_k = state.get("retrieval_top_k") or settings.retrieval_top_k

    if span := mlflow.get_current_active_span():
        span.set_inputs(
            {
                "query": query,
                "filters": filters,
                "top_k": top_k,
                "strategy": strategy,
                "attempt": attempt_no,
            }
        )

    t0 = time.perf_counter()
    vector_store = await _get_vector_store()
    merged_chunks = await vector_store.similarity_search(
        query=query, filters=filters, top_k=top_k
    )

    top_chunks = []
    top_score: float | None = None
    if merged_chunks:
        reranker = await _get_reranker()
        pairs = [[query, c.content] for c in merged_chunks]
        scores = await asyncio.to_thread(reranker.predict, pairs)

        for chunk, score in zip(merged_chunks, scores, strict=False):
            chunk.metadata["cross_score"] = float(score)

        merged_chunks.sort(key=lambda c: c.metadata["cross_score"], reverse=True)
        top_chunks = merged_chunks[: settings.retrieval_final_k]
        # The best score is the loop's relevance signal: a pool that survives
        # re-ranking with nothing above the threshold is a miss, not an answer.
        top_score = float(merged_chunks[0].metadata["cross_score"])

    final_chunks = []
    seen_docs: set[str] = set()
    for c in top_chunks:
        doc_id = c.document_id
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            if c.metadata.get("is_child") and "parent_content" in c.metadata:
                # Replace content with parent but preserve all other metadata.
                c = c.model_copy(update={"content": c.metadata["parent_content"]})
            final_chunks.append(c)

    elapsed_ms = (time.perf_counter() - t0) * 1_000

    if span := mlflow.get_current_active_span():
        span.set_outputs(
            {
                "chunk_count": len(final_chunks),
                "used_filters": filters is not None,
                "strategy": strategy,
                "attempt": attempt_no,
                "top_score": top_score,
                "latency_ms": round(elapsed_ms, 3),
            }
        )

    return {
        "context": final_chunks,
        "top_score": top_score,
        "retrieval_attempts": attempt_no,
        "latency_ms": {"retrieval_ms": round(elapsed_ms, 3)},
    }


@mlflow.trace(name="Relax_Retrieval", span_type="TOOL")
async def relax_retrieval(state: RetrospectState) -> dict[str, Any]:
    """Give up one retrieval constraint, then hand back to ``retrieve_documents``.

    This is the body of the cycle. Dropping the self-query filters comes first
    because an over-eager metadata or date filter is the most common reason a
    hybrid search returns nothing: the filtered prefetch paths exclude every
    point before RRF fusion ever sees them. Only when there is nothing left to
    unfilter does it fall back to the verbatim question, which restores terms
    the rewrite step may have discarded, over a wider candidate pool.
    """
    settings = get_settings()
    strategy = state.get("retrieval_strategy") or STRATEGY_FILTERED

    if strategy == STRATEGY_FILTERED and state.get("filters"):
        logger.info("Retrieval relaxed: dropping self-query filters and retrying.")
        relaxed = {
            "retrieval_strategy": STRATEGY_UNFILTERED,
            "filters": None,
            "retrieval_top_k": settings.retrieval_top_k,
        }
    else:
        logger.info("Retrieval relaxed: falling back to the verbatim query over a wider pool.")
        relaxed = {
            "retrieval_strategy": STRATEGY_BROAD,
            "filters": None,
            "search_query": state.get("original_query") or state.get("search_query"),
            "retrieval_top_k": settings.retrieval_broad_top_k,
        }

    if span := mlflow.get_current_active_span():
        span.set_inputs({"from_strategy": strategy, "top_score": state.get("top_score")})
        span.set_outputs(relaxed)

    return relaxed


def should_retry_retrieval(state: RetrospectState) -> str:
    """Conditional edge: relax and search again, or go answer the question.

    Retries when the pass came back empty, or when nothing in it cleared the
    cross-encoder relevance threshold. Terminates on three independent
    guarantees — the attempt budget, an exhausted relaxation ladder, and an
    empty message list — so the cycle can never spin.
    """
    settings = get_settings()

    if not state.get("messages"):
        return "call_model"

    if int(state.get("retrieval_attempts", 0)) >= settings.retrieval_max_attempts:
        return "call_model"

    if (state.get("retrieval_strategy") or STRATEGY_FILTERED) == STRATEGY_BROAD:
        return "call_model"  # nothing left to give up

    context = state.get("context") or []
    if not context:
        return "relax_retrieval"

    top_score = state.get("top_score")
    if top_score is not None and top_score < settings.retrieval_relevance_threshold:
        return "relax_retrieval"

    return "call_model"


@mlflow.trace(name="Call_Model", span_type="CHAT_MODEL")
async def call_model(state: RetrospectState) -> dict[str, Any]:
    settings = get_settings()
    llm = _get_chat_llm()
    context_chunks = state.get("context", [])

    if span := mlflow.get_current_active_span():
        messages = state.get("messages", [])
        last_msg = str(messages[-1].content) if messages else ""
        span.set_inputs(
            {
                "user_message": last_msg,
                "context_chunk_count": len(context_chunks),
                "model": settings.ollama_model,
            }
        )

    if context_chunks:
        context_str = "\n\n".join(
            [f"Document {c.document_id}:\n{c.content}" for c in context_chunks]
        )
        from app.prompts import ANSWER_GENERATION_PROMPT
        rag_prompt = ANSWER_GENERATION_PROMPT.format(context_str=context_str)

        base_system_prompt = state.get("system_prompt")
        final_system_prompt = (
            f"{base_system_prompt}\n\n{rag_prompt}" if base_system_prompt else rag_prompt
        )
        messages_to_send = [SystemMessage(content=final_system_prompt)] + state["messages"]
    else:
        messages_to_send = list(state["messages"])
        if state.get("system_prompt"):
            messages_to_send = [SystemMessage(content=state["system_prompt"])] + messages_to_send

    t0 = time.perf_counter()
    async for attempt in AsyncRetrying(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    ):
        with attempt:
            response = await llm.ainvoke(messages_to_send)
    elapsed_ms = (time.perf_counter() - t0) * 1_000

    _meta = response.usage_metadata
    token_usage: dict[str, int] = {
        "input_tokens": int(_meta.get("input_tokens", 0)) if _meta is not None else 0,
        "output_tokens": int(_meta.get("output_tokens", 0)) if _meta is not None else 0,
    }

    if span := mlflow.get_current_active_span():
        span.set_outputs(
            {
                "reply": str(response.content),
                "input_tokens": token_usage["input_tokens"],
                "output_tokens": token_usage["output_tokens"],
                "latency_ms": round(elapsed_ms, 3),
            }
        )

    return {
        "messages": [response],
        "token_usage": token_usage,
        "latency_ms": {"llm_ms": round(elapsed_ms, 3)},
    }
