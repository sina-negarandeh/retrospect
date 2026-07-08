from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import mlflow
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import get_settings
from app.domain.enums import MessageRole
from app.domain.models import (
    ChatRequest,
    ChatResponse,
    Chunk,
    LatencyMetrics,
    Message,
    RAGQuery,
    RAGResponse,
    TokenUsage,
)
from app.graph.builder import build_graph

logger = logging.getLogger(__name__)


async def _invoke_graph(
    messages: list[BaseMessage],
    session_id: str,
    system_prompt: str | None = None,
) -> tuple[dict[str, Any], float]:
    graph = build_graph()
    t0 = time.perf_counter()
    try:
        result: dict[str, Any] = await graph.ainvoke(
            {
                "messages": messages,
                "session_id": session_id,
                "system_prompt": system_prompt,
                "token_usage": {},
                "latency_ms": {},
                "error": None,
            }
        )
    except Exception as exc:
        logger.exception("Graph invocation failed for session=%s", session_id)
        raise RuntimeError(f"LLM orchestration failed: {exc}") from exc

    total_ms = (time.perf_counter() - t0) * 1_000
    return result, total_ms


def _extract_metrics(
    result: dict[str, Any], total_ms: float
) -> tuple[TokenUsage, LatencyMetrics]:
    """Extract ``TokenUsage`` and ``LatencyMetrics`` from a graph result dict."""
    raw_usage: dict[str, int] = result.get("token_usage", {})
    raw_latency: dict[str, float] = result.get("latency_ms", {})

    token_usage = TokenUsage(
        input_tokens=raw_usage.get("input_tokens", 0),
        output_tokens=raw_usage.get("output_tokens", 0),
    )
    latency = LatencyMetrics(
        llm_ms=raw_latency.get("llm_ms", 0.0),
        total_ms=round(total_ms, 3),
    )
    return token_usage, latency


async def handle_chat(request: ChatRequest) -> ChatResponse:
    settings = get_settings()

    with mlflow.start_span(name="Chat_Pipeline", span_type="CHAIN") as span:
        span.set_inputs(
            {
                "session_id": request.session_id,
                "message": request.message,
                "system_prompt": request.system_prompt or "",
            }
        )

        # Build message list (system prompt handled by the graph node, not here)
        lc_messages: list[BaseMessage] = []
        for msg in request.conversation_history:
            if msg.role == MessageRole.user:
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == MessageRole.assistant:
                lc_messages.append(AIMessage(content=msg.content))
            # system messages in history are skipped; supplied via system_prompt
        lc_messages.append(HumanMessage(content=request.message))

        result, total_ms = await _invoke_graph(
            messages=lc_messages,
            session_id=request.session_id,
            system_prompt=request.system_prompt,
        )

        token_usage, latency = _extract_metrics(result, total_ms)
        last_lc_message = result["messages"][-1]
        assistant_message = Message(
            role=MessageRole.assistant,
            content=str(last_lc_message.content),
        )

        span.set_outputs(assistant_message.content)
        span.set_attributes(
            {
                "total_latency_ms": latency.total_ms,
                "llm_latency_ms": latency.llm_ms,
                "input_tokens": token_usage.input_tokens,
                "output_tokens": token_usage.output_tokens,
                "model": settings.ollama_model,
            }
        )
        
        try:
            mlflow.log_metrics({
                "total_latency_ms": latency.total_ms,
                "llm_latency_ms": latency.llm_ms,
                "input_tokens": token_usage.input_tokens,
                "output_tokens": token_usage.output_tokens,
            })
        except Exception as e:
            logger.debug("Failed to log MLflow metrics: %s", e)

        return ChatResponse(
            session_id=request.session_id,
            message=assistant_message,
            token_usage=token_usage,
            latency=latency,
            model=settings.ollama_model,
        )


async def handle_rag(request: RAGQuery) -> RAGResponse:
    settings = get_settings()
    rag_session_id = f"rag-{uuid.uuid4()}"

    with mlflow.start_span(name="RAG_Pipeline", span_type="CHAIN") as span:
        span.set_inputs({"session_id": rag_session_id, "query": request.query})

        lc_messages: list[BaseMessage] = [HumanMessage(content=request.query)]
        result, total_ms = await _invoke_graph(
            messages=lc_messages,
            session_id=rag_session_id,
            system_prompt=None,
        )

        token_usage, latency = _extract_metrics(result, total_ms)
        context_chunks: list[Chunk] = result.get("context", [])
        last_lc_message = result["messages"][-1]

        span.set_outputs(str(last_lc_message.content))
        span.set_attributes(
            {
                "total_latency_ms": latency.total_ms,
                "llm_latency_ms": latency.llm_ms,
                "input_tokens": token_usage.input_tokens,
                "output_tokens": token_usage.output_tokens,
                "model": settings.ollama_model,
            }
        )
        
        try:
            mlflow.log_metrics({
                "total_latency_ms": latency.total_ms,
                "llm_latency_ms": latency.llm_ms,
                "input_tokens": token_usage.input_tokens,
                "output_tokens": token_usage.output_tokens,
            })
        except Exception as e:
            logger.debug("Failed to log MLflow metrics: %s", e)

        return RAGResponse(
            answer=str(last_lc_message.content),
            source_chunks=context_chunks,
            token_usage=token_usage,
            latency=latency,
        )
