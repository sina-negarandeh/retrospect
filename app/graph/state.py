from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph import MessagesState

from app.domain.models import Chunk


def add_metrics(
    left: dict[str, int | float], right: dict[str, int | float] | None
) -> dict[str, int | float]:
    """Reducer that adds numeric values for overlapping keys."""
    if not right:
        return left
    
    result = left.copy()
    for k, v in right.items():
        result[k] = result.get(k, 0) + v
    return result


class RetrospectState(MessagesState):
    """Full state carried through the Retrospect LangGraph.

    Inherits:
        messages: list[AnyMessage]  (managed by MessagesState add-reducer)

    Additions:
        session_id:    Unique identifier tying the graph run to an HTTP session.
        system_prompt: Optional system-level instruction injected at the start.
        token_usage:   Accumulated token counts; merged via ``operator.or_``.
        latency_ms:    Accumulated latency measurements; merged via ``operator.or_``.
        error:         Human-readable error message if a node fails.

    Retrieval-loop additions (see ``app.graph.nodes.should_retry_retrieval``):
        original_query:     The verbatim user question, kept so the final
                            broadened attempt can fall back to it after the
                            rewrite step has narrowed the wording.
        retrieval_strategy: Which rung of the relaxation ladder the current
                            attempt is on — "filtered", "unfiltered", "broad".
        retrieval_attempts: Passes through ``retrieve_documents`` so far; the
                            hard stop that guarantees the cycle terminates.
        retrieval_top_k:    Candidate pool size for the current attempt.
        top_score:          Best cross-encoder score in the current result set,
                            compared against the relevance threshold to decide
                            whether to relax and retry.
    """

    session_id: str
    system_prompt: str | None
    search_query: str | None
    filters: dict[str, Any] | None
    context: list[Chunk]
    original_query: str | None
    retrieval_strategy: str
    retrieval_attempts: int
    retrieval_top_k: int | None
    top_score: float | None
    token_usage: Annotated[dict[str, int], add_metrics]
    latency_ms: Annotated[dict[str, float], add_metrics]
    error: str | None
