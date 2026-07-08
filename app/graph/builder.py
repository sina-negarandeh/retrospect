from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes import call_model, retrieve_documents, rewrite_query
from app.graph.state import RetrospectState


@lru_cache(maxsize=1)
def build_graph() -> CompiledStateGraph:  # type: ignore[type-arg]
    """Build, compile, and cache the Retrospect LangGraph.

    Returns:
        A compiled ``CompiledStateGraph`` instance ready for ``ainvoke`` /
        ``invoke``. Typed as ``CompiledStateGraph`` (not ``StateGraph``) so
        callers can call ``.ainvoke()`` without type errors.
    """
    builder: StateGraph = StateGraph(RetrospectState)  # type: ignore[type-arg]

    # Register nodes
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("retrieve_documents", retrieve_documents)
    builder.add_node("call_model", call_model)

    # Wire edges using add_edge (not deprecated set_entry_point)
    builder.add_edge(START, "rewrite_query")
    builder.add_edge("rewrite_query", "retrieve_documents")
    builder.add_edge("retrieve_documents", "call_model")
    builder.add_edge("call_model", END)

    return builder.compile()
