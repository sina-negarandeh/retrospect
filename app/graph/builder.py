from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes import (
    call_model,
    relax_retrieval,
    retrieve_documents,
    rewrite_query,
    should_retry_retrieval,
)
from app.graph.state import RetrospectState


@lru_cache(maxsize=1)
def build_graph() -> CompiledStateGraph:  # type: ignore[type-arg]
    """Build, compile, and cache the Retrospect LangGraph.

    Topology::

        START → rewrite_query → retrieve_documents → (relevant?) → call_model → END
                                        ↑                 │
                                        └─ relax_retrieval ┘

    The retrieve → relax → retrieve cycle is what makes the graph self-correcting:
    a pass that returns nothing, or nothing that clears the cross-encoder
    relevance threshold, gives up one constraint and searches again rather than
    handing an empty context to the generation model. ``should_retry_retrieval``
    bounds the cycle so it always terminates.

    Returns:
        A compiled ``CompiledStateGraph`` instance ready for ``ainvoke`` /
        ``invoke``. Typed as ``CompiledStateGraph`` (not ``StateGraph``) so
        callers can call ``.ainvoke()`` without type errors.
    """
    builder: StateGraph = StateGraph(RetrospectState)  # type: ignore[type-arg]

    # Register nodes
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("retrieve_documents", retrieve_documents)
    builder.add_node("relax_retrieval", relax_retrieval)
    builder.add_node("call_model", call_model)

    # Wire edges using add_edge (not deprecated set_entry_point)
    builder.add_edge(START, "rewrite_query")
    builder.add_edge("rewrite_query", "retrieve_documents")
    builder.add_conditional_edges(
        "retrieve_documents",
        should_retry_retrieval,
        {
            "relax_retrieval": "relax_retrieval",
            "call_model": "call_model",
        },
    )
    # Closes the cycle: a relaxed strategy feeds straight back into retrieval.
    builder.add_edge("relax_retrieval", "retrieve_documents")
    builder.add_edge("call_model", END)

    return builder.compile()
