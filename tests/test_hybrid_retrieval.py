import pytest

from app.services.vector_store import VectorStore


@pytest.mark.asyncio
async def test_hybrid_returns_results() -> None:
    """Smoke test to verify that hybrid retrieval (Sparse + Dense + RRF) returns results."""
    vector_store = VectorStore()
    await vector_store.initialize()
    
    results = await vector_store.similarity_search("anxiety at work", top_k=20)
    
    assert len(results) > 0, "Hybrid search returned 0 results."
    assert all("score" in chunk.metadata for chunk in results), "Scores missing from chunk metadata."
    assert all("parent_content" in chunk.metadata or not chunk.metadata.get("is_child", False) for chunk in results), "parent_content missing for chunk."
