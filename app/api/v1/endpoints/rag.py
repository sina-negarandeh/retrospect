from __future__ import annotations

import logging
from typing import cast

from fastapi import APIRouter, HTTPException, status

from app.domain.models import RAGQuery, RAGResponse
from app.services.llm_service import handle_rag

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/rag",
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
    summary="RAG query endpoint",
    description=(
        "Send a query to the RAG system. It will retrieve relevant context "
        "and generate an answer using the LLM."
    ),
)
async def rag_query(request: RAGQuery) -> RAGResponse:
    """Handle a single RAG query.

    Args:
        request: Validated RAGQuery from the client.

    Returns:
        ``RAGResponse`` containing the answer, source chunks, and observability data.

    Raises:
        HTTPException (502): If the upstream LLM or vector store invocation fails.
    """
    try:
        return cast(RAGResponse, await handle_rag(request))
    except RuntimeError as exc:
        logger.error(
            "RAG query failed: %s",
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream RAG service unavailable. Please try again later.",
        ) from exc
