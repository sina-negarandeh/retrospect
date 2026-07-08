from __future__ import annotations

import logging
from typing import cast

from fastapi import APIRouter, HTTPException, status

from app.domain.models import ChatRequest, ChatResponse
from app.services.llm_service import handle_chat

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat completion",
    description=(
        "Send a message and optional conversation history to the configured LLM. "
        "Returns the model's reply along with token usage and latency metrics."
    ),
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Handle a single chat completion turn.

    Args:
        request: Validated request body from the client.

    Returns:
        ``ChatResponse`` containing the assistant reply and observability data.

    Raises:
        HTTPException (502): If the upstream LLM or graph invocation fails.
    """
    try:
        return cast(ChatResponse, await handle_chat(request))
    except RuntimeError as exc:
        logger.error(
            "Chat completion failed for session=%s: %s",
            request.session_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream LLM service unavailable. Please try again later.",
        ) from exc
