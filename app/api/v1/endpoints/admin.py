from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.config import get_settings
from app.services.ingestion_service import run_ingestion
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin")

# Simple in-process flag so the UI knows if ingestion is running.
_ingestion_running: bool = False
_ingestion_lock = asyncio.Lock()

security = HTTPBearer()

def verify_admin(credentials: HTTPAuthorizationCredentials = Depends(security)) -> None:
    """Verify the bearer token against the admin_token setting."""
    import secrets
    settings = get_settings()
    if not secrets.compare_digest(credentials.credentials, settings.admin_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Response models ────────────────────────────────────────────────────────────


class VectorStoreStatus(BaseModel):
    """Current state of the Qdrant collection."""

    collection_name: str
    point_count: int
    ingestion_running: bool


class IngestRequest(BaseModel):
    """Body for POST /admin/ingest."""

    limit: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum number of documents to ingest. "
            "Omit (or set null) to process all missing documents. "
            "Use a small number (e.g. 3) for quick smoke-tests."
        ),
    )
    wipe_first: bool = Field(
        default=False,
        description=(
            "If true, drop and recreate the vector store collection before "
            "ingesting. All existing embeddings will be deleted. "
            "Use this when you have changed the embedding model."
        ),
    )


class IngestResponse(BaseModel):
    """Summary returned after an ingestion run."""

    status: str
    total_found: int
    already_existed: int
    ingested: int
    skipped: int
    errors: list[str]


# ── Helpers ────────────────────────────────────────────────────────────────────


async def _background_ingest(limit: int | None, wipe_first: bool) -> None:
    """Run ingestion and clear the running flag when done."""
    global _ingestion_running
    try:
        await run_ingestion(limit=limit, wipe_first=wipe_first)
    finally:
        async with _ingestion_lock:
            _ingestion_running = False


# ── Endpoints ──────────────────────────────────────────────────────────────────


@router.get(
    "/vector-store/status",
    response_model=VectorStoreStatus,
    summary="Vector store status",
    description="Returns the current point count in Qdrant and whether ingestion is running.",
    dependencies=[Depends(verify_admin)],
)
async def vector_store_status() -> VectorStoreStatus:
    """Return the current state of the vector store."""
    settings = get_settings()
    try:
        vs = VectorStore()
        await vs.initialize()
        count = await vs.count_documents()
    except Exception as exc:
        logger.error("Cannot reach Qdrant: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cannot reach Qdrant.",
        ) from exc

    async with _ingestion_lock:
        is_running = _ingestion_running

    return VectorStoreStatus(
        collection_name=settings.qdrant_collection_name,
        point_count=count,
        ingestion_running=is_running,
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Trigger document ingestion",
    description=(
        "Kick off the ingestion pipeline. "
        "Use ``limit`` to process only a subset of documents (great for testing). "
        "Use ``wipe_first=true`` to clear all existing embeddings before re-ingesting. "
        "The pipeline runs in the background; poll ``/admin/vector-store/status`` "
        "to track progress."
    ),
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_admin)],
)
async def trigger_ingest(
    body: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """Trigger document ingestion as a background task."""
    global _ingestion_running

    async with _ingestion_lock:
        if _ingestion_running:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Ingestion is already running. Wait for it to finish or restart the server.",
            )
        _ingestion_running = True

    background_tasks.add_task(_background_ingest, body.limit, body.wipe_first)

    logger.info(
        "Ingestion triggered via API. limit=%s wipe_first=%s",
        body.limit,
        body.wipe_first,
    )

    # Return immediately — the actual counts will be 0 since it's still running.
    return IngestResponse(
        status="started",
        total_found=0,
        already_existed=0,
        ingested=0,
        skipped=0,
        errors=[],
    )


@router.delete(
    "/vector-store",
    summary="Wipe vector store",
    description=(
        "Delete ALL embeddings from the Qdrant collection and recreate it empty. "
        "This does NOT re-ingest documents — call ``POST /admin/ingest`` afterwards. "
        "This is irreversible."
    ),
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_admin)],
)
async def wipe_vector_store() -> dict[str, str]:
    """Drop and recreate the Qdrant collection (empty, no documents)."""
    global _ingestion_running

    async with _ingestion_lock:
        if _ingestion_running:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Ingestion is currently running. Stop it by restarting the server first.",
            )

    try:
        from app.services.ingestion_service import _drop_and_recreate_collection

        await _drop_and_recreate_collection()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to wipe collection: {exc}",
        ) from exc

    logger.info("Vector store wiped via DELETE /admin/vector-store.")
    return {"status": "ok", "message": "Collection wiped and recreated empty."}
