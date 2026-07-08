from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints import admin, chat, health, rag

v1_router = APIRouter()

v1_router.include_router(health.router, tags=["Health"])
v1_router.include_router(chat.router, tags=["Chat"])
v1_router.include_router(rag.router, tags=["RAG"])
v1_router.include_router(admin.router, tags=["Admin"])
