"""Domain models for pure data shapes, with no business logic.

Design decisions:
- Value objects (TokenUsage, LatencyMetrics, Message) are ``frozen=True``
  so they are hashable and immutable after construction.
- Aggregate roots (Conversation, ChatRequest, ChatResponse) are mutable to
  allow downstream code to update their state without re-constructing them.
- ``uuid4`` default factories ensure globally unique IDs without external
  coordination.
- ``datetime.now(UTC)`` is used everywhere instead of the deprecated
  ``datetime.utcnow()`` to produce timezone-aware timestamps.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from app.domain.enums import ConversationStatus, MessageRole

# Value objects (immutable)


class TokenUsage(BaseModel, frozen=True):
    """Token consumption for a single LLM call."""

    input_tokens: int = Field(ge=0, description="Tokens supplied in the prompt.")
    output_tokens: int = Field(ge=0, description="Tokens generated in the completion.")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LatencyMetrics(BaseModel, frozen=True):
    """Wall-clock latency measurements for a single request."""

    llm_ms: float = Field(ge=0.0, description="Time spent inside the LLM call (ms).")
    total_ms: float = Field(ge=0.0, description="End-to-end request latency (ms).")


class Message(BaseModel, frozen=True):
    """A single conversational turn."""

    role: MessageRole
    content: str = Field(min_length=1, description="Text content of the message.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# Aggregate roots


class Conversation(BaseModel):
    """A stateful session comprising multiple messages."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[Message] = Field(default_factory=list)
    status: ConversationStatus = ConversationStatus.active
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


# Request / Response DTOs


class ChatRequest(BaseModel):
    """Incoming payload for a chat completion request."""

    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Client-supplied or auto-generated session identifier.",
    )
    message: str = Field(min_length=1, description="User message text.")
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt to prepend to the conversation.",
    )
    conversation_history: list[Message] = Field(
        default_factory=list,
        description="Prior messages to include for multi-turn context.",
    )


class ChatResponse(BaseModel):
    """Outgoing payload returned from a chat completion request."""

    session_id: str
    message: Message
    token_usage: TokenUsage
    latency: LatencyMetrics
    model: str = Field(description="Model tag that produced this response.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# Operational DTOs


class HealthResponse(BaseModel):
    """Shallow health-check response."""

    status: str = Field(default="ok")
    app_name: str
    app_version: str
    ollama_reachable: bool
    mlflow_reachable: bool
    qdrant_reachable: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# RAG Domain Models


class JournalMetadata(BaseModel):
    """Structured metadata extracted from a journal entry."""

    topics: list[str] = Field(
        default_factory=list, description="Main topics discussed in the entry"
    )
    people: list[str] = Field(default_factory=list, description="Names of people mentioned")
    places: list[str] = Field(
        default_factory=list, description="Names of places or locations mentioned"
    )
    sentiment: str = Field(
        default="Neutral",
        description="Overall sentiment of the entry (e.g. Joy, Sorrow, Anger, Peace)",
    )
    emotions: list[str] = Field(default_factory=list, description="Specific emotions expressed")


class MetadataFilters(BaseModel):
    """Filters extracted from a user query."""

    topics: list[str] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    places: list[str] = Field(default_factory=list)
    sentiment: str | None = None
    emotions: list[str] = Field(default_factory=list)


class TranslatedQuery(BaseModel):
    """The result of query translation."""

    search_query: str = Field(description="The optimized semantic search query")
    filters: MetadataFilters = Field(
        description="Explicit metadata filters extracted from the user's request"
    )


class Document(BaseModel):
    """Represents a raw markdown document."""

    id: str = Field(description="Unique identifier for the document (e.g. filename).")
    content: str = Field(description="The raw text content of the document.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata.")


class Chunk(BaseModel):
    """Represents a chunked piece of a document, ready for embedding."""

    id: str = Field(description="Unique identifier for the chunk.")
    document_id: str = Field(description="ID of the parent document.")
    content: str = Field(description="Text content of this chunk.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata including chunk index."
    )


class RAGQuery(BaseModel):
    """User query for the RAG system."""

    query: str = Field(description="The user's question or prompt.")
    top_k: int = Field(default=3, description="Number of context chunks to retrieve.")


class RAGResponse(BaseModel):
    """Response from the RAG system."""

    answer: str = Field(description="The generated answer.")
    source_chunks: list[Chunk] = Field(
        default_factory=list, description="The chunks used as context."
    )
    token_usage: TokenUsage | None = None
    latency: LatencyMetrics | None = None
