"""Domain-level enumerations.

Using ``str`` as a mixin makes values JSON-serialisable out of the box
and ensures Pydantic v2 serialises them as their string value rather than
the enum member itself.
"""

from __future__ import annotations

from enum import Enum


class MessageRole(str, Enum):
    """Role of a participant in a conversation turn."""

    user = "user"
    assistant = "assistant"
    system = "system"


class ConversationStatus(str, Enum):
    """Lifecycle state of a conversation."""

    active = "active"
    completed = "completed"
    failed = "failed"
