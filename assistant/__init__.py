"""Contextual assistant for live scene-aware Q&A."""

from .engine import ContextualAssistant
from .types import AssistantReply, AssistantSnapshot

__all__ = ["AssistantReply", "AssistantSnapshot", "ContextualAssistant"]
