"""Contextual assistant for live scene-aware Q&A."""

from .engine import ContextualAssistant
from .grounded_llm import GroundedLLMResponder
from .types import AssistantReply, AssistantSnapshot

__all__ = ["AssistantReply", "AssistantSnapshot", "ContextualAssistant", "GroundedLLMResponder"]
