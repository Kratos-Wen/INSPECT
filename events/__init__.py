"""Structured sparse event logging for governance and observability."""

from .bus import JsonlEventBus
from .types import PipelineEvent

__all__ = ["JsonlEventBus", "PipelineEvent"]
