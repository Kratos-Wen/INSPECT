"""Lightweight intent and slot parsing for the grounded assistant."""

from __future__ import annotations

import re

from .types import ParsedAssistantQuery
from .utils import contains_any, tokenize

RELATION_KEYWORDS = (
    "contact",
    "touch",
    "touching",
    "inserted",
    "insert",
    "inside",
    "slot",
    "seated",
    "seat",
    "alignment",
    "aligned",
    "containment",
    "gap",
    "support",
    "supported",
    "relation",
    "front",
    "behind",
    "left",
    "right",
    "above",
    "below",
    "overlap",
)


def parse_query(text: str) -> ParsedAssistantQuery:
    """Parse a natural-language question into a lightweight structured form."""

    normalized = str(text).strip()
    lowered = normalized.lower()
    tokens = tokenize(lowered)

    intent = "unknown"
    if contains_any(lowered, ("what can you do", "help", "how can you help", "what do you know")):
        intent = "capability"
    elif contains_any(
        lowered,
        (
            "previous step",
            "last step",
            "step before",
            "what changed",
            "did the step change",
            "earlier step",
            "what was before",
        ),
    ):
        intent = "history_step"
    elif contains_any(
        lowered,
        (
            "what did i correct",
            "what feedback did i give",
            "what did i accept",
            "last feedback",
            "last correction",
            "did i correct",
            "did i accept",
        ),
    ):
        intent = "history_feedback"
    elif contains_any(
        lowered,
        (
            "have we seen this before",
            "have you seen this before",
            "what does memory say",
            "what does memory recall",
            "what did memory retrieve",
            "what did memory recall",
            "is this similar to before",
            "similar case",
            "do we have a similar previous case",
        ),
    ):
        intent = "memory_context"
    elif contains_any(lowered, ("what step", "current step", "which step", "status", "where are we")):
        intent = "current_step"
    elif contains_any(lowered, ("next step", "what next", "what should i do next", "what do i do next")) or re.search(r"\bnext\b", lowered):
        intent = "next_step"
    elif contains_any(
        lowered,
        (
            "why not progressing",
            "why not progress",
            "why is it not progressing",
            "why are we stuck",
            "why stuck",
            "why blocked",
            "what evidence is missing",
            "what evidence should i inspect",
            "missing evidence",
            "what is missing",
            "why not",
            "can i proceed",
            "can i continue",
            "safe to continue",
            "sufficient to continue",
            "current view sufficient",
            "view sufficient",
            "enough evidence",
            "what should i inspect",
            "where should i look",
            "which evidence",
        ),
    ):
        intent = "why_not_progressing"
    elif contains_any(lowered, ("orientation", "orient")):
        intent = "component_info"
    elif contains_any(lowered, ("correct gear", "right gear", "wrong gear", "correct part", "right part", "wrong part")):
        intent = "object_presence"
    elif contains_any(lowered, ("pending claim", "claim is still pending", "still pending", "active claim", "unresolved claim", "insufficient claim")):
        intent = "why_not_progressing"
    elif any(keyword in lowered for keyword in RELATION_KEYWORDS):
        intent = "object_relation"
    elif contains_any(lowered, ("how many", "count", "number of")) and not contains_any(lowered, ("part number", "part no")):
        intent = "object_count"
    elif contains_any(lowered, ("do you see", "can you see", "is there", "are there", "visible", "what do you see", "which parts", "what parts")):
        intent = "object_presence"
    elif contains_any(lowered, ("safety", "safe")):
        intent = "safety"
    elif contains_any(
        lowered,
        (
            "fault",
            "problem",
            "issue",
            "wrong",
            "loose",
            "stuck",
            "not working",
            "won't fit",
            "cannot close",
            "can't close",
            "unable to close",
            "not close",
            "cannot fit",
            "does not rotate",
            "do not rotate",
            "not rotate",
            "rotate smoothly",
            "smoothly",
            "jammed",
        ),
    ):
        intent = "troubleshooting"
    elif contains_any(
        lowered,
        (
            "what is this",
            "what part is this",
            "which part is this",
            "tell me about",
            "part number",
            "part no",
            "feature",
            "material",
            "marking",
            "marked",
            "color",
            "compatible",
            "parts list",
            "assembly step",
            "ordered step",
            "order of",
            "orientation",
            "orient",
            "tool",
            "equipment",
            "workspace",
            "prepare",
            "storage",
            "store",
            "maintenance",
            "maintain",
            "care",
        ),
    ):
        intent = "component_info"

    target_step_hint = ""
    direct_step = re.search(r"\bs\s*([0-9]+)\b", lowered)
    if direct_step:
        target_step_hint = f"S{direct_step.group(1)}"

    relation_type = ""
    relation_map = {
        "contacting": ("contact", "touch", "touching"),
        "supporting": ("support", "supporting"),
        "supported_by": ("supported", "supported by"),
        "in_front_of": ("front", "in front"),
        "behind": ("behind",),
        "left_of": ("left",),
        "right_of": ("right",),
        "above": ("above", "on top"),
        "below": ("below", "under"),
        "overlapping": ("overlap", "overlapping"),
    }
    for label, hints in relation_map.items():
        if contains_any(lowered, hints):
            relation_type = label
            break

    asks_reason = bool(re.search(r"\bwhy\b|\bhow come\b", lowered))
    asks_next = bool(intent == "next_step" or re.search(r"\bnext\b", lowered))

    return ParsedAssistantQuery(
        text=normalized,
        intent=intent,
        tokens=tokens,
        target_component_hint="",
        target_step_hint=target_step_hint,
        relation_type=relation_type,
        asks_reason=asks_reason,
        asks_next=asks_next,
    )
