"""Resolve parsed queries against scene context and knowledge-base entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from ..components.kb import KnowledgeBase
from .types import AssistantSnapshot, ParsedAssistantQuery, ResolvedAssistantQuery
from .utils import overlap_score, tokenize


@dataclass
class DialogueState:
    """Minimal dialogue memory used for pronoun resolution."""

    last_component: str = ""
    last_intent: str = ""


def resolve_query(
    parsed: ParsedAssistantQuery,
    snapshot: AssistantSnapshot,
    kb: KnowledgeBase,
    state: DialogueState,
) -> ResolvedAssistantQuery:
    """Resolve component and step references against the current scene and KB."""

    component_scoped_intents = {
        "object_presence",
        "object_count",
        "object_relation",
        "safety",
        "troubleshooting",
        "component_info",
    }
    target_component = ""
    if parsed.intent in component_scoped_intents:
        target_component = _resolve_component(parsed.text, parsed.tokens, snapshot, kb, state)
    used_history = bool(target_component and state.last_component and target_component == state.last_component)
    target_step = _resolve_step(parsed.target_step_hint, snapshot)

    clarification = ""
    can_answer = True
    confidence = 0.85
    if parsed.intent in {"object_count", "object_relation", "safety", "troubleshooting", "component_info"} and not target_component:
        can_answer = False
        clarification = _clarify_component(snapshot)
        confidence = 0.25
    elif parsed.intent == "unknown":
        can_answer = False
        clarification = "Ask about the current step, next step, visible parts, counts, relations, safety, or troubleshooting."
        confidence = 0.10
    elif parsed.intent == "why_not_progressing":
        confidence = 0.65
    elif parsed.intent in {"current_step", "next_step"} and snapshot.step_confidence < 0.45:
        confidence = 0.45

    return ResolvedAssistantQuery(
        text=parsed.text,
        intent=parsed.intent,
        tokens=list(parsed.tokens),
        target_component=target_component,
        target_step=target_step,
        relation_type=parsed.relation_type,
        asks_reason=parsed.asks_reason,
        clarification=clarification,
        can_answer=can_answer,
        confidence=confidence,
        used_history=used_history,
    )


def update_dialogue_state(state: DialogueState, resolved: ResolvedAssistantQuery) -> None:
    """Update the lightweight dialogue state from the latest resolved query."""

    if resolved.target_component:
        state.last_component = resolved.target_component
    if resolved.intent:
        state.last_intent = resolved.intent


def _resolve_component(
    text: str,
    query_tokens: Iterable[str],
    snapshot: AssistantSnapshot,
    kb: KnowledgeBase,
    state: DialogueState,
) -> str:
    lowered = str(text).lower()
    visible_preferred = list(_unique(snapshot.relevant_objects + snapshot.visible_objects))
    for name in visible_preferred:
        display = kb.component_display_name(name).lower()
        if name in lowered or display in lowered:
            return name

    best_name = _best_component_match(query_tokens, visible_preferred, kb)
    if best_name:
        return best_name

    if any(token in lowered for token in ("this part", "that part", "this one", "that one", "it")):
        if len(snapshot.relevant_objects) == 1:
            return str(snapshot.relevant_objects[0]).strip().lower()
        if state.last_component:
            return state.last_component

    best_global = _best_component_match(query_tokens, kb.component_names(), kb, threshold=0.55)
    return best_global


def _best_component_match(
    query_tokens: Iterable[str],
    candidates: Iterable[str],
    kb: KnowledgeBase,
    threshold: float = 0.35,
) -> str:
    best_name = ""
    best_score = 0.0
    query = list(query_tokens)
    for name in candidates:
        candidate_tokens = tokenize(name) + tokenize(kb.component_display_name(name))
        score = overlap_score(query, candidate_tokens)
        if score > best_score:
            best_name = str(name).strip().lower()
            best_score = score
    return best_name if best_score >= threshold else ""


def _resolve_step(step_hint: str, snapshot: AssistantSnapshot) -> str:
    if step_hint:
        return str(step_hint).strip().upper()
    return str(snapshot.step_id).strip().upper()


def _clarify_component(snapshot: AssistantSnapshot) -> str:
    visible = list(_unique(snapshot.relevant_objects or snapshot.visible_objects))
    if len(visible) == 1:
        label = visible[0]
        return f"Do you mean {label}?"
    if visible:
        shortlist = ", ".join(visible[:3])
        return f"Which part do you mean? I currently see {shortlist}."
    return "I need a specific visible part to answer that."


def _unique(items: List[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in items:
        name = str(item).strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered
