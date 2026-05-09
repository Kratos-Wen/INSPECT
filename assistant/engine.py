"""Grounded scene-aware assistant for live Q&A without heavy latency."""

from __future__ import annotations

from typing import List

from ..components.kb import KnowledgeBase
from .evidence import select_evidence
from .parser import parse_query
from .policy import decide_policy
from .responder import render_answer
from .resolver import DialogueState, resolve_query, update_dialogue_state
from .types import AssistantReply, AssistantSnapshot


class ContextualAssistant:
    """Answer natural-language questions from current live context with grounded constraints."""

    def __init__(
        self,
        kb: KnowledgeBase,
        enabled: bool = True,
        history_limit: int = 12,
        relation_limit: int = 3,
        overlay_chars: int = 120,
    ) -> None:
        self.kb = kb
        self.enabled = bool(enabled)
        self.history_limit = int(history_limit)
        self.relation_limit = int(relation_limit)
        self.overlay_chars = int(overlay_chars)
        self.history: List[AssistantReply] = []
        self.dialogue_state = DialogueState()

    def answer(self, query: str, snapshot: AssistantSnapshot) -> AssistantReply:
        """Answer one query using a parse-resolve-evidence-policy-response pipeline."""

        normalized = str(query).strip()
        parsed = parse_query(normalized)
        resolved = resolve_query(parsed, snapshot, self.kb, self.dialogue_state)
        evidence = select_evidence(resolved, snapshot, self.kb)
        decision = decide_policy(resolved, evidence, snapshot, enabled=self.enabled)
        if decision.action == "answer":
            answer = render_answer(resolved, evidence, snapshot)
        else:
            answer = decision.message

        reply = AssistantReply(
            query=normalized,
            answer=answer,
            route=resolved.intent,
            status=decision.action,
            evidence={
                "intent": resolved.intent,
                "target_component": resolved.target_component,
                "target_step": resolved.target_step,
                "relation_type": resolved.relation_type,
                "policy": decision.action,
                "grounded": evidence.grounded,
                "facts": dict(evidence.facts),
            },
        )
        self.history.append(reply)
        if len(self.history) > self.history_limit:
            self.history.pop(0)
        if decision.action == "answer":
            update_dialogue_state(self.dialogue_state, resolved)
        return reply

    def latest_overlay_text(self) -> str:
        """Return the most recent answer shortened for the UI overlay."""

        if not self.history:
            return ""
        return self.history[-1].answer[: self.overlay_chars]
