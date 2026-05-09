"""Answer policy for the grounded assistant."""

from __future__ import annotations

from .types import AssistantEvidence, AssistantSnapshot, PolicyDecision, ResolvedAssistantQuery


def decide_policy(
    resolved: ResolvedAssistantQuery,
    evidence: AssistantEvidence,
    snapshot: AssistantSnapshot,
    enabled: bool,
) -> PolicyDecision:
    """Decide whether to answer, clarify, or abstain."""

    if not enabled:
        return PolicyDecision(action="abstain", message="The live assistant is disabled.")

    if not resolved.can_answer:
        return PolicyDecision(action="clarify", message=resolved.clarification or "Please clarify your question.")

    if resolved.intent in {"current_step", "next_step"} and snapshot.step_confidence < 0.25:
        return PolicyDecision(action="abstain", message="I cannot answer that reliably because the current step estimate is too uncertain.")

    if resolved.intent in {"object_presence", "object_count", "object_relation", "component_info", "safety", "troubleshooting"} and not snapshot.visible_objects:
        return PolicyDecision(action="abstain", message="I cannot verify that because I do not see any recognized parts right now.")

    if resolved.intent == "object_relation" and not evidence.facts.get("relations"):
        target = str(evidence.facts.get("target_component", "")).strip()
        if target:
            return PolicyDecision(action="abstain", message=f"I do not have a reliable relation for {target} in the current view.")
        return PolicyDecision(action="abstain", message="I do not have a reliable scene relation in the current view.")

    if resolved.intent == "safety" and not evidence.facts.get("items"):
        label = str(evidence.facts.get("display_name", resolved.target_component)).strip() or "that part"
        return PolicyDecision(action="abstain", message=f"I do not have specific safety instructions for {label}.")

    if resolved.intent == "troubleshooting" and not evidence.facts.get("problems"):
        label = str(evidence.facts.get("display_name", resolved.target_component)).strip() or "that part"
        return PolicyDecision(action="abstain", message=f"I do not have troubleshooting notes for {label}.")

    return PolicyDecision(action="answer")
