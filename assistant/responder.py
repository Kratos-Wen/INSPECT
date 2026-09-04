"""Surface realization for grounded assistant responses."""

from __future__ import annotations

from typing import List

from .types import AssistantEvidence, AssistantSnapshot, ResolvedAssistantQuery


def render_answer(
    resolved: ResolvedAssistantQuery,
    evidence: AssistantEvidence,
    snapshot: AssistantSnapshot,
) -> str:
    """Render a concise, question-focused answer from structured evidence."""

    intent = resolved.intent
    facts = evidence.facts
    if intent == "current_step":
        step_id = str(facts.get("step_id", snapshot.step_id))
        confidence = float(facts.get("step_confidence", snapshot.step_confidence))
        proposed_step = str(facts.get("proposed_step", snapshot.proposed_step)).strip().upper()
        claim_state = str(facts.get("claim_state", snapshot.claim_state)).strip().lower()
        active_claim = _claim_label(str(facts.get("active_claim", snapshot.active_claim)))
        if step_id in {"", "HOLD"}:
            if proposed_step:
                return (
                    f"No procedural step is committed yet. {proposed_step} is only a proposal; "
                    f"the claim {active_claim or 'for that step'} is {claim_state}."
                )
            return "No procedural step is committed yet because the current evidence is insufficient."
        if confidence < 0.55:
            answer = f"The last committed step is {step_id}; the current proposal confidence is low at {confidence:.2f}."
        else:
            answer = f"The last committed step is {step_id}."
        if proposed_step and proposed_step != step_id:
            answer += f" {proposed_step} remains a {claim_state} proposal."
        return answer

    if intent == "history_step":
        recent_steps = list(facts.get("recent_steps", []) or [])
        if not recent_steps:
            return "I do not have enough confirmed step history yet."
        current_step = str(facts.get("current_step", snapshot.step_id)).strip().upper()
        ordered = sorted(recent_steps, key=lambda item: getattr(item, "frame_index", 0))
        if len(ordered) == 1:
            item = ordered[-1]
            return f"The most recent confirmed step is {item.step_id} from frame {item.frame_index}."
        latest = ordered[-1]
        previous = ordered[-2]
        if latest.step_id != previous.step_id:
            return (
                f"The last confirmed step before {current_step} was {previous.step_id}. "
                f"It changed to {latest.step_id} at frame {latest.frame_index} via {str(latest.source).replace('_', ' ')}."
            )
        trail = " -> ".join(item.step_id for item in ordered[-3:])
        return f"The recent confirmed step history is {trail}."

    if intent == "history_feedback":
        recent_feedback = list(facts.get("recent_feedback", []) or [])
        if not recent_feedback:
            return "I do not have any recent feedback events to report."
        latest = sorted(recent_feedback, key=lambda item: getattr(item, "frame_index", 0))[-1]
        action = "accepted" if bool(getattr(latest, "accepted", False)) else "corrected"
        label = str(getattr(latest, "label", "")).strip().upper() or "the step"
        source = str(getattr(latest, "source", "")).replace("_", " ").strip() or "feedback"
        note = str(getattr(latest, "note", "")).strip()
        answer = f"The latest feedback {action} {label} at frame {latest.frame_index} via {source}."
        if note:
            answer += f" Note: {note}."
        return answer

    if intent == "memory_context":
        matches = list(facts.get("memory_matches", []) or [])
        if not matches:
            return "I do not have a strong recalled historical match for the current scene."
        memory_step = str(facts.get("memory_step", snapshot.memory_step)).strip().upper()
        memory_conf = float(facts.get("memory_confidence", snapshot.memory_confidence))
        top = matches[0]
        store = str(getattr(top, "store", "")).replace("_", " ").strip() or "memory"
        source = str(getattr(top, "source", "")).replace("_", " ").strip() or "history"
        score = float(getattr(top, "score", 0.0))
        return (
            f"Memory currently favors {memory_step} with confidence {memory_conf:.2f}. "
            f"The strongest recalled match comes from {store} ({source}) with score {score:.2f}."
        )

    if intent == "next_step":
        current_step = str(facts.get("current_step", snapshot.step_id))
        next_step = str(facts.get("next_step", current_step))
        missing = [str(item) for item in facts.get("missing_for_next", []) if str(item).strip()]
        claim_state = str(facts.get("claim_state", snapshot.claim_state)).strip().lower()
        active_claim = _claim_label(str(facts.get("active_claim", snapshot.active_claim)))
        missing_roles = _role_labels(facts.get("missing_evidence_roles", snapshot.missing_evidence_roles))
        if claim_state == "contradicted":
            return (
                f"I cannot advance to {next_step}: the active claim {active_claim or 'for the proposed step'} "
                "is contradicted by the current evidence. Correct the assembly before continuing."
            )
        if claim_state == "insufficient" and snapshot.active_claim:
            detail = ", ".join(missing_roles[:3])
            if bool(facts.get("external_observation_recommended", snapshot.external_observation_recommended)):
                request = f"show {detail}" if detail else "show the claim-critical relation"
                return (
                    f"The next expected step is {next_step}, but I cannot verify the active claim yet. "
                    f"A different viewpoint is needed to {request}."
                )
            pending = f" for {detail}" if detail else ""
            return (
                f"The next expected step is {next_step}, but I cannot verify the active claim yet. "
                f"I am checking the current observation{pending}."
            )
        if next_step == current_step:
            return f"The workflow is currently at {current_step}; no later step is defined."
        if missing:
            return f"The next expected step after {current_step} is {next_step}. I still need {', '.join(missing[:3])} for that transition."
        return f"The next expected step after {current_step} is {next_step}."

    if intent == "object_presence":
        target = str(facts.get("target_component", "")).strip()
        label = str(facts.get("display_name", target)).strip() or target
        if target:
            return f"Yes, I can see {label}." if bool(facts.get("visible", False)) else f"I do not currently see {label}."
        visible_objects = [str(item) for item in facts.get("visible_display_names", []) if str(item).strip()]
        if not visible_objects:
            return "I do not currently see any recognized parts."
        return "I can currently see " + ", ".join(visible_objects[:4]) + "."

    if intent == "object_count":
        target = str(facts.get("target_component", "")).strip()
        count = int(facts.get("count", 0))
        if not target:
            names = [str(item) for item in facts.get("visible_display_names", []) if str(item).strip()]
            if not names:
                return "I do not currently see any recognized objects."
            distinct = ", ".join(names[:4])
            duplicate_names = [
                str(name).replace("_", " ")
                for name, value in (facts.get("object_counts", {}) or {}).items()
                if int(value) > 1
            ]
            duplicate_text = " No duplicate components are visible." if not duplicate_names else f" Duplicates: {', '.join(duplicate_names[:3])}."
            return f"I currently see {count} recognized object{'s' if count != 1 else ''}: {distinct}.{duplicate_text}"
        label = str(facts.get("display_name", target)).strip() or target
        noun = label or "that part"
        return f"I currently see {count} instance{'s' if count != 1 else ''} of {noun}."

    if intent == "object_relation":
        relations = list(facts.get("relations_display", []) or facts.get("relations", []) or [])
        if not relations:
            return "I do not have a reliable relation to report from the current view."
        first = relations[0]
        subject = str(first.get("subject", "")).strip()
        predicate = str(first.get("predicate", "")).strip().replace("_", " ")
        obj = str(first.get("object", "")).strip()
        return f"I can see {subject} {predicate} {obj}."

    if intent == "component_info":
        label = str(facts.get("display_name", resolved.target_component)).strip() or resolved.target_component
        part_no = str(facts.get("part_no", "")).strip()
        features = [str(item) for item in facts.get("features", []) if str(item).strip()]
        parts: List[str] = [f"{label} is the referenced part."]
        if part_no:
            parts.append(f"Part number: {part_no}.")
        if features:
            parts.append(f"Key feature: {features[0]}.")
        return " ".join(parts[:3])

    if intent == "safety":
        label = str(facts.get("display_name", resolved.target_component)).strip() or resolved.target_component
        items = [str(item) for item in facts.get("items", []) if str(item).strip()]
        return f"Safety for {label}: {'; '.join(_clean_fragment(item) for item in items[:2])}."

    if intent == "troubleshooting":
        label = str(facts.get("display_name", resolved.target_component)).strip() or resolved.target_component
        problems = list(facts.get("problems", []) or [])
        if not problems:
            return f"I do not have troubleshooting notes for {label}."
        first = problems[0]
        if isinstance(first, dict):
            problem = str(first.get("Problem", "")).strip()
            solution = str(first.get("Solution", "")).strip()
            if solution:
                return f"A common issue with {label} is {_clean_fragment(problem)}. Try this: {_clean_fragment(solution)}."
            return f"A common issue with {label} is {_clean_fragment(problem)}."
        return f"Troubleshooting for {label}: {_clean_fragment(str(first).strip())}."

    if intent == "why_not_progressing":
        reasons = list(facts.get("reasons", []) or [])
        active_claim = _claim_label(str(facts.get("active_claim", snapshot.active_claim)))
        missing_roles = _role_labels(facts.get("missing_evidence_roles", snapshot.missing_evidence_roles))
        if "claim_inadmissible" in reasons:
            return f"The proposed claim {active_claim or ''} violates verified procedural history, so the transition is blocked."
        if "active_claim_contradicted" in reasons:
            return f"The workflow is blocked because the active claim {active_claim or ''} is contradicted by current evidence."
        if "active_claim_insufficient" in reasons:
            if bool(facts.get("external_observation_recommended", snapshot.external_observation_recommended)):
                if missing_roles:
                    return (
                        "The current view remains insufficient after targeted evidence processing. "
                        f"A different viewpoint should expose {', '.join(missing_roles[:3])}."
                    )
                return "The current view remains insufficient; a different viewpoint is needed."
            if missing_roles:
                return f"The active claim remains insufficient. I still need {', '.join(missing_roles[:3])}."
            return "The active claim remains insufficient because the current view does not expose decisive evidence."
        recent_feedback = list(facts.get("recent_feedback", []) or [])
        if recent_feedback:
            latest = sorted(recent_feedback, key=lambda item: getattr(item, "frame_index", 0))[-1]
            label = str(getattr(latest, "label", "")).strip().upper() or "the step"
            note = str(getattr(latest, "note", "")).strip()
            if note:
                return f"The workflow is being conservative after recent feedback on {label}. Evidence must be confirmed before proceeding: {note}."
            return f"The workflow is being conservative after recent feedback on {label}; I need confirming evidence before proceeding."
        if not reasons:
            return "I do not see a strong reason for the workflow to be blocked right now."
        if "no_visible_parts" in reasons:
            return "The workflow is not progressing because I do not have enough visible parts in view to confirm the next state."
        if "missing_required_parts" in reasons:
            missing = [str(item) for item in facts.get("missing_for_next", []) if str(item).strip()]
            if missing:
                return f"The workflow is likely blocked because I cannot confirm the required parts for the next step: {', '.join(missing[:3])}."
        if "review:request_human" in reasons or "review:hold" in reasons:
            reason = str(facts.get("review_reason", "")).strip()
            return f"The system is waiting because the review layer flagged this state as uncertain{': ' + reason if reason else ''}."
        if "low_step_confidence" in reasons:
            return f"The workflow is not progressing because the current step estimate is too uncertain at {float(facts.get('step_confidence', 0.0)):.2f}."
        if "memory_step_disagrees" in reasons:
            return "The workflow is being held because recent memory evidence disagrees with the current step estimate."
        return "The workflow is being conservative because the current evidence is not strong enough to confirm a transition."

    if intent == "capability":
        capabilities = [str(item) for item in facts.get("capabilities", []) if str(item).strip()]
        return "I can answer questions about " + ", ".join(capabilities[:6]) + "."

    return "I am not sure how to answer that reliably."


def _clean_fragment(text: str) -> str:
    return str(text).strip().rstrip(".")


def _claim_label(claim_id: str) -> str:
    return str(claim_id or "").strip().replace("_", " ")


def _role_labels(values) -> List[str]:
    labels = {
        "alignment": "alignment evidence",
        "boundary_visibility": "a clear boundary view",
        "claim_disambiguation": "claim-disambiguating evidence",
        "contact_verification": "contact evidence",
        "containment": "containment evidence",
        "identity_disambiguation": "identity evidence",
        "insertion": "insertion evidence",
        "object_presence": "object-presence evidence",
        "occlusion_recovery": "an unoccluded view",
        "slot_relation": "the slot relation",
    }
    return [labels.get(str(item).strip(), str(item).strip().replace("_", " ")) for item in (values or []) if str(item).strip()]
