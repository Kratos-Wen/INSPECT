"""Evidence selection for grounded assistant responses."""

from __future__ import annotations

from typing import Dict, List

from ..components.kb import KnowledgeBase
from .types import AssistantEvidence, AssistantSnapshot, ResolvedAssistantQuery


def select_evidence(
    resolved: ResolvedAssistantQuery,
    snapshot: AssistantSnapshot,
    kb: KnowledgeBase,
) -> AssistantEvidence:
    """Select only the evidence relevant to the resolved query intent."""

    intent = resolved.intent
    if intent == "current_step":
        return AssistantEvidence(
            intent=intent,
            facts={
                "step_id": snapshot.step_id,
                "step_confidence": snapshot.step_confidence,
                "runner_up": snapshot.runner_up,
                "review_action": snapshot.review_action,
                "review_reason": snapshot.review_reason,
                "proposed_step": snapshot.proposed_step,
                "active_claim": snapshot.active_claim,
                "claim_state": snapshot.claim_state,
                "claim_support": snapshot.claim_support,
                "claim_contradiction": snapshot.claim_contradiction,
                "claim_margin": snapshot.claim_margin,
                "missing_evidence_roles": list(snapshot.missing_evidence_roles),
                "acquisition_mode": snapshot.acquisition_mode,
                "external_observation_recommended": snapshot.external_observation_recommended,
            },
        )
    if intent == "history_step":
        return AssistantEvidence(
            intent=intent,
            facts={
                "current_step": snapshot.step_id,
                "recent_steps": list(snapshot.recent_steps),
            },
        )
    if intent == "history_feedback":
        return AssistantEvidence(
            intent=intent,
            facts={
                "recent_feedback": list(snapshot.recent_feedback),
            },
        )
    if intent == "memory_context":
        return AssistantEvidence(
            intent=intent,
            facts={
                "memory_step": snapshot.memory_step,
                "memory_confidence": snapshot.memory_confidence,
                "memory_reason": snapshot.memory_reason,
                "memory_recalled": snapshot.memory_recalled,
                "memory_matches": list(snapshot.memory_matches),
            },
        )
    if intent == "next_step":
        next_step = kb.next_step(snapshot.step_id)
        return AssistantEvidence(
            intent=intent,
            facts={
                "current_step": snapshot.step_id,
                "current_confidence": snapshot.step_confidence,
                "next_step": next_step,
                "missing_for_next": _missing_for_step(next_step, snapshot.object_counts, kb),
                "proposed_step": snapshot.proposed_step,
                "active_claim": snapshot.active_claim,
                "claim_state": snapshot.claim_state,
                "claim_support": snapshot.claim_support,
                "claim_contradiction": snapshot.claim_contradiction,
                "claim_admissible": snapshot.claim_admissible,
                "missing_evidence_roles": list(snapshot.missing_evidence_roles),
                "acquisition_mode": snapshot.acquisition_mode,
                "external_observation_recommended": snapshot.external_observation_recommended,
            },
        )
    if intent == "object_presence":
        target = resolved.target_component
        visible = int(snapshot.object_counts.get(target, 0)) > 0 if target else bool(snapshot.visible_objects)
        visible_objects = [str(item).strip() for item in snapshot.visible_objects if str(item).strip()]
        return AssistantEvidence(
            intent=intent,
            facts={
                "target_component": target,
                "display_name": kb.component_display_name(target) if target else "",
                "visible": visible,
                "object_counts": dict(snapshot.object_counts),
                "visible_objects": visible_objects,
                "visible_display_names": [kb.component_display_name(item) for item in visible_objects],
                "relevant_objects": list(snapshot.relevant_objects),
            },
        )
    if intent == "object_count":
        target = resolved.target_component
        if not target:
            counts = dict(snapshot.object_counts)
            return AssistantEvidence(
                intent=intent,
                facts={
                    "target_component": "",
                    "display_name": "",
                    "count": sum(int(value) for value in counts.values()),
                    "object_counts": counts,
                    "visible_objects": list(snapshot.visible_objects),
                    "visible_display_names": [kb.component_display_name(item) for item in snapshot.visible_objects],
                },
            )
        return AssistantEvidence(
            intent=intent,
            facts={
                "target_component": target,
                "display_name": kb.component_display_name(target) if target else "",
                "count": int(snapshot.object_counts.get(target, 0)),
                "object_counts": dict(snapshot.object_counts),
            },
        )
    if intent == "object_relation":
        relations = _filter_relations(
            snapshot.relation_facts or _fallback_relation_facts(snapshot.scene_relations),
            target_component=resolved.target_component,
            relation_type=resolved.relation_type,
        )
        return AssistantEvidence(
            intent=intent,
            facts={
                "target_component": resolved.target_component,
                "display_name": kb.component_display_name(resolved.target_component) if resolved.target_component else "",
                "relation_type": resolved.relation_type,
                "relations": relations,
                "relations_display": [
                    {
                        "subject": kb.component_display_name(relation.get("subject", "")),
                        "predicate": str(relation.get("predicate", "")).strip().lower(),
                        "object": kb.component_display_name(relation.get("object", "")),
                    }
                    for relation in relations
                ],
            },
        )
    if intent == "component_info":
        record = kb.component_record(resolved.target_component)
        return AssistantEvidence(
            intent=intent,
            facts={
                "target_component": resolved.target_component,
                "display_name": kb.component_display_name(resolved.target_component),
                "visible": int(snapshot.object_counts.get(resolved.target_component, 0)) > 0,
                "part_no": str(record.get("Part No.") or record.get("part_no") or "").strip(),
                "color": str(record.get("Color") or record.get("color") or "").strip(),
                "features": [str(item).strip() for item in (record.get("Key Features") or record.get("features") or []) if str(item).strip()],
                "parts_list": [str(item).strip() for item in (record.get("Parts List") or record.get("parts_list") or []) if str(item).strip()],
                "assembly_steps": [str(item).strip() for item in (record.get("Assembly Steps") or record.get("assembly_steps") or []) if str(item).strip()],
                "tools": [str(item).strip() for item in (record.get("Tools and Equipment") or record.get("tools") or []) if str(item).strip()],
                "maintenance": [str(item).strip() for item in (record.get("Maintenance and Care") or record.get("maintenance") or []) if str(item).strip()],
            },
        )
    if intent == "safety":
        record = kb.component_record(resolved.target_component)
        return AssistantEvidence(
            intent=intent,
            facts={
                "target_component": resolved.target_component,
                "display_name": kb.component_display_name(resolved.target_component),
                "visible": int(snapshot.object_counts.get(resolved.target_component, 0)) > 0,
                "items": [str(item).strip() for item in (record.get("Assembly Safety Instructions") or record.get("safety") or []) if str(item).strip()],
            },
        )
    if intent == "troubleshooting":
        record = kb.component_record(resolved.target_component)
        return AssistantEvidence(
            intent=intent,
            facts={
                "target_component": resolved.target_component,
                "display_name": kb.component_display_name(resolved.target_component),
                "visible": int(snapshot.object_counts.get(resolved.target_component, 0)) > 0,
                "problems": list(record.get("Common Problems and Solutions") or record.get("common_problems") or []),
            },
        )
    if intent == "why_not_progressing":
        next_step = kb.next_step(snapshot.step_id)
        reasons: List[str] = []
        if snapshot.claim_state == "contradicted":
            reasons.append("active_claim_contradicted")
        elif snapshot.claim_state == "insufficient":
            reasons.append("active_claim_insufficient")
        if not snapshot.claim_admissible:
            reasons.append("claim_inadmissible")
        if snapshot.review_action:
            reasons.append(f"review:{snapshot.review_action}")
        if snapshot.step_confidence < 0.55:
            reasons.append("low_step_confidence")
        if not snapshot.visible_objects:
            reasons.append("no_visible_parts")
        if snapshot.memory_step and snapshot.memory_step != snapshot.step_id and snapshot.memory_confidence >= max(0.55, snapshot.step_confidence + 0.05):
            reasons.append("memory_step_disagrees")
        missing = _missing_for_step(next_step, snapshot.object_counts, kb)
        if missing:
            reasons.append("missing_required_parts")
        return AssistantEvidence(
            intent=intent,
            facts={
                "step_id": snapshot.step_id,
                "step_confidence": snapshot.step_confidence,
                "review_action": snapshot.review_action,
                "review_reason": snapshot.review_reason,
                "memory_step": snapshot.memory_step,
                "memory_confidence": snapshot.memory_confidence,
                "visible_objects": list(snapshot.visible_objects),
                "next_step": next_step,
                "missing_for_next": missing,
                "reasons": reasons,
                "recent_steps": list(snapshot.recent_steps),
                "recent_feedback": list(snapshot.recent_feedback),
                "proposed_step": snapshot.proposed_step,
                "active_claim": snapshot.active_claim,
                "claim_state": snapshot.claim_state,
                "claim_support": snapshot.claim_support,
                "claim_contradiction": snapshot.claim_contradiction,
                "claim_margin": snapshot.claim_margin,
                "claim_admissible": snapshot.claim_admissible,
                "missing_evidence_roles": list(snapshot.missing_evidence_roles),
                "acquisition_mode": snapshot.acquisition_mode,
                "external_observation_recommended": snapshot.external_observation_recommended,
            },
        )
    if intent == "capability":
        return AssistantEvidence(
            intent=intent,
            facts={
                "capabilities": [
                    "current step",
                    "next step",
                    "visible parts",
                    "part counts",
                    "scene relations",
                    "component details",
                    "safety guidance",
                    "troubleshooting",
                ]
            },
        )
    return AssistantEvidence(intent=intent, facts={})


def _missing_for_step(step_id: str, object_counts: Dict[str, int], kb: KnowledgeBase) -> List[str]:
    requirements = kb.requirements_for(step_id)
    variants = list(requirements.get("variants") or [])
    if not variants:
        return []
    best_missing: List[str] = []
    best_score = -1.0
    for variant in variants:
        all_of = {str(key).strip().lower(): int(value) for key, value in (variant.get("all_of", {}) or {}).items()}
        matched = sum(1 for name, count in all_of.items() if int(object_counts.get(name, 0)) >= count)
        missing = [name for name, count in all_of.items() if int(object_counts.get(name, 0)) < count]
        score = matched / float(max(1, len(all_of)))
        if score > best_score:
            best_score = score
            best_missing = missing
    return best_missing


def _filter_relations(
    relations: List[Dict[str, str]],
    target_component: str,
    relation_type: str,
) -> List[Dict[str, str]]:
    filtered = []
    for relation in relations:
        subject = str(relation.get("subject", "")).strip().lower()
        predicate = str(relation.get("predicate", "")).strip().lower()
        obj = str(relation.get("object", "")).strip().lower()
        if target_component and target_component not in {subject, obj}:
            continue
        if relation_type and predicate != relation_type:
            continue
        filtered.append(relation)
    return filtered


def _fallback_relation_facts(scene_relations: List[str]) -> List[Dict[str, str]]:
    facts: List[Dict[str, str]] = []
    for item in scene_relations:
        parts = str(item).split(" ", 2)
        if len(parts) != 3:
            continue
        facts.append({"subject": parts[0].strip().lower(), "predicate": parts[1].strip().lower(), "object": parts[2].strip().lower()})
    return facts
