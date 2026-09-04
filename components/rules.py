"""Rule-based workflow expert."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Tuple

from ..core_types import Detection, SceneGraphRelation, StepPrediction
from .kb import KnowledgeBase


_SEMANTIC_ROLE = {
    "type_3_gear": "small_gear",
    "type_7_gear": "small_gear",
    "type_2_gear": "big_gear",
    "type_8_gear": "big_gear",
    "type_5_gearbox_cover": "cover",
    "type_6_gearbox_cover": "cover",
    "type_5_gearbox_housing": "housing",
    "type_6_gearbox_housing": "housing",
}


def _role(name: object) -> str:
    key = str(name or "").strip().lower()
    return _SEMANTIC_ROLE.get(key, key)


def _counts(detections: List[Detection]) -> Counter[str]:
    return Counter(detection.name.lower() for detection in detections)


def _role_counts(counts: Counter[str]) -> Counter[str]:
    output: Counter[str] = Counter()
    for name, count in counts.items():
        output[_role(name)] = max(int(output.get(_role(name), 0)), int(count))
    return output


def _relation_facts(values: object) -> set[tuple[str, str, str]]:
    facts: set[tuple[str, str, str]] = set()
    for item in values or []:
        if isinstance(item, SceneGraphRelation):
            facts.add(
                (
                    str(item.subject_name).strip().lower(),
                    str(item.predicate).strip().lower(),
                    str(item.object_name).strip().lower(),
                )
            )
            continue
        if isinstance(item, dict):
            subject = str(item.get("subject", item.get("subject_name", ""))).strip().lower()
            predicate = str(item.get("predicate", "")).strip().lower()
            obj = str(item.get("object", item.get("object_name", ""))).strip().lower()
            if subject and predicate and obj:
                facts.add((subject, predicate, obj))
            continue
        if isinstance(item, (list, tuple)) and len(item) == 3:
            subject, predicate, obj = item
            facts.add((str(subject).strip().lower(), str(predicate).strip().lower(), str(obj).strip().lower()))
    return facts


def _role_relation_facts(facts: set[tuple[str, str, str]]) -> set[tuple[str, str, str]]:
    return {(_role(subject), predicate, _role(obj)) for subject, predicate, obj in facts}


def _compile_expected_relations(values: Iterable[object]) -> list[tuple[str, str, str]]:
    relations: list[tuple[str, str, str]] = []
    for item in values or []:
        if isinstance(item, dict):
            subject = str(item.get("subject", "")).strip().lower()
            predicate = str(item.get("predicate", "")).strip().lower()
            obj = str(item.get("object", "")).strip().lower()
            if subject and predicate and obj:
                relations.append((subject, predicate, obj))
            continue
        if isinstance(item, (list, tuple)) and len(item) == 3:
            subject, predicate, obj = item
            relations.append((str(subject).strip().lower(), str(predicate).strip().lower(), str(obj).strip().lower()))
    return relations


def _predicate_weight(predicate: str) -> float:
    weights = {
        "inside": 1.00,
        "aligned_with": 0.82,
        "contacting": 0.62,
        "overlapping": 0.45,
        "supported_by": 0.42,
        "supporting": 0.42,
        "near": 0.18,
    }
    return float(weights.get(str(predicate).strip().lower(), 0.25))


def _relation_score(expected: list[tuple[str, str, str]], facts: set[tuple[str, str, str]]) -> float:
    if not expected:
        return 0.0
    total = sum(_predicate_weight(predicate) for _, predicate, _ in expected)
    if total <= 1e-6:
        return 0.0
    matched = sum(_predicate_weight(predicate) for subject, predicate, obj in expected if (subject, predicate, obj) in facts)
    return float(max(0.0, min(1.0, matched / total)))


_ASSEMBLY_RELATIONS = {
    "inside",
    "aligned_with",
    "contacting",
    "overlapping",
    "supported_by",
    "supporting",
}


def _assembled_forbid_hit(
    required_roles: set[str],
    forbidden_roles: set[str],
    facts: set[tuple[str, str, str]],
) -> bool:
    """Return whether a forbidden role participates in the active assembly.

    Workflow ``forbid`` entries describe parts that must not yet participate in
    the assembled state. Merely seeing a future or spare part on the workbench
    is not negative evidence for the current step.
    """

    if not required_roles or not forbidden_roles:
        return False
    anchor_roles = {"housing"} if "housing" in required_roles else required_roles
    for subject, predicate, obj in facts:
        if predicate not in _ASSEMBLY_RELATIONS:
            continue
        if (subject in forbidden_roles and obj in anchor_roles) or (
            obj in forbidden_roles and subject in anchor_roles
        ):
            return True
    return False


def _score_one_require(
    requirement: Dict[str, object],
    counts: Counter[str],
    relations: set[tuple[str, str, str]],
) -> Tuple[float, Dict[str, float]]:
    all_source = requirement.get("proposal_all_of", requirement.get("all_of", {}))
    any_source = requirement.get("proposal_any_of", requirement.get("any_of", {}))
    relation_source = requirement.get(
        "proposal_expected_relations",
        requirement.get("expected_relations", []),
    )
    all_of = {
        str(key).strip().lower(): int(value)
        for key, value in (all_source or {}).items()
    }
    any_of = {
        str(key).strip().lower(): int(value)
        for key, value in (any_source or {}).items()
    }
    forbid = [str(item).strip().lower() for item in (requirement.get("forbid", []) or [])]
    expected_relations = _compile_expected_relations(relation_source or [])
    relation_required = bool(requirement.get("relation_required_for_proposal", False))

    role_counts = _role_counts(counts)
    semantic_all_of: Counter[str] = Counter()
    for key, value in (requirement.get("all_of", {}) or {}).items():
        semantic_all_of[_role(key)] = max(
            semantic_all_of[_role(key)],
            int(value),
        )
    role_all_of: Counter[str] = Counter()
    for key, value in all_of.items():
        role_all_of[_role(key)] = max(role_all_of[_role(key)], int(value))
    role_any_of: Counter[str] = Counter()
    for key, value in any_of.items():
        role_any_of[_role(key)] = max(role_any_of[_role(key)], int(value))
    required_roles = set(role_all_of) | set(role_any_of) | set(semantic_all_of)
    role_forbid = {_role(item) for item in forbid} - required_roles
    role_relations = _role_relation_facts(relations)
    role_expected = list(dict.fromkeys(
        (_role(subject), predicate, _role(obj))
        for subject, predicate, obj in expected_relations
    ))

    coverage: List[float] = []
    for key, required_count in role_all_of.items():
        observed = int(role_counts.get(key, 0))
        coverage.append(min(1.0, observed / float(max(1, required_count))))
    all_score = sum(coverage) / max(1, len(coverage)) if coverage else 1.0

    any_score = 1.0
    if role_any_of:
        any_score = 1.0 if any(int(role_counts.get(key, 0)) >= target for key, target in role_any_of.items()) else 0.0

    forbid_hit = _assembled_forbid_hit(required_roles, role_forbid, role_relations)
    forbid_penalty = 0.35 if forbid_hit else 0.0
    rel_score = _relation_score(role_expected, role_relations)
    specificity_bonus = 0.04 * max(0, len(semantic_all_of) - 1) * all_score
    relation_gate_penalty = 0.25 * (1.0 - rel_score) if relation_required else 0.0
    if role_expected:
        score = (
            0.55 * all_score
            + 0.10 * any_score
            + 0.35 * rel_score
            + specificity_bonus
            - forbid_penalty
            - relation_gate_penalty
        )
    else:
        score = 0.45 * all_score + 0.10 * any_score + specificity_bonus - forbid_penalty
    score = max(0.0, min(1.0, score))
    explanation = {
        "all_score": all_score,
        "any_score": any_score,
        "forbid_penalty": forbid_penalty,
        "relation_score": rel_score,
        "specificity_bonus": specificity_bonus,
        "relation_gate_penalty": relation_gate_penalty,
        "relation_required_for_proposal": float(relation_required),
        "num_expected_relations": float(len(expected_relations)),
        "role_level_proposal": 1.0,
        "observable_signature": float(
            "proposal_all_of" in requirement
            or "proposal_any_of" in requirement
            or "proposal_expected_relations" in requirement
        ),
    }
    return score, explanation


def _best_score(
    requirement: Dict[str, object],
    counts: Counter[str],
    relations: set[tuple[str, str, str]],
) -> Tuple[float, Dict[str, float]]:
    variants = requirement.get("variants", None)
    if variants and isinstance(variants, list):
        best_score = -1.0
        best_explanation: Dict[str, float] = {}
        for variant in variants:
            score, explanation = _score_one_require(variant, counts, relations)
            if score > best_score:
                best_score = score
                best_explanation = explanation
        return max(0.0, best_score), best_explanation
    return _score_one_require(requirement, counts, relations)


class RuleBasedStepExpert:
    """Workflow-compatible rule expert that scores all steps."""

    def __init__(self, kb: KnowledgeBase, steps: List[str]) -> None:
        self.kb = kb
        self.steps = [str(step).strip().upper() for step in steps]

    def predict(self, payload: object) -> StepPrediction:
        """Score steps from a list of canonicalized detections."""

        relations: set[tuple[str, str, str]] = set()
        if isinstance(payload, dict):
            detections = list(payload.get("detections") or [])
            relations = _relation_facts(payload.get("relations") or payload.get("relation_facts") or [])
        else:
            detections = list(payload) if isinstance(payload, list) else []
        if not detections:
            scores = {step_id: 0.0 for step_id in self.steps}
            return StepPrediction(
                step_id=self.steps[0],
                confidence=0.0,
                scores=scores,
                extras={"counts": {}, "explanations": {}, "reason": "empty_detections"},
            )
        counts = _counts(detections)
        scores: Dict[str, float] = {}
        explanations: Dict[str, Dict[str, float]] = {}
        for step_id in self.steps:
            score, explanation = _best_score(self.kb.requirements_for(step_id), counts, relations)
            scores[step_id] = float(score)
            explanations[step_id] = explanation

        top_step = max(scores, key=scores.get) if scores else self.steps[0]
        return StepPrediction(
            step_id=top_step,
            confidence=float(scores.get(top_step, 0.0)),
            scores=scores,
            extras={
                "counts": dict(counts),
                "relation_facts": [list(item) for item in sorted(relations)],
                "explanations": explanations,
            },
        )
