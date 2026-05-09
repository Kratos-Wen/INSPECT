"""Rule-based workflow expert."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

from ..types import Detection, StepPrediction
from .kb import KnowledgeBase


def _counts(detections: List[Detection]) -> Counter[str]:
    return Counter(detection.name.lower() for detection in detections)


def _score_one_require(requirement: Dict[str, object], counts: Counter[str]) -> Tuple[float, Dict[str, float]]:
    all_of = {str(key).strip().lower(): int(value) for key, value in (requirement.get("all_of", {}) or {}).items()}
    any_of = {str(key).strip().lower(): int(value) for key, value in (requirement.get("any_of", {}) or {}).items()}
    forbid = [str(item).strip().lower() for item in (requirement.get("forbid", []) or [])]

    coverage: List[float] = []
    for key, required_count in all_of.items():
        observed = int(counts.get(key, 0))
        coverage.append(min(1.0, observed / float(max(1, required_count))))
    all_score = sum(coverage) / max(1, len(coverage)) if coverage else 1.0

    any_score = 1.0
    if any_of:
        any_score = 1.0 if any(int(counts.get(key, 0)) >= target for key, target in any_of.items()) else 0.0

    forbid_hit = any(int(counts.get(key, 0)) > 0 for key in forbid)
    forbid_penalty = 0.5 if forbid_hit else 0.0
    score = max(0.0, min(1.0, 0.6 * all_score + 0.4 * any_score - forbid_penalty))
    explanation = {
        "all_score": all_score,
        "any_score": any_score,
        "forbid_penalty": forbid_penalty,
    }
    return score, explanation


def _best_score(requirement: Dict[str, object], counts: Counter[str]) -> Tuple[float, Dict[str, float]]:
    variants = requirement.get("variants", None)
    if variants and isinstance(variants, list):
        best_score = -1.0
        best_explanation: Dict[str, float] = {}
        for variant in variants:
            score, explanation = _score_one_require(variant, counts)
            if score > best_score:
                best_score = score
                best_explanation = explanation
        return max(0.0, best_score), best_explanation
    return _score_one_require(requirement, counts)


class RuleBasedStepExpert:
    """Workflow-compatible rule expert that scores all steps."""

    def __init__(self, kb: KnowledgeBase, steps: List[str]) -> None:
        self.kb = kb
        self.steps = [str(step).strip().upper() for step in steps]

    def predict(self, payload: object) -> StepPrediction:
        """Score steps from a list of canonicalized detections."""

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
            score, explanation = _best_score(self.kb.requirements_for(step_id), counts)
            scores[step_id] = float(score)
            explanations[step_id] = explanation

        top_step = max(scores, key=scores.get) if scores else self.steps[0]
        return StepPrediction(
            step_id=top_step,
            confidence=float(scores.get(top_step, 0.0)),
            scores=scores,
            extras={
                "counts": dict(counts),
                "explanations": explanations,
            },
        )
