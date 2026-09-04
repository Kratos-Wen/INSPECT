"""Trace-guided active observation for robot procedural verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

from .types import ActiveObservationDecision, ProceduralEvidenceGraph, VerificationResult, ViewCandidate


def _load_json_or_jsonl(path: Path) -> List[dict]:
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text.startswith("["):
        return [dict(item) for item in json.loads(text)]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_view_candidates(path: Path) -> List[ViewCandidate]:
    return [ViewCandidate.from_dict(item) for item in _load_json_or_jsonl(Path(path))]


def write_active_observation_jsonl(results: Iterable[ActiveObservationDecision], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")


def _state_required_evidence(graph: ProceduralEvidenceGraph, state_id: str) -> Set[str]:
    keys: Set[str] = set()
    for edge in graph.verified_by_edges(state_id):
        if edge.target.startswith("evidence:"):
            keys.add(edge.target.split(":", 1)[1])
    return keys


def choose_view_for_missing_evidence(
    verification: VerificationResult,
    candidates: Sequence[ViewCandidate],
    graph: ProceduralEvidenceGraph,
    motion_weight: float = 0.20,
    occlusion_weight: float = 0.30,
    state_context_weight: float = 0.35,
) -> ActiveObservationDecision:
    """Select the viewpoint expected to cover the missing procedural evidence."""

    missing = set(verification.missing_evidence)
    required = _state_required_evidence(graph, verification.predicted_state)
    target = missing or (required - set(verification.observed_evidence))
    candidate_scores: Dict[str, float] = {}
    best_candidate: ViewCandidate | None = None
    best_utility = float("-inf")
    best_seen: Set[str] = set()

    for candidate in candidates:
        visible = set(candidate.visible_evidence)
        missing_hits = visible & target
        required_hits = visible & required
        target_score = len(missing_hits) / max(1, len(target))
        context_score = len(required_hits) / max(1, len(required))
        utility = (
            target_score
            + state_context_weight * context_score
            + float(candidate.prior)
            - motion_weight * float(candidate.motion_cost)
            - occlusion_weight * float(candidate.occlusion_risk)
        )
        candidate_scores[candidate.view_id] = float(utility)
        if utility > best_utility:
            best_utility = utility
            best_candidate = candidate
            best_seen = visible

    if best_candidate is None:
        return ActiveObservationDecision(
            observation_id=verification.observation_id,
            selected_view="",
            utility=0.0,
            expected_observed_evidence=[],
            unresolved_evidence=sorted(target),
            candidate_scores={},
            metadata={"reason": "no_view_candidates"},
        )

    expected = sorted(best_seen & (target | required))
    unresolved = sorted(target - best_seen)
    return ActiveObservationDecision(
        observation_id=verification.observation_id,
        selected_view=best_candidate.view_id,
        utility=float(best_utility),
        expected_observed_evidence=expected,
        unresolved_evidence=unresolved,
        candidate_scores=candidate_scores,
        metadata={
            "predicted_state": verification.predicted_state,
            "recommended_action": verification.recommended_action,
            "target_evidence": sorted(target),
        },
    )


def choose_views_for_results(
    results: Sequence[VerificationResult],
    candidates: Sequence[ViewCandidate],
    graph: ProceduralEvidenceGraph,
) -> List[ActiveObservationDecision]:
    return [choose_view_for_missing_evidence(result, candidates, graph) for result in results]
