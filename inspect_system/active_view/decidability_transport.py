"""Structured evidence transport for claim-disambiguating inspection.

The policy transports assistant-derived evidence roles onto a fixed robot
lattice and optimizes expected claim decidability. It never consumes robot
utility labels or candidate-view images.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .active_selector import SelectionResult
from .counterfactual_transport import normalize_counterfactual_family
from .evidence_state import EvidenceItem, EvidenceState, merge_missing_names
from .ontology import normalize_claim, normalize_key
from .view_lattice import (
    ORBIT_ACTIONS,
    SIX_VIEWS,
    ViewNode,
    action_compatibility,
    candidate_views,
    transition_cost,
)


@dataclass(frozen=True)
class EvidenceClause:
    """Alternative visual roles that establish one required distinction."""

    name: str
    roles: tuple[str, ...]
    weight: float = 1.0


GEAR_IDENTITY = EvidenceClause(
    "identity",
    ("identity_disambiguation_view", "claim_disambiguation_view"),
)
GEAR_RELATION = EvidenceClause(
    "spatial_relation",
    (
        "insertion_verification_view",
        "containment_verification_view",
        "slot_relation_view",
        "claim_disambiguation_view",
    ),
)
COVER_SEATING = EvidenceClause(
    "seating_contact",
    (
        "gap_visibility_view",
        "boundary_alignment_view",
        "contact_verification_view",
        "claim_disambiguation_view",
    ),
)
PRESENCE = EvidenceClause(
    "state_absence",
    (
        "identity_disambiguation_view",
        "object_presence_view",
        "occlusion_recovery_view",
        "claim_disambiguation_view",
    ),
)


def typed_claim_id(value: object) -> str:
    """Preserve small/big-gear semantics while exposing a generic parent."""

    key = normalize_key(value)
    aliases = {
        "step2": "small_gear_inserted",
        "step2_small_gear_inserted": "small_gear_inserted",
        "small_gear_inserted": "small_gear_inserted",
        "step3": "big_gear_inserted",
        "step3_big_gear_inserted": "big_gear_inserted",
        "big_gear_inserted": "big_gear_inserted",
        "step4": "cover_seated",
        "step4_cover_seated": "cover_seated",
        "cover_fully_seated": "cover_seated",
        "cover_seated": "cover_seated",
    }
    return aliases.get(key, key or "state_validity")


def parent_claim_id(value: object) -> str:
    typed = typed_claim_id(value)
    if typed in {"small_gear_inserted", "big_gear_inserted"}:
        return "gear_inserted"
    return normalize_claim(typed)


def _typed_event_claim(event: Mapping[str, Any]) -> str:
    metadata = event.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("source_claim_id"):
        return typed_claim_id(metadata["source_claim_id"])
    direct = typed_claim_id(event.get("claim_id", ""))
    if direct not in {"gear_inserted", "state_validity"}:
        return direct
    searchable = " ".join(
        str(event.get(key, "")).lower()
        for key in ("event_id", "video", "source_path")
    )
    if "small_gear" in searchable or "smallgear" in searchable:
        return "small_gear_inserted"
    if "big_gear" in searchable or "biggear" in searchable:
        return "big_gear_inserted"
    return direct


def _event_weight(event: Mapping[str, Any], *, requirement: bool) -> float:
    if requirement and event.get("requirement_weight") not in (None, ""):
        return max(0.0, float(event["requirement_weight"]))
    return math.prod(
        max(0.0, float(event.get(key, default) or 0.0))
        for key, default in (
            ("transfer_weight", 0.0),
            ("label_confidence", 1.0),
            ("evidence_stability", 1.0),
            ("evidence_importance", 1.0),
        )
    )


@dataclass
class TypedTraceBackoffRevealModel:
    """Assistant-only typed posterior with a generic-policy backoff."""

    base_model: Any
    concentration: float = 2.0
    counts: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _key(claim: object, role: object) -> str:
        return f"{typed_claim_id(claim)}|{normalize_key(role) or '__any__'}"

    @classmethod
    def from_report(
        cls,
        base_model: Any,
        report_path: str | Path,
        *,
        concentration: float | None = None,
    ) -> "TypedTraceBackoffRevealModel":
        payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
        if payload.get("uses_robot_view_training"):
            raise RuntimeError("Typed reveal supervision contains robot-view training.")
        counts: Dict[str, Dict[str, float]] = {}
        videos = set()
        accepted = 0
        for event in payload.get("trainable_events") or []:
            if not isinstance(event, Mapping):
                continue
            if event.get("uses_robot_view_training") or event.get(
                "uses_robot_utility_labels"
            ):
                raise RuntimeError("Typed reveal event contains robot supervision.")
            action = normalize_key(event.get("relative_action", ""))
            if action not in ORBIT_ACTIONS or not bool(event.get("transferable", True)):
                continue
            if int(event.get("label", 1)) <= 0:
                continue
            weight = _event_weight(event, requirement=False)
            if weight <= 0.0:
                continue
            claim = _typed_event_claim(event)
            role = normalize_key(event.get("evidence_role", ""))
            for key in (cls._key(claim, role), cls._key(claim, "__any__")):
                bucket = counts.setdefault(key, {})
                bucket[action] = float(bucket.get(action, 0.0)) + weight
            videos.add(str(event.get("video", "")))
            accepted += 1
        alpha = (
            float(concentration)
            if concentration is not None
            else max(1e-6, float(getattr(base_model, "alpha", 2.0)))
        )
        return cls(
            base_model=base_model,
            concentration=alpha,
            counts=counts,
            metadata={
                "model_type": "typed_trace_backoff_reveal",
                "assistant_events": accepted,
                "assistant_videos": len(videos),
                "uses_robot_view_training": False,
                "uses_robot_utility_labels": False,
                "uses_candidate_view_images": False,
            },
        )

    def _posterior(
        self,
        claim_id: str,
        evidence_role: str,
        context: Mapping[str, Any] | None,
    ) -> tuple[Dict[str, float], float]:
        base = self.base_model.action_distribution(
            parent_claim_id(claim_id), evidence_role, context=context
        )
        claim = typed_claim_id(claim_id)
        any_counts = self.counts.get(self._key(claim, "__any__"), {})
        role_counts = self.counts.get(self._key(claim, evidence_role), {})

        def update(
            prior: Mapping[str, float], values: Mapping[str, float]
        ) -> Dict[str, float]:
            support = sum(max(0.0, float(value)) for value in values.values())
            return {
                action: (
                    self.concentration * max(0.0, float(prior.get(action, 0.0)))
                    + max(0.0, float(values.get(action, 0.0)))
                )
                / max(1e-12, self.concentration + support)
                for action in ORBIT_ACTIONS
            }

        claim_posterior = update(base, any_counts) if any_counts else dict(base)
        posterior = update(claim_posterior, role_counts) if role_counts else claim_posterior
        total = sum(max(0.0, value) for value in posterior.values())
        normalized = {
            action: max(0.0, posterior.get(action, 0.0)) / max(1e-12, total)
            for action in ORBIT_ACTIONS
        }
        support = sum(any_counts.values()) + sum(role_counts.values())
        return normalized, support

    def action_distribution(
        self,
        claim_id: str,
        evidence_role: str,
        context: Mapping[str, Any] | None = None,
    ) -> Dict[str, float]:
        return self._posterior(claim_id, evidence_role, context)[0]

    def transport_confidence(
        self,
        claim_id: str,
        evidence_role: str,
        context: Mapping[str, Any] | None = None,
    ) -> float:
        _, support = self._posterior(claim_id, evidence_role, context)
        typed_reliability = support / max(1e-12, support + self.concentration)
        function = getattr(self.base_model, "transport_confidence", None)
        base = (
            float(function(parent_claim_id(claim_id), evidence_role, context=context))
            if callable(function)
            else 1.0
        )
        return _clamp01(base * (0.5 + 0.5 * typed_reliability))

    def candidate_affordance_factor(self, *args: Any, **kwargs: Any) -> float:
        return self._forward_claim_method("candidate_affordance_factor", *args, **kwargs)

    def counterfactual_relevance(self, *args: Any, **kwargs: Any) -> float:
        return self._forward_claim_method("counterfactual_relevance", *args, **kwargs)

    def _forward_claim_method(self, name: str, *args: Any, **kwargs: Any) -> float:
        function = getattr(self.base_model, name, None)
        if not callable(function):
            return 1.0
        positional = list(args)
        if positional:
            positional[0] = parent_claim_id(positional[0])
        elif "claim_id" in kwargs:
            kwargs = {**kwargs, "claim_id": parent_claim_id(kwargs["claim_id"])}
        return float(function(*positional, **kwargs))


def load_typed_requirement_counts(
    report_path: str | Path,
) -> Dict[str, Dict[str, float]]:
    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if payload.get("uses_robot_view_training"):
        raise RuntimeError("Requirement supervision contains robot-view training.")
    counts: Dict[str, Dict[str, float]] = {}
    episode_mass: Dict[str, float] = {}
    accepted: List[tuple[str, str, str, float]] = []
    for index, event in enumerate(payload.get("requirement_events") or []):
        if not isinstance(event, Mapping):
            continue
        if event.get("uses_robot_view_training") or event.get("uses_robot_utility_labels"):
            raise RuntimeError("Requirement event contains robot supervision.")
        claim = _typed_event_claim(event)
        role = normalize_key(event.get("evidence_role", ""))
        weight = _event_weight(event, requirement=True)
        if not claim or not role or weight <= 0.0:
            continue
        episode = "|".join(
            (
                str(event.get("video", "")),
                str(event.get("after_frame", "")),
                claim,
            )
        ) or str(index)
        accepted.append((episode, claim, role, weight))
        episode_mass[episode] = episode_mass.get(episode, 0.0) + weight
    for episode, claim, role, weight in accepted:
        balanced = weight / max(1e-12, episode_mass[episode])
        bucket = counts.setdefault(claim, {})
        bucket[role] = float(bucket.get(role, 0.0)) + balanced
    return counts


def mix_typed_requirement_weights(
    state: EvidenceState,
    parent_counts: Mapping[str, Mapping[str, float]],
    typed_counts: Mapping[str, Mapping[str, float]],
    *,
    beta: float,
    temperature: float,
) -> EvidenceState:
    """Apply a typed posterior without forcing missing-role mass to uniform."""

    if not state.items:
        return state
    typed = typed_claim_id(state.claim_id)
    parent = parent_claim_id(state.claim_id)
    ontology = [max(1e-12, float(item.importance)) for item in state.items]
    ontology = [value ** max(0.0, float(temperature)) for value in ontology]
    ontology_mass = sum(ontology)
    prior = [value / max(1e-12, ontology_mass) for value in ontology]
    parent_values = parent_counts.get(parent) or parent_counts.get("__global__", {})
    typed_values = typed_counts.get(typed, {})

    def posterior(
        base: Sequence[float], values: Mapping[str, float]
    ) -> List[float]:
        mass = sum(max(0.0, float(value)) for value in values.values())
        concentration = max(1e-9, float(beta))
        result = []
        for index, item in enumerate(state.items):
            role = normalize_key(item.evidence_role or item.name)
            result.append(
                (
                    concentration * base[index]
                    + max(0.0, float(values.get(role, 0.0)))
                )
                / max(1e-12, concentration + mass)
            )
        total = sum(result)
        return [value / max(1e-12, total) for value in result]

    parent_posterior = posterior(prior, parent_values) if parent_values else prior
    typed_posterior = posterior(parent_posterior, typed_values) if typed_values else parent_posterior
    items = [
        EvidenceItem(
            name=item.name,
            score=item.score,
            threshold=item.threshold,
            importance=typed_posterior[index],
            evidence_role=item.evidence_role,
        )
        for index, item in enumerate(state.items)
    ]
    return EvidenceState(
        claim_id=typed,
        items=items,
        claim_score=state.claim_score,
        contradiction_score=state.contradiction_score,
        margin=state.margin,
        counterfactual_family=state.counterfactual_family,
        counterfactual_scores=dict(state.counterfactual_scores),
        current_utility_proxy=state.current_utility_proxy,
    )


def active_clauses(state: EvidenceState) -> List[EvidenceClause]:
    claim = typed_claim_id(state.claim_id)
    family = normalize_counterfactual_family(state.counterfactual_family)
    if claim in {"small_gear_inserted", "big_gear_inserted", "gear_inserted"}:
        if family == "identity":
            return [GEAR_IDENTITY]
        if family == "spatial_relation":
            return [GEAR_RELATION]
        if family == "state_absence":
            return [PRESENCE]
        return [GEAR_IDENTITY, GEAR_RELATION]
    if claim == "cover_seated":
        return [COVER_SEATING]
    if claim == "object_identity":
        return [GEAR_IDENTITY]
    return [
        EvidenceClause(
            "claim", tuple(item.evidence_role or item.name for item in state.items)
        )
    ]


def _normalized_evidence(item: EvidenceItem) -> float:
    return _clamp01(float(item.score) / max(1e-9, float(item.threshold)))


def _entropy_confidence(distribution: Mapping[str, float]) -> float:
    values = [
        max(1e-12, float(distribution.get(action, 0.0)))
        for action in ORBIT_ACTIONS
    ]
    total = sum(values)
    values = [value / max(1e-12, total) for value in values]
    entropy = -sum(value * math.log(value) for value in values)
    return _clamp01(1.0 - entropy / math.log(len(ORBIT_ACTIONS)))


def _weighted_disjunction(
    values: Mapping[str, float],
    weights: Mapping[str, float],
    roles: Iterable[str],
) -> float:
    available = [normalize_key(role) for role in roles if normalize_key(role) in values]
    if not available:
        return 0.0
    mass = sum(max(0.0, float(weights.get(role, 0.0))) for role in available)
    normalized = (
        {role: max(0.0, float(weights.get(role, 0.0))) / mass for role in available}
        if mass > 1e-12
        else {role: 1.0 / len(available) for role in available}
    )
    complement = 1.0
    for role in available:
        complement *= max(1e-9, 1.0 - _clamp01(values[role])) ** normalized[role]
    return _clamp01(1.0 - complement)


def claim_decidability(
    state: EvidenceState,
    role_values: Mapping[str, float],
) -> tuple[float, Dict[str, float]]:
    weights = {
        normalize_key(item.evidence_role or item.name): max(0.0, float(item.importance))
        for item in state.items
    }
    clauses = active_clauses(state)
    values = {
        clause.name: _weighted_disjunction(role_values, weights, clause.roles)
        for clause in clauses
    }
    clause_mass = sum(max(0.0, clause.weight) for clause in clauses)
    log_value = sum(
        (max(0.0, clause.weight) / max(1e-12, clause_mass))
        * math.log(max(1e-9, values[clause.name]))
        for clause in clauses
    )
    return _clamp01(math.exp(log_value)), values


@dataclass
class StructuredDecidabilitySelector:
    """Move only for positive conservative claim-decidability gain."""

    model: Any
    views: Mapping[str, ViewNode] = field(default_factory=lambda: dict(SIX_VIEWS))
    lambda_cost: float = 0.05
    tau_view: float = 0.02
    geometry_loss_cap: float = 0.50

    def _candidate_role_values(
        self,
        state: EvidenceState,
        current_view: str,
        context: Mapping[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        candidates = candidate_views(current_view, self.views)
        result = {view: {} for view in candidates}
        ray_factors = dict(
            context.get("preservation_role_factors")
            or context.get("candidate_role_factors")
            or {}
        )
        for item in state.items:
            role = normalize_key(item.evidence_role or item.name)
            current = _normalized_evidence(item)
            item_context = {
                **context,
                "item_score": float(item.score),
                "item_threshold": float(item.threshold),
                "missing_weight": 1.0 - current,
            }
            distribution = self.model.action_distribution(
                state.claim_id, role, context=item_context
            )
            directional_confidence = _entropy_confidence(distribution)
            confidence_fn = getattr(self.model, "transport_confidence", None)
            trace_confidence = (
                _clamp01(confidence_fn(state.claim_id, role, context=item_context))
                if callable(confidence_fn)
                else 1.0
            )
            reliability = trace_confidence * math.sqrt(directional_confidence)
            relevance_fn = getattr(self.model, "counterfactual_relevance", None)
            relevance = (
                _clamp01(relevance_fn(state.claim_id, role, context=item_context))
                if callable(relevance_fn)
                else 1.0
            )
            raw: Dict[str, float] = {}
            geometry: Dict[str, float] = {}
            for view in candidates:
                directional = sum(
                    float(distribution.get(action, 0.0))
                    * action_compatibility(action, current_view, view, self.views)
                    for action in ORBIT_ACTIONS
                )
                factor_fn = getattr(self.model, "candidate_affordance_factor", None)
                factor = (
                    max(
                        0.25,
                        min(
                            4.0,
                            float(
                                factor_fn(
                                    state.claim_id,
                                    role,
                                    current_view,
                                    view,
                                    self.views,
                                    context=item_context,
                                )
                            ),
                        ),
                    )
                    if callable(factor_fn)
                    else 1.0
                )
                current_ray = max(
                    0.05,
                    float(dict(ray_factors.get(current_view) or {}).get(role, 1.0)),
                )
                candidate_ray = max(
                    0.05,
                    float(dict(ray_factors.get(view) or {}).get(role, 1.0)),
                )
                geometry[view] = max(0.25, min(4.0, candidate_ray / current_ray))
                raw[view] = max(0.0, directional) * math.sqrt(factor)
            best = max(raw.values(), default=0.0)
            for view in candidates:
                relative_reveal = raw[view] / max(1e-12, best)
                gain = (1.0 - current) * reliability * relevance * relative_reveal
                predicted = current + gain
                if geometry[view] < 1.0:
                    loss = current * min(
                        self.geometry_loss_cap, 1.0 - geometry[view]
                    )
                    predicted -= loss
                result[view][role] = _clamp01(predicted)
        return result

    def select(
        self,
        *,
        current_view: str,
        evidence_state: EvidenceState,
        visited_views: List[str] | None = None,
        candidate_context: Mapping[str, Any] | None = None,
    ) -> SelectionResult:
        if current_view not in self.views:
            raise KeyError(f"Unknown current view: {current_view}")
        if not evidence_state.items:
            return SelectionResult(
                action="defer",
                current_view=current_view,
                reason="no_structured_evidence",
            )
        current_values = {
            normalize_key(item.evidence_role or item.name): _normalized_evidence(item)
            for item in evidence_state.items
        }
        current_decidability, current_clauses = claim_decidability(
            evidence_state, current_values
        )
        context = {
            **dict(candidate_context or {}),
            "counterfactual_family": evidence_state.counterfactual_family,
            "counterfactual_scores": dict(evidence_state.counterfactual_scores),
            "counterfactual_margin": float(evidence_state.margin),
        }
        projected = self._candidate_role_values(evidence_state, current_view, context)
        visited = set(visited_views or [])
        ranked: List[Dict[str, Any]] = []
        for view, role_values in projected.items():
            if view in visited:
                continue
            decidability, clause_values = claim_decidability(evidence_state, role_values)
            cost = transition_cost(current_view, view, self.views)
            gain = decidability - current_decidability
            ranked.append(
                {
                    "view_id": view,
                    "score": gain - self.lambda_cost * cost,
                    "evidence_gain": gain,
                    "cost": cost,
                    "current_decidability": current_decidability,
                    "projected_decidability": decidability,
                    "current_clauses": current_clauses,
                    "projected_clauses": clause_values,
                    "projected_role_values": role_values,
                }
            )
        ranked.sort(
            key=lambda item: (
                float(item["score"]),
                float(item["projected_decidability"]),
                -float(item["cost"]),
                str(item["view_id"]),
            ),
            reverse=True,
        )
        if not ranked or float(ranked[0]["score"]) <= self.tau_view:
            return SelectionResult(
                action="defer",
                current_view=current_view,
                selected_view=current_view,
                score=float(ranked[0]["score"]) if ranked else 0.0,
                reason="no_positive_decidability_gain",
                missing_evidence=merge_missing_names(evidence_state.missing()),
                ranked_views=ranked,
            )
        return SelectionResult(
            action="move",
            current_view=current_view,
            selected_view=str(ranked[0]["view_id"]),
            score=float(ranked[0]["score"]),
            reason="structured_decidability_gain",
            missing_evidence=merge_missing_names(evidence_state.missing()),
            ranked_views=ranked,
        )


def _clamp01(value: object) -> float:
    return max(0.0, min(1.0, float(value)))
