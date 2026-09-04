"""Project learned relative view actions onto the robot six-view lattice."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .counterfactual_transport import counterfactual_role_gate
from .evidence_state import EvidenceItem
from .reveal_model import PriorTableRevealModel
from .view_lattice import ORBIT_ACTIONS, SIX_VIEWS, ViewNode, action_compatibility, candidate_views, transition_cost


@dataclass
class ProjectedViewScore:
    view_id: str
    score: float
    evidence_gain: float
    cost: float
    risk_penalty: float = 0.0
    revisit_penalty: float = 0.0
    action_contributions: Dict[str, float] = field(default_factory=dict)


def relative_role_affordance_factor(
    *,
    candidate_role_factors: Mapping[str, Mapping[str, float]],
    current_view: str,
    candidate_view: str,
    role: str,
) -> float:
    """Return candidate evidence visibility relative to the current view.

    The current-view geometry is an online observation, not a calibrated
    utility label. Normalizing by its own affordance keeps STAY neutral and
    rewards a move only when the same geometric model predicts that the
    candidate exposes more of the requested evidence role.
    """
    current_values = dict(candidate_role_factors.get(current_view) or {})
    candidate_values = dict(candidate_role_factors.get(candidate_view) or {})
    current = max(0.05, float(current_values.get(role, 1.0)))
    candidate = max(0.05, float(candidate_values.get(role, 1.0)))
    return max(0.25, min(4.0, candidate / current))


def score_candidate_view(
    *,
    model: PriorTableRevealModel,
    claim_id: str,
    missing_evidence: Iterable[EvidenceItem],
    current_view: str,
    candidate_view: str,
    views: Mapping[str, ViewNode] | None = None,
    lambda_cost: float = 0.15,
    lambda_risk: float = 0.0,
    lambda_revisit: float = 0.0,
    view_risk: Mapping[str, float] | None = None,
    visited_views: Iterable[str] | None = None,
    counterfactual_context: Mapping[str, Any] | None = None,
) -> ProjectedViewScore:
    table = views or SIX_VIEWS
    gain = 0.0
    contributions: Dict[str, float] = {action: 0.0 for action in ORBIT_ACTIONS}
    items = list(missing_evidence)
    relevance_fn = getattr(model, "counterfactual_relevance", None)
    explicit_counterfactual = str(
        (counterfactual_context or {}).get("counterfactual_family", "")
    ).strip()

    def role_relevance(item: EvidenceItem) -> float:
        learned = (
            float(
                relevance_fn(
                    claim_id,
                    item.evidence_role or item.name,
                    context=dict(counterfactual_context or {}),
                )
            )
            if callable(relevance_fn)
            else 1.0
        )
        hard_gate = (
            counterfactual_role_gate(
                explicit_counterfactual,
                item.evidence_role or item.name,
            )
            if explicit_counterfactual
            else 1.0
        )
        return float(hard_gate) * learned

    relevance_values = [role_relevance(item) for item in items]
    relevance_mass = sum(
        max(0.0, float(item.importance)) * max(0.0, relevance)
        for item, relevance in zip(items, relevance_values)
    )
    importance_mass = sum(max(0.0, float(item.importance)) for item in items)
    relevance_normalizer = relevance_mass / max(1e-9, importance_mass)
    if relevance_normalizer <= 1e-9:
        relevance_normalizer = 1.0
    for item_index, item in enumerate(items):
        role = item.evidence_role or item.name
        context = {
            **dict(counterfactual_context or {}),
            "item_score": float(item.score),
            "item_threshold": float(item.threshold),
            "missing_weight": max(0.0, min(1.0, 1.0 - float(item.score) / max(1e-9, float(item.threshold)))),
        }
        candidate_factor_fn = getattr(model, "candidate_affordance_factor", None)
        candidate_factor = (
            float(
                candidate_factor_fn(
                    claim_id,
                    role,
                    current_view,
                    candidate_view,
                    table,
                    context=context,
                )
            )
            if callable(candidate_factor_fn)
            else 1.0
        )
        ray_factors = dict(context.get("candidate_role_factors") or {})
        candidate_factor *= relative_role_affordance_factor(
            candidate_role_factors=ray_factors,
            current_view=current_view,
            candidate_view=candidate_view,
            role=role,
        )
        reveal_distribution = model.action_distribution(claim_id, role, context=context)
        confidence_fn = getattr(model, "transport_confidence", None)
        transport_confidence = (
            float(confidence_fn(claim_id, role, context=context))
            if callable(confidence_fn)
            else 1.0
        )
        counterfactual_relevance = (
            max(0.0, relevance_values[item_index]) / relevance_normalizer
        )
        for action in ORBIT_ACTIONS:
            reveal_prob = reveal_distribution[action]
            compat = action_compatibility(action, current_view, candidate_view, table)
            value = (
                float(item.importance)
                * context["missing_weight"]
                * transport_confidence
                * counterfactual_relevance
                * reveal_prob
                * compat
                * candidate_factor
            )
            contributions[action] += value
            gain += value
    cost = transition_cost(current_view, candidate_view, table)
    risk_penalty = float((view_risk or {}).get(candidate_view, 0.0))
    revisit_penalty = 1.0 if candidate_view in set(visited_views or []) else 0.0
    return ProjectedViewScore(
        view_id=candidate_view,
        score=gain - lambda_cost * cost - lambda_risk * risk_penalty - lambda_revisit * revisit_penalty,
        evidence_gain=gain,
        cost=cost,
        risk_penalty=risk_penalty,
        revisit_penalty=revisit_penalty,
        action_contributions={key: value for key, value in contributions.items() if value > 1e-9},
    )


def rank_candidate_views(
    *,
    model: PriorTableRevealModel,
    claim_id: str,
    missing_evidence: Iterable[EvidenceItem],
    current_view: str,
    views: Mapping[str, ViewNode] | None = None,
    lambda_cost: float = 0.15,
    lambda_risk: float = 0.0,
    lambda_revisit: float = 0.0,
    view_risk: Mapping[str, float] | None = None,
    visited_views: Iterable[str] | None = None,
    counterfactual_context: Mapping[str, Any] | None = None,
) -> List[ProjectedViewScore]:
    table = views or SIX_VIEWS
    scores = [
        score_candidate_view(
            model=model,
            claim_id=claim_id,
            missing_evidence=missing_evidence,
            current_view=current_view,
            candidate_view=view_id,
            views=table,
            lambda_cost=lambda_cost,
            lambda_risk=lambda_risk,
            lambda_revisit=lambda_revisit,
            view_risk=view_risk,
            visited_views=visited_views,
            counterfactual_context=counterfactual_context,
        )
        for view_id in candidate_views(current_view, table)
    ]
    return sorted(scores, key=lambda item: (item.score, item.evidence_gain, -item.cost, item.view_id), reverse=True)
