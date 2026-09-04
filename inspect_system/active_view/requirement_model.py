"""Assistant-trace evidence requirements with uncertainty-aware shrinkage."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping

from .counterfactual_transport import (
    infer_counterfactual_family,
    normalize_counterfactual_family,
)
from .evidence_state import EvidenceItem, EvidenceState
from .ontology import normalize_claim


def counterfactual_requirement_key(claim: object, family: object) -> str:
    normalized_claim = normalize_claim(claim)
    normalized_family = normalize_counterfactual_family(family)
    return (
        f"{normalized_claim}|cf:{normalized_family}"
        if normalized_family
        else normalized_claim
    )


def trace_event_weight(event: Mapping[str, Any]) -> float:
    """Return directional supervision mass for the relative reveal policy."""

    return (
        max(0.0, float(event.get("transfer_weight", 0.0) or 0.0))
        * max(0.0, float(event.get("label_confidence", 0.0) or 0.0))
        * max(0.0, float(event.get("evidence_stability", 0.0) or 0.0))
        * max(0.0, float(event.get("evidence_importance", 0.0) or 0.0))
    )


def requirement_event_weight(event: Mapping[str, Any]) -> float:
    """Return role-incidence supervision mass, independent of view direction."""

    direct = event.get("requirement_weight")
    if direct not in (None, ""):
        return max(0.0, float(direct))
    return trace_event_weight(event)


def load_requirement_counts(
    path: str | Path | None,
    *,
    episode_balanced: bool = True,
    counterfactual_conditioned: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Load hierarchical claim-role counts without dense-frame overcounting.

    In addition to claim-specific counts, ``__global__`` stores the same
    episode-balanced assistant supervision pooled across claims. It is used
    only as a backoff for unseen claims, without target-domain labels.
    """

    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    valid = []
    source_events = (
        payload.get("requirement_events") or payload.get("trainable_events") or []
    )
    for index, raw in enumerate(source_events):
        if not isinstance(raw, Mapping):
            continue
        event = dict(raw)
        claim = normalize_claim(event.get("claim_id", ""))
        role = str(event.get("evidence_role", "")).strip()
        weight = requirement_event_weight(event)
        if not claim or not role or weight <= 0.0:
            continue
        event["_claim"] = claim
        event["_role"] = role
        event["_weight"] = weight
        event["_family"] = infer_counterfactual_family(
            claim,
            role,
            (
                event.get("metadata")
                if isinstance(event.get("metadata"), Mapping)
                else event
            ),
        )
        event["_episode"] = "|".join(
            [
                str(event.get("video", "")),
                str(event.get("after_frame", "")),
                claim,
            ]
        ) or str(event.get("event_id", index))
        valid.append(event)

    episode_mass: Dict[str, float] = {}
    for event in valid:
        key = str(event["_episode"])
        episode_mass[key] = episode_mass.get(key, 0.0) + float(event["_weight"])
    counts: Dict[str, Dict[str, float]] = {}
    for event in valid:
        weight = float(event["_weight"])
        if episode_balanced:
            weight /= max(1e-12, episode_mass[str(event["_episode"])])
        role = str(event["_role"])
        keys = [str(event["_claim"]), "__global__"]
        if counterfactual_conditioned:
            keys.append(
                counterfactual_requirement_key(event["_claim"], event["_family"])
            )
        for key in dict.fromkeys(keys):
            bucket = counts.setdefault(key, {})
            bucket[role] = float(bucket.get(role, 0.0)) + weight
    return counts


@dataclass(frozen=True)
class RequirementCalibration:
    """Frozen assistant-only calibration for combining R(c,q) with pi."""

    beta: float = 2.0
    strength: float = 1.0
    temperature: float = 1.0

    @classmethod
    def load(
        cls,
        path: str | Path | None,
        *,
        beta: float = 2.0,
        strength: float = 1.0,
        temperature: float = 1.0,
    ) -> "RequirementCalibration":
        if path is None:
            return cls(
                beta=max(1e-9, float(beta)),
                strength=_clamp01(strength),
                temperature=max(0.0, float(temperature)),
            )
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        protocol = dict(payload.get("protocol") or {})
        if protocol.get("uses_robot_view_training") or protocol.get(
            "uses_robot_utility_labels"
        ):
            raise RuntimeError(
                "Evidence-requirement calibration must not use robot supervision."
            )
        selected = dict(payload.get("selected") or {})
        return cls(
            beta=max(1e-9, float(selected.get("beta", beta))),
            strength=_clamp01(selected.get("requirement_strength", strength)),
            temperature=max(
                0.0, float(selected.get("requirement_temperature", temperature))
            ),
        )


def mix_requirement_weights(
    state: EvidenceState,
    learned: Mapping[str, Mapping[str, float]],
    *,
    beta: float,
    strength: float,
    temperature: float = 1.0,
    uniform: bool = False,
    preserve_scores: bool = True,
) -> EvidenceState:
    """Apply a calibrated unit-mass R(c,q) over the current evidence roles.

    ``strength=0`` is the strict pi-only fallback with uniform role mass;
    ``strength=1`` recovers the ontology-plus-trace posterior. Intermediate
    values are selected on held-out assistant sessions. ``uniform=True``
    always implements the explicit pi-only ablation.
    """

    if not state.items:
        return state
    neutral = 1.0 / len(state.items)
    base_total = sum(max(0.0, float(item.importance)) for item in state.items)
    raw_prior = [
        (max(0.0, float(item.importance)) / max(1e-9, base_total))
        ** max(0.0, float(temperature))
        for item in state.items
    ]
    tempered_mass = sum(raw_prior)
    tempered_prior = [value / max(1e-9, tempered_mass) for value in raw_prior]
    claim = normalize_claim(state.claim_id)
    conditional_key = counterfactual_requirement_key(
        claim,
        state.counterfactual_family,
    )
    claim_values = (
        learned.get(conditional_key)
        or learned.get(claim)
        or learned.get("__global__", {})
    )
    event_mass = sum(max(0.0, float(value)) for value in claim_values.values())
    concentration = max(1e-9, float(beta))
    residual = 0.0 if uniform else _clamp01(strength)
    items: list[EvidenceItem] = []
    for item_index, item in enumerate(state.items):
        prior = tempered_prior[item_index]
        role = item.evidence_role or item.name
        learned_value = max(0.0, float(claim_values.get(role, 0.0)))
        posterior = (concentration * prior + learned_value) / max(
            1e-9, concentration + event_mass
        )
        importance = (
            neutral if uniform else (1.0 - residual) * neutral + residual * posterior
        )
        score = (
            item.score
            if preserve_scores
            else (item.threshold if item.observed else 0.0)
        )
        items.append(replace(item, importance=importance, score=score))

    mass = sum(item.importance for item in items)
    if mass > 1e-9:
        items = [replace(item, importance=item.importance / mass) for item in items]
    missing = [index for index, item in enumerate(items) if not item.observed]
    observed = [index for index, item in enumerate(items) if item.observed]
    if missing and observed:
        # R ranks missing roles without changing MOVE/DEFER score scale.
        target_missing_mass = len(missing) / len(items)
        current_missing_mass = sum(items[index].importance for index in missing)
        current_observed_mass = sum(items[index].importance for index in observed)
        missing_scale = target_missing_mass / max(1e-9, current_missing_mass)
        observed_scale = (1.0 - target_missing_mass) / max(1e-9, current_observed_mass)
        items = [
            replace(
                item,
                importance=item.importance
                * (missing_scale if index in missing else observed_scale),
            )
            for index, item in enumerate(items)
        ]
    return EvidenceState(
        claim_id=state.claim_id,
        items=items,
        claim_score=state.claim_score,
        contradiction_score=state.contradiction_score,
        margin=state.margin,
        counterfactual_family=state.counterfactual_family,
        counterfactual_scores=dict(state.counterfactual_scores),
        current_utility_proxy=None,
    )


def blend_session_requirement_weights(
    state: EvidenceState,
    session_counts: Mapping[str, float] | None,
    *,
    blend: float = 1.0,
    prior_strength: float = 0.05,
) -> EvidenceState:
    """Fuse causal session-local evidence requirements with frozen R(c,q).

    The incoming state already contains the frozen assistant-trained
    requirement posterior. Session counts are converted to a Dirichlet-smoothed
    role distribution and mixed convexly with that posterior. The total mass
    assigned to missing versus observed roles is preserved, so online feedback
    changes role ranking without changing the selector's MOVE/DEFER scale.
    """

    if not state.items or not session_counts or _clamp01(blend) <= 0.0:
        return state
    prior_mass = sum(max(0.0, float(item.importance)) for item in state.items)
    if prior_mass <= 1e-12:
        return state
    prior = [max(0.0, float(item.importance)) / prior_mass for item in state.items]
    local = [
        max(
            0.0,
            float(session_counts.get(item.evidence_role or item.name, 0.0)),
        )
        for item in state.items
    ]
    local_mass = sum(local)
    if local_mass <= 1e-12:
        return state
    concentration = max(1e-9, float(prior_strength))
    session = [
        (local[index] + concentration * prior[index]) / (local_mass + concentration)
        for index in range(len(state.items))
    ]
    residual = _clamp01(blend)
    mixed = [
        (1.0 - residual) * prior[index] + residual * session[index]
        for index in range(len(state.items))
    ]

    missing = [index for index, item in enumerate(state.items) if not item.observed]
    observed = [index for index, item in enumerate(state.items) if item.observed]
    if missing and observed:
        target_missing = sum(prior[index] for index in missing)
        mixed_missing = sum(mixed[index] for index in missing)
        mixed_observed = sum(mixed[index] for index in observed)
        missing_scale = target_missing / max(1e-12, mixed_missing)
        observed_scale = (1.0 - target_missing) / max(1e-12, mixed_observed)
        mixed = [
            value * (missing_scale if index in missing else observed_scale)
            for index, value in enumerate(mixed)
        ]
    mass = sum(mixed)
    items = [
        replace(item, importance=mixed[index] / max(1e-12, mass))
        for index, item in enumerate(state.items)
    ]
    return EvidenceState(
        claim_id=state.claim_id,
        items=items,
        claim_score=state.claim_score,
        contradiction_score=state.contradiction_score,
        margin=state.margin,
        counterfactual_family=state.counterfactual_family,
        counterfactual_scores=dict(state.counterfactual_scores),
        current_utility_proxy=state.current_utility_proxy,
    )


def _clamp01(value: object) -> float:
    return max(0.0, min(1.0, float(value)))
