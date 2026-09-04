"""Counterfactual-conditioned evidence transport learned from assistant traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping

from .evidence_transport import (
    ACTION_FAMILIES,
    HierarchicalEvidenceTransportModel,
    _event_weight,
    _normalized,
    _posterior,
    action_family,
)
from .ontology import normalize_claim, normalize_key
from .trace_event_miner import MinedEvent


COUNTERFACTUAL_FAMILIES = (
    "identity",
    "spatial_relation",
    "seating_contact",
    "state_absence",
    "generic",
)

COUNTERFACTUAL_ALIASES: Mapping[str, str] = {
    "wrong_identity": "identity",
    "wrong_family": "identity",
    "identity": "identity",
    "installed_wrongly": "spatial_relation",
    "wrong_orientation": "spatial_relation",
    "incomplete_insertion": "spatial_relation",
    "relation_failure": "spatial_relation",
    "spatial_relation": "spatial_relation",
    "not_seated": "seating_contact",
    "cover_not_seated": "seating_contact",
    "gap_or_contact": "seating_contact",
    "seating_contact": "seating_contact",
    "not_installed": "state_absence",
    "absent": "state_absence",
    "state_absence": "state_absence",
    "generic": "generic",
    "negated_claim": "generic",
}

_GENERIC_EVIDENCE_ROLES = {
    "",
    "claim_disambiguation_view",
    "claim_evidence_visibility_view",
    "object_presence_view",
}


def normalize_counterfactual_family(value: object) -> str:
    key = normalize_key(value)
    return COUNTERFACTUAL_ALIASES.get(key, key if key in COUNTERFACTUAL_FAMILIES else "")


def infer_counterfactual_family(
    claim_id: object,
    evidence_role: object,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    meta = metadata or {}
    for key in ("counterfactual_family", "counterfactual_type", "error_type"):
        family = normalize_counterfactual_family(meta.get(key, ""))
        if family:
            return family
    claim = normalize_claim(claim_id)
    projected_role = normalize_key(evidence_role)
    source_role = normalize_key(meta.get("source_evidence_role", ""))
    # A generic trace event may be projected to several ontology roles.  The
    # projected role carries the counterfactual semantics; the generic source
    # role is only a fallback for unprojected events.
    role = (
        projected_role
        if projected_role not in _GENERIC_EVIDENCE_ROLES
        else source_role or projected_role
    )
    if "identity" in role:
        return "identity"
    if claim == "cover_seated" and any(token in role for token in ("gap", "boundary", "contact")):
        return "seating_contact"
    if any(token in role for token in ("insert", "contain", "slot", "align", "orientation")):
        return "spatial_relation"
    return "generic"


def counterfactual_mixture(
    context: Mapping[str, Any] | None,
    fallback: object,
) -> Dict[str, float]:
    raw: Dict[str, float] = {}
    scores = (context or {}).get("counterfactual_scores", {})
    if isinstance(scores, Mapping):
        for key, value in scores.items():
            family = normalize_counterfactual_family(key)
            mass = max(0.0, float(value))
            if not family or mass <= 1e-12:
                continue
            raw[family] = raw.get(family, 0.0) + mass
    if sum(raw.values()) > 1e-12:
        raw["generic"] = raw.get("generic", 0.0) + max(0.0, 1.0 - sum(raw.values()))
    else:
        explicit = normalize_counterfactual_family(
            (context or {}).get("counterfactual_family", "")
        )
        family = explicit or normalize_counterfactual_family(fallback) or "generic"
        raw[family] = 1.0
    total = sum(raw.values())
    return {key: value / max(1e-12, total) for key, value in raw.items()}


def counterfactual_role_gate(family: object, role: object) -> float:
    """Binary H(c, q): whether a role can separate the counterfactual."""

    cf = normalize_counterfactual_family(family) or "generic"
    evidence_role = normalize_key(role)
    if cf == "generic":
        return 1.0
    separating_roles: Mapping[str, set[str]] = {
        "identity": {
            "identity_disambiguation_view",
            "claim_disambiguation_view",
            "occlusion_recovery_view",
        },
        "spatial_relation": {
            "insertion_verification_view",
            "claim_disambiguation_view",
            "containment_verification_view",
            "slot_relation_view",
            "boundary_alignment_view",
            "contact_verification_view",
        },
        "seating_contact": {
            "gap_visibility_view",
            "claim_disambiguation_view",
            "boundary_alignment_view",
            "contact_verification_view",
            "occlusion_recovery_view",
        },
        "state_absence": {
            "identity_disambiguation_view",
            "occlusion_recovery_view",
            "claim_disambiguation_view",
            "object_presence_view",
        },
    }
    return float(evidence_role in separating_roles.get(cf, set()))


def counterfactual_role_relevance(family: object, role: object) -> float:
    """Ontology prior for whether a role separates a counterfactual family."""
    cf = normalize_counterfactual_family(family) or "generic"
    evidence_role = normalize_key(role)
    if cf == "generic":
        return 1.0
    tables: Mapping[str, Mapping[str, float]] = {
        "identity": {
            "identity_disambiguation_view": 1.0,
            "occlusion_recovery_view": 0.55,
            "claim_disambiguation_view": 0.45,
        },
        "spatial_relation": {
            "insertion_verification_view": 1.0,
            "containment_verification_view": 1.0,
            "slot_relation_view": 1.0,
            "boundary_alignment_view": 0.80,
            "contact_verification_view": 0.65,
            "claim_disambiguation_view": 0.45,
            "identity_disambiguation_view": 0.25,
        },
        "seating_contact": {
            "gap_visibility_view": 1.0,
            "boundary_alignment_view": 1.0,
            "contact_verification_view": 0.90,
            "occlusion_recovery_view": 0.55,
            "claim_disambiguation_view": 0.45,
        },
        "state_absence": {
            "identity_disambiguation_view": 0.75,
            "occlusion_recovery_view": 1.0,
            "claim_disambiguation_view": 0.65,
        },
    }
    return float(tables.get(cf, {}).get(evidence_role, 0.15))


def signed_counterfactual_gain(event: MinedEvent) -> float | None:
    metadata = event.metadata or {}
    value = metadata.get("signed_counterfactual_margin_gain")
    if value not in (None, ""):
        return float(value)
    before = metadata.get("counterfactual_margin_before")
    after = metadata.get("counterfactual_margin_after")
    if before in (None, "") or after in (None, ""):
        return None
    sign = -1.0 if normalize_key(metadata.get("world_outcome", "")) == "contradicted" else 1.0
    return sign * (float(after) - float(before))


def resolution_decidability_gain(event: MinedEvent) -> float | None:
    """Return evidence gain toward a resolved decision, independent of polarity.

    Counterfactual-margin events already encode whether the target-versus-
    alternative margin moved in the correct direction. Older assistant traces
    instead record a role-specific evidence score before and after a
    user-validated resolution. Both support and contradiction are useful
    inspection outcomes, so the latter is a positive decidability gain rather
    than a signed claim-support target.
    """

    metadata = event.metadata or {}
    explicit = metadata.get("resolution_decidability_gain")
    if explicit not in (None, ""):
        return max(0.0, float(explicit))
    margin_gain = signed_counterfactual_gain(event)
    if margin_gain is not None:
        return max(0.0, float(margin_gain))
    score_gain = metadata.get("score_gain")
    if score_gain not in (None, ""):
        return max(0.0, float(score_gain))
    before = metadata.get("before_score")
    after = metadata.get("after_score")
    if before in (None, "") or after in (None, ""):
        return None
    return max(0.0, float(after) - float(before))


@dataclass
class CounterfactualEvidenceTransportModel(HierarchicalEvidenceTransportModel):
    """Transport posterior weighted by target-versus-counterfactual separation."""

    counterfactual_strength: float = 2.0
    unmatched_resolution_weight: float = 0.25
    margin_gain_floor: float = 0.05
    counterfactual_success_counts: Dict[str, Dict[str, float]] = field(default_factory=dict)
    counterfactual_failure_counts: Dict[str, Dict[str, float]] = field(default_factory=dict)
    counterfactual_role_success_counts: Dict[str, Dict[str, float]] = field(default_factory=dict)
    counterfactual_role_failure_counts: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @staticmethod
    def _counterfactual_role_key(family: str, role: str) -> str:
        return f"{normalize_counterfactual_family(family) or 'generic'}|{normalize_key(role)}"

    @staticmethod
    def _add_counter(
        table: Dict[str, Dict[str, float]],
        key: str,
        family: str,
        weight: float,
    ) -> None:
        bucket = table.setdefault(key, {})
        bucket[family] = float(bucket.get(family, 0.0)) + float(weight)

    def fit_events(self, events: Iterable[MinedEvent]) -> "CounterfactualEvidenceTransportModel":
        rows = list(events)
        super().fit_events(rows)
        self.counterfactual_success_counts = {}
        self.counterfactual_failure_counts = {}
        self.counterfactual_role_success_counts = {}
        self.counterfactual_role_failure_counts = {}
        matched = 0
        positive = 0
        by_family: Dict[str, int] = {}
        for event in rows:
            if not event.transferable or not event.label:
                continue
            action_group = action_family(event.relative_action)
            if not action_group:
                continue
            cf = infer_counterfactual_family(event.claim_id, event.evidence_role, event.metadata)
            by_family[cf] = by_family.get(cf, 0) + 1
            base = _event_weight(event, self.gain_power)
            gain = signed_counterfactual_gain(event)
            if gain is None:
                is_success = True
                weight = base * max(0.0, float(self.unmatched_resolution_weight))
            else:
                matched += 1
                is_success = gain > 0.0
                positive += int(is_success)
                weight = base * max(
                    float(self.margin_gain_floor),
                    min(1.0, abs(float(gain))),
                )
            if weight <= 0.0:
                continue
            family_table = (
                self.counterfactual_success_counts
                if is_success
                else self.counterfactual_failure_counts
            )
            role_table = (
                self.counterfactual_role_success_counts
                if is_success
                else self.counterfactual_role_failure_counts
            )
            self._add_counter(family_table, cf, action_group, weight)
            self._add_counter(
                role_table,
                self._counterfactual_role_key(cf, event.evidence_role),
                action_group,
                weight,
            )
        self.metadata.update(
            {
                "model_type": "counterfactual_evidence_transport",
                "counterfactual_margin_matched_events": matched,
                "positive_counterfactual_gain_events": positive,
                "counterfactual_families": by_family,
                "uses_robot_view_training": False,
            }
        )
        return self

    def _counterfactual_distribution(
        self,
        claim_id: str,
        evidence_role: str,
        family: str,
    ) -> Dict[str, float]:
        base = super().family_distribution(claim_id, evidence_role)
        cf = normalize_counterfactual_family(family) or infer_counterfactual_family(
            claim_id,
            evidence_role,
        )
        success = self.counterfactual_success_counts.get(cf, {})
        cf_p = _posterior(success, base, self.counterfactual_strength)
        role_key = self._counterfactual_role_key(cf, evidence_role)
        exact_success = self.counterfactual_role_success_counts.get(role_key, {})
        exact_failure = self.counterfactual_role_failure_counts.get(role_key, {})
        family_failure = self.counterfactual_failure_counts.get(cf, {})
        exact_p = _posterior(exact_success, cf_p, self.counterfactual_strength)
        calibrated: Dict[str, float] = {}
        for action_group in ACTION_FAMILIES:
            success_mass = float(success.get(action_group, 0.0)) + float(
                exact_success.get(action_group, 0.0)
            )
            failure_mass = float(family_failure.get(action_group, 0.0)) + float(
                exact_failure.get(action_group, 0.0)
            )
            reliability = (success_mass + 1.0) / (success_mass + failure_mass + 2.0)
            calibrated[action_group] = exact_p[action_group] * reliability
        return _normalized(calibrated)

    def probability(
        self,
        claim_id: str,
        evidence_role: str,
        action: str,
        context: Mapping[str, Any] | None = None,
    ) -> float:
        normalized_action = normalize_key(action)
        family = action_family(normalized_action)
        if not family:
            return 0.0
        mixture = counterfactual_mixture(
            context,
            infer_counterfactual_family(claim_id, evidence_role),
        )
        distribution = {action_group: 0.0 for action_group in ACTION_FAMILIES}
        for counterfactual_family, mixture_weight in mixture.items():
            conditional = self._counterfactual_distribution(
                claim_id,
                evidence_role,
                counterfactual_family,
            )
            for action_group in ACTION_FAMILIES:
                distribution[action_group] += mixture_weight * conditional[action_group]
        members = tuple(
            candidate
            for candidate in self.metadata.get("actions", [])
            if action_family(str(candidate)) == family
        )
        if not members:
            from .evidence_transport import FAMILY_MEMBERS

            members = FAMILY_MEMBERS[family]
        return distribution[family] / len(members)

    def counterfactual_relevance(
        self,
        claim_id: str,
        evidence_role: str,
        context: Mapping[str, Any] | None = None,
    ) -> float:
        mixture = counterfactual_mixture(
            context,
            infer_counterfactual_family(claim_id, evidence_role),
        )
        return sum(
            weight * counterfactual_role_gate(family, evidence_role)
            for family, weight in mixture.items()
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "counterfactual_strength": self.counterfactual_strength,
                "unmatched_resolution_weight": self.unmatched_resolution_weight,
                "margin_gain_floor": self.margin_gain_floor,
                "counterfactual_success_counts": self.counterfactual_success_counts,
                "counterfactual_failure_counts": self.counterfactual_failure_counts,
                "counterfactual_role_success_counts": self.counterfactual_role_success_counts,
                "counterfactual_role_failure_counts": self.counterfactual_role_failure_counts,
            }
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CounterfactualEvidenceTransportModel":
        base = HierarchicalEvidenceTransportModel.from_dict(payload)

        def nested(name: str) -> Dict[str, Dict[str, float]]:
            return {
                str(key): {str(k): float(v) for k, v in dict(value).items()}
                for key, value in dict(payload.get(name, {}) or {}).items()
            }

        return cls(
            **{
                key: getattr(base, key)
                for key in (
                    "alpha",
                    "default_probability",
                    "counts",
                    "metadata",
                    "base_concentration",
                    "hierarchy_strength",
                    "risk_beta",
                    "gain_power",
                    "confidence_scale",
                    "global_family_counts",
                    "claim_family_counts",
                    "role_family_counts",
                    "claim_role_family_counts",
                )
            },
            counterfactual_strength=float(payload.get("counterfactual_strength", 2.0)),
            unmatched_resolution_weight=float(payload.get("unmatched_resolution_weight", 0.25)),
            margin_gain_floor=float(payload.get("margin_gain_floor", 0.05)),
            counterfactual_success_counts=nested("counterfactual_success_counts"),
            counterfactual_failure_counts=nested("counterfactual_failure_counts"),
            counterfactual_role_success_counts=nested("counterfactual_role_success_counts"),
            counterfactual_role_failure_counts=nested("counterfactual_role_failure_counts"),
        )
