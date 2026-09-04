"""Provenance-preserving evidence memory for multi-view claim verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping


RELATION_ROLES = (
    "insertion_verification_view",
    "containment_verification_view",
    "slot_relation_view",
    "gap_visibility_view",
    "boundary_alignment_view",
    "contact_verification_view",
)


@dataclass(frozen=True)
class EvidenceBundleConfig:
    identity_confidence: float = 0.25
    frozen_identity_margin_floor: float = -0.08
    alternative_identity_confidence: float = 0.65
    counterfactual_margin: float = 0.0
    explicit_contradiction_margin: float = 0.08
    relation_evidence: float = 0.18
    support_threshold: float = 0.35
    contradiction_threshold: float = 0.45
    score_margin: float = 0.05


@dataclass
class ProvenanceEvidenceBundle:
    """Accumulate compatible role evidence while retaining source views.

    Positive identity, explicit counterfactual identity, and relation evidence
    remain separate. Conflicting identity evidence produces abstention instead
    of being averaged into a confident claim.
    """

    config: EvidenceBundleConfig = field(default_factory=EvidenceBundleConfig)
    role_evidence: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    positive_identity: Dict[str, Any] | None = None
    counterfactual_identity: Dict[str, Any] | None = None
    support_candidate: Dict[str, Any] | None = None
    contradiction_candidate: Dict[str, Any] | None = None
    observed_views: list[str] = field(default_factory=list)

    @staticmethod
    def _replace_if_better(
        current: Dict[str, Any] | None,
        candidate: Dict[str, Any],
        key: str,
    ) -> Dict[str, Any]:
        if current is None or float(candidate[key]) > float(current[key]):
            return candidate
        return current

    def observe(self, view_id: str, prediction: Mapping[str, Any]) -> None:
        features = dict(prediction.get("features") or {})
        roles = dict(prediction.get("role_scores") or {})
        target = float(features.get("target_conf", 0.0))
        wrong = float(features.get("wrong_same_role_conf", 0.0))
        housing = float(features.get("housing_role_conf", 0.0))
        identity_margin = float(features.get("identity_margin", target - wrong))
        alternative_reliable = wrong >= self.config.alternative_identity_confidence
        margin_floor = (
            self.config.counterfactual_margin
            if alternative_reliable
            else self.config.frozen_identity_margin_floor
        )
        identity_positive = bool(
            target >= self.config.identity_confidence
            and identity_margin >= margin_floor
        )
        explicit_counterfactual = bool(
            wrong >= self.config.alternative_identity_confidence
            and wrong >= target + self.config.explicit_contradiction_margin
            and housing >= 0.10
        )

        self.observed_views.append(str(view_id))
        if identity_positive:
            self.positive_identity = self._replace_if_better(
                self.positive_identity,
                {
                    "view_id": str(view_id),
                    "score": target,
                    "margin": identity_margin,
                },
                "score",
            )
        if explicit_counterfactual:
            self.counterfactual_identity = self._replace_if_better(
                self.counterfactual_identity,
                {
                    "view_id": str(view_id),
                    "score": wrong,
                    "margin": wrong - target,
                },
                "score",
            )

        for role, value in roles.items():
            score = float(value)
            if role == "identity_disambiguation_view" and not identity_positive:
                continue
            if role in RELATION_ROLES and min(target, housing) < 0.10:
                continue
            current = self.role_evidence.get(str(role))
            candidate = {"view_id": str(view_id), "score": score}
            self.role_evidence[str(role)] = self._replace_if_better(
                current, candidate, "score"
            )

        support = float(prediction.get("support_score", 0.0))
        contradiction = float(prediction.get("contradiction_score", 0.0))
        self.support_candidate = self._replace_if_better(
            self.support_candidate,
            {"view_id": str(view_id), "score": support},
            "score",
        )
        if explicit_counterfactual:
            self.contradiction_candidate = self._replace_if_better(
                self.contradiction_candidate,
                {
                    "view_id": str(view_id),
                    "score": max(contradiction, wrong),
                },
                "score",
            )

    def decision(self) -> Dict[str, Any]:
        support = float((self.support_candidate or {}).get("score", 0.0))
        contradiction = float(
            (self.contradiction_candidate or {}).get("score", 0.0)
        )
        relation = max(
            (
                float(self.role_evidence.get(role, {}).get("score", 0.0))
                for role in RELATION_ROLES
            ),
            default=0.0,
        )
        identity_available = self.positive_identity is not None
        counterfactual_available = self.counterfactual_identity is not None
        identity_conflict = bool(identity_available and counterfactual_available)

        if identity_conflict:
            label = "insufficient"
            reason = "counterfactual_identity_conflict"
        elif (
            counterfactual_available
            and contradiction >= self.config.contradiction_threshold
            and contradiction >= support + self.config.score_margin
        ):
            label = "contradicted"
            reason = "explicit_counterfactual_identity"
        elif (
            identity_available
            and relation >= self.config.relation_evidence
            and support >= self.config.support_threshold
            and support >= contradiction + self.config.score_margin
        ):
            label = "supported"
            reason = "cross_view_positive_evidence_bundle"
        else:
            label = "insufficient"
            reason = "bundle_not_decidable"

        return {
            "decision": label,
            "reason": reason,
            "support_score": support,
            "contradiction_score": contradiction,
            "relation_score": relation,
            "identity_available": identity_available,
            "counterfactual_available": counterfactual_available,
            "identity_conflict": identity_conflict,
            "positive_identity": self.positive_identity,
            "counterfactual_identity": self.counterfactual_identity,
            "role_evidence": dict(self.role_evidence),
            "observed_views": list(self.observed_views),
        }
