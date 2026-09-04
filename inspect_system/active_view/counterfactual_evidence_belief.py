"""Conservative multi-view belief fusion for procedural claim evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from .ontology import normalize_claim, normalize_key, roles_for_claim
from ..types import VerificationResult


def _clamp01(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _as_bool(metadata: Mapping[str, Any], key: str, fallback: bool) -> bool:
    value = metadata.get(key)
    if value is None:
        return fallback
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "supported"}
    return bool(value)


@dataclass(frozen=True)
class EvidenceMass:
    """Mass over support, contradiction, ignorance, and explicit conflict."""

    support: float = 0.0
    contradiction: float = 0.0
    ignorance: float = 1.0
    conflict: float = 0.0

    @classmethod
    def from_scores(
        cls,
        *,
        support: float,
        contradiction: float,
        support_availability: float,
        contradiction_availability: float,
    ) -> "EvidenceMass":
        support_mass = _clamp01(support) * _clamp01(support_availability)
        contradiction_mass = _clamp01(contradiction) * _clamp01(
            contradiction_availability
        )
        committed = support_mass + contradiction_mass
        if committed > 1.0:
            support_mass /= committed
            contradiction_mass /= committed
            committed = 1.0
        return cls(
            support=support_mass,
            contradiction=contradiction_mass,
            ignorance=1.0 - committed,
            conflict=0.0,
        )

    def combine(self, other: "EvidenceMass") -> "EvidenceMass":
        """Conjunctive fusion that retains disagreement as ignorance-inducing conflict."""

        left = {
            "support": self.support,
            "contradiction": self.contradiction,
            "ignorance": self.ignorance,
            "conflict": self.conflict,
        }
        right = {
            "support": other.support,
            "contradiction": other.contradiction,
            "ignorance": other.ignorance,
            "conflict": other.conflict,
        }
        result = {key: 0.0 for key in left}
        for left_key, left_value in left.items():
            for right_key, right_value in right.items():
                value = left_value * right_value
                if "conflict" in {left_key, right_key}:
                    result["conflict"] += value
                elif left_key == "ignorance":
                    result[right_key] += value
                elif right_key == "ignorance":
                    result[left_key] += value
                elif left_key == right_key:
                    result[left_key] += value
                else:
                    result["conflict"] += value
        return EvidenceMass(**result)

    def to_dict(self) -> Dict[str, float]:
        return {
            "support": float(self.support),
            "contradiction": float(self.contradiction),
            "ignorance": float(self.ignorance),
            "conflict": float(self.conflict),
        }


@dataclass(frozen=True)
class CounterfactualEvidenceBeliefConfig:
    support_threshold: float = 0.35
    contradiction_threshold: float = 0.45
    margin_threshold: float = 0.05
    role_threshold: float = 0.65
    conflict_threshold: float = 0.20


@dataclass
class CounterfactualEvidenceBelief:
    """Accumulate complementary evidence without hiding cross-view conflicts."""

    claim_id: str
    config: CounterfactualEvidenceBeliefConfig = field(
        default_factory=CounterfactualEvidenceBeliefConfig
    )
    mass: EvidenceMass = field(default_factory=EvidenceMass)
    role_scores: Dict[str, float] = field(default_factory=dict)
    observed_evidence: set[str] = field(default_factory=set)
    observation_ids: set[str] = field(default_factory=set)
    identity_available: bool = False
    relation_available: bool = False
    contradiction_available: bool = False
    last_template: VerificationResult | None = field(default=None, repr=False)

    def observe(self, verification: VerificationResult) -> VerificationResult:
        observation_id = str(verification.observation_id)
        if observation_id in self.observation_ids:
            return self.to_verification(verification)
        self.observation_ids.add(observation_id)
        self.last_template = verification

        metadata = dict(verification.metadata)
        decision = str(metadata.get("decision", "")).strip().lower()
        if not decision:
            if verification.verified:
                decision = "contradicted" if verification.anomaly else "supported"
            else:
                decision = "insufficient"

        identity = _as_bool(
            metadata,
            "identity_available",
            decision == "supported" and verification.verified,
        )
        relation = _as_bool(
            metadata,
            "relation_available",
            decision == "supported" and verification.verified,
        )
        wrong_identity = _as_bool(
            metadata,
            "wrong_identity_visible",
            decision == "contradicted" and verification.anomaly,
        )
        incomplete_relation = _as_bool(
            metadata,
            "incomplete_relation_visible",
            decision == "contradicted" and verification.anomaly,
        )
        contradiction_available = _as_bool(
            metadata,
            "contradiction_evidence_available",
            wrong_identity or incomplete_relation,
        )

        self.identity_available = self.identity_available or identity
        self.relation_available = self.relation_available or relation
        self.contradiction_available = (
            self.contradiction_available or contradiction_available
        )

        role_scores = dict(metadata.get("role_scores") or {})
        role_scores.update(dict(metadata.get("evidence_role_scores") or {}))
        for role, score in role_scores.items():
            key = normalize_key(role)
            self.role_scores[key] = max(
                self.role_scores.get(key, 0.0),
                _clamp01(score),
            )
        self.observed_evidence.update(
            normalize_key(value)
            for value in verification.observed_evidence
            if str(value).strip()
        )

        support_availability = 0.5 * (float(identity) + float(relation))
        if "identity_available" not in metadata and "relation_available" not in metadata:
            support_availability = (
                1.0 if decision == "supported" else _clamp01(verification.evidence_coverage)
            )
        contradiction_availability = float(contradiction_available)
        if "contradiction_evidence_available" not in metadata:
            contradiction_availability = (
                1.0 if decision == "contradicted" else contradiction_availability
            )
        view_mass = EvidenceMass.from_scores(
            support=_clamp01(metadata.get("support_score", verification.confidence)),
            contradiction=_clamp01(metadata.get("contradiction_score", 0.0)),
            support_availability=support_availability,
            contradiction_availability=contradiction_availability,
        )
        self.mass = self.mass.combine(view_mass)
        return self.to_verification(verification)

    def decision(self) -> str:
        conflict = float(self.mass.conflict)
        support = float(self.mass.support)
        contradiction = float(self.mass.contradiction)
        if conflict >= float(self.config.conflict_threshold):
            return "insufficient"
        if (
            self.contradiction_available
            and contradiction >= float(self.config.contradiction_threshold)
            and contradiction >= support + float(self.config.margin_threshold)
        ):
            return "contradicted"
        if (
            self.identity_available
            and self.relation_available
            and support >= float(self.config.support_threshold)
            and support >= contradiction + float(self.config.margin_threshold)
        ):
            return "supported"
        return "insufficient"

    def to_verification(
        self,
        template: VerificationResult | None = None,
    ) -> VerificationResult:
        source = template or self.last_template
        if source is None:
            raise RuntimeError("Cannot render a claim belief before observing evidence.")
        decision = self.decision()
        expected_roles = roles_for_claim(self.claim_id)
        threshold = float(self.config.role_threshold)
        observed = sorted(
            set(self.observed_evidence)
            | {
                role
                for role, score in self.role_scores.items()
                if float(score) >= threshold
            }
        )
        missing = sorted(
            role
            for role in expected_roles
            if float(self.role_scores.get(role, 0.0)) < threshold
        )
        resolved = decision in {"supported", "contradicted"}
        confidence = max(float(self.mass.support), float(self.mass.contradiction))
        metadata = {
            **dict(source.metadata),
            "decision": decision,
            "support_score": float(self.mass.support),
            "contradiction_score": float(self.mass.contradiction),
            "counterfactual_margin": float(
                self.mass.support - self.mass.contradiction
            ),
            "role_scores": dict(self.role_scores),
            "evidence_role_scores": dict(self.role_scores),
            "missing_roles": missing,
            "identity_available": bool(self.identity_available),
            "relation_available": bool(self.relation_available),
            "contradiction_evidence_available": bool(
                self.contradiction_available
            ),
            "counterfactual_evidence_belief": {
                "mass": self.mass.to_dict(),
                "observations": len(self.observation_ids),
                "decision": decision,
            },
        }
        coverage = (
            sum(float(self.role_scores.get(role, 0.0)) for role in expected_roles)
            / max(1, len(expected_roles))
        )
        return VerificationResult(
            observation_id=source.observation_id,
            predicted_state=source.predicted_state,
            verified=resolved,
            confidence=confidence,
            evidence_coverage=_clamp01(coverage),
            observed_evidence=observed,
            missing_evidence=missing,
            contradicted_evidence=(missing if decision == "contradicted" else []),
            next_step_admissible=decision == "supported",
            anomaly=decision == "contradicted",
            recommended_action=(
                "continue"
                if decision == "supported"
                else "pause" if decision == "contradicted" else "observe"
            ),
            state_scores={
                "supported": float(self.mass.support),
                "contradicted": float(self.mass.contradiction),
                "insufficient": float(self.mass.ignorance + self.mass.conflict),
            },
            metadata=metadata,
        )


@dataclass
class CounterfactualEvidenceBeliefMemory:
    config: CounterfactualEvidenceBeliefConfig = field(
        default_factory=CounterfactualEvidenceBeliefConfig
    )
    beliefs: Dict[str, CounterfactualEvidenceBelief] = field(default_factory=dict)

    def observe(self, verification: VerificationResult) -> VerificationResult:
        metadata = dict(verification.metadata)
        claim_id = normalize_claim(
            metadata.get("claim_id", verification.predicted_state)
        )
        belief = self.beliefs.setdefault(
            claim_id,
            CounterfactualEvidenceBelief(claim_id=claim_id, config=self.config),
        )
        return belief.observe(verification)

    def clear(self) -> None:
        self.beliefs.clear()
