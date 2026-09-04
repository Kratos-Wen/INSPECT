"""Evidence-state containers used by the relative view policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Mapping


@dataclass(frozen=True)
class EvidenceItem:
    name: str
    score: float = 0.0
    threshold: float = 0.85
    importance: float = 1.0
    evidence_role: str = ""

    @property
    def observed(self) -> bool:
        return float(self.score) >= float(self.threshold)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class EvidenceState:
    claim_id: str
    items: List[EvidenceItem] = field(default_factory=list)
    claim_score: float = 0.0
    contradiction_score: float = 0.0
    margin: float = 0.0
    counterfactual_family: str = ""
    counterfactual_scores: Dict[str, float] = field(default_factory=dict)
    current_utility_proxy: float | None = None

    def missing(self) -> List[EvidenceItem]:
        return [item for item in self.items if not item.observed]

    def observed(self) -> List[EvidenceItem]:
        return [item for item in self.items if item.observed]

    def is_certain(self, threshold: float = 0.85) -> bool:
        if self.current_utility_proxy is not None:
            return float(self.current_utility_proxy) >= 2.0
        if self.contradiction_score >= threshold:
            return True
        return bool(self.items) and all(item.observed for item in self.items)

    def evidence_vector(self) -> Dict[str, float]:
        return {item.name: float(item.score) for item in self.items}

    def to_dict(self) -> Dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "items": [item.to_dict() for item in self.items],
            "claim_score": self.claim_score,
            "contradiction_score": self.contradiction_score,
            "margin": self.margin,
            "counterfactual_family": self.counterfactual_family,
            "counterfactual_scores": self.counterfactual_scores,
            "current_utility_proxy": self.current_utility_proxy,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EvidenceState":
        return cls(
            claim_id=str(payload.get("claim_id", "")),
            items=[
                EvidenceItem(
                    name=str(item.get("name", "")),
                    score=float(item.get("score", 0.0)),
                    threshold=float(item.get("threshold", 0.85)),
                    importance=float(item.get("importance", 1.0)),
                    evidence_role=str(item.get("evidence_role", "")),
                )
                for item in payload.get("items", [])  # type: ignore[arg-type]
                if isinstance(item, Mapping)
            ],
            claim_score=float(payload.get("claim_score", 0.0)),
            contradiction_score=float(payload.get("contradiction_score", 0.0)),
            margin=float(payload.get("margin", 0.0)),
            counterfactual_family=str(payload.get("counterfactual_family", "")),
            counterfactual_scores={
                str(key): float(value)
                for key, value in dict(payload.get("counterfactual_scores", {}) or {}).items()
            },
            current_utility_proxy=(
                float(payload["current_utility_proxy"])
                if payload.get("current_utility_proxy") is not None
                else None
            ),
        )


def state_from_roles(
    *,
    claim_id: str,
    roles: Mapping[str, float],
    observed_score: float,
    threshold: float = 0.85,
    current_utility_proxy: float | None = None,
) -> EvidenceState:
    items = [
        EvidenceItem(
            name=role,
            evidence_role=role,
            score=float(observed_score),
            threshold=float(threshold),
            importance=float(importance),
        )
        for role, importance in roles.items()
    ]
    return EvidenceState(
        claim_id=claim_id,
        items=items,
        current_utility_proxy=current_utility_proxy,
    )


def merge_missing_names(items: Iterable[EvidenceItem]) -> List[str]:
    names: List[str] = []
    for item in items:
        role = item.evidence_role or item.name
        if role and role not in names:
            names.append(role)
    return names
