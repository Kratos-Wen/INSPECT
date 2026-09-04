"""Assistant-use evidence events for claim-level active inspection.

These records are deliberately view-agnostic.  A human Assistant trace should
not label robot view IDs.  It should identify which procedural claim was
uncertain, which evidence was missing, and which evidence became visible after
the user's observation behavior.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .evidence_view import EvidenceViewType


def _slug(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text).strip("_")


def _list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


@dataclass(frozen=True)
class AssistantEvidenceEvent:
    """One human-Assistant observation/correction event.

    The event can be a point feedback event or a before/after reveal event.  It
    is intentionally independent of robot view IDs.
    """

    video: str = ""
    event_id: str = ""
    before_frame: Optional[int] = None
    after_frame: Optional[int] = None
    before_time: Optional[float] = None
    after_time: Optional[float] = None
    state: str = ""
    claim_type: str = ""
    feedback: str = ""
    outcome: str = ""
    evidence_view_type: str = ""
    human_observation_behavior: str = ""
    missing_evidence: List[str] = field(default_factory=list)
    revealed_evidence: List[str] = field(default_factory=list)
    contradictory_evidence: List[str] = field(default_factory=list)
    reason: str = ""
    note: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AssistantEvidenceEvent":
        before_frame = payload.get("before_frame")
        after_frame = payload.get("after_frame")
        return cls(
            video=str(payload.get("video", "")),
            event_id=str(payload.get("event_id", "")),
            before_frame=(int(before_frame) if before_frame is not None and str(before_frame) != "" else None),
            after_frame=(int(after_frame) if after_frame is not None and str(after_frame) != "" else None),
            before_time=(float(payload.get("before_time")) if payload.get("before_time") not in (None, "") else None),
            after_time=(float(payload.get("after_time")) if payload.get("after_time") not in (None, "") else None),
            state=str(payload.get("state", "")).upper(),
            claim_type=_slug(payload.get("claim_type", "")),
            feedback=_slug(payload.get("feedback", "")),
            outcome=_slug(payload.get("outcome", "")),
            evidence_view_type=_slug(payload.get("evidence_view_type", "")),
            human_observation_behavior=_slug(payload.get("human_observation_behavior", "")),
            missing_evidence=_list(payload.get("missing_evidence")),
            revealed_evidence=_list(payload.get("revealed_evidence")),
            contradictory_evidence=_list(payload.get("contradictory_evidence")),
            reason=_slug(payload.get("reason", "")),
            note=str(payload.get("note", "")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass
class EvidenceRequirementProfile:
    """Claim-conditioned evidence-view requirements learned from Assistant use."""

    claim_view_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    view_type_weights: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def weight(self, claim_type: str, view_type: EvidenceViewType | str) -> float:
        claim = _slug(claim_type)
        view = view_type.value if isinstance(view_type, EvidenceViewType) else _slug(view_type)
        if claim and view in self.claim_view_weights.get(claim, {}):
            return max(0.0, float(self.claim_view_weights[claim][view]))
        if view in self.view_type_weights:
            return max(0.0, float(self.view_type_weights[view]))
        return 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_view_weights": self.claim_view_weights,
            "view_type_weights": self.view_type_weights,
            "counts": self.counts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceRequirementProfile":
        return cls(
            claim_view_weights={
                _slug(claim): {_slug(view): float(weight) for view, weight in dict(values).items()}
                for claim, values in dict(payload.get("claim_view_weights", {})).items()
            },
            view_type_weights={_slug(view): float(weight) for view, weight in dict(payload.get("view_type_weights", {})).items()},
            counts={
                _slug(claim): {_slug(view): int(count) for view, count in dict(values).items()}
                for claim, values in dict(payload.get("counts", {})).items()
            },
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    @classmethod
    def load(cls, path: Path | str) -> "EvidenceRequirementProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: Path | str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_assistant_evidence_events(path: Path | str) -> List[AssistantEvidenceEvent]:
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [AssistantEvidenceEvent.from_dict(record) for record in records if isinstance(record, dict)]


def profile_from_assistant_evidence_events(paths: Iterable[Path | str]) -> EvidenceRequirementProfile:
    """Build claim -> evidence-view weights from human Assistant-use events."""

    raw_counts: Dict[str, Dict[str, float]] = {}
    raw_trials: Dict[str, Dict[str, int]] = {}
    global_counts: Dict[str, float] = {}
    global_trials: Dict[str, int] = {}
    records_seen = 0
    records_used = 0
    for path in paths:
        for event in load_assistant_evidence_events(path):
            records_seen += 1
            view_type = _slug(event.evidence_view_type)
            claim = _slug(event.claim_type or event.state)
            if not view_type:
                continue
            success = bool(event.revealed_evidence) or event.outcome in {"verified", "rejected", "corrected"}
            strength = 1.0 if success else 0.35
            if event.contradictory_evidence:
                strength += 0.25
            if event.feedback in {"reject", "rejected", "correct", "corrected"}:
                strength += 0.20
            records_used += 1
            claim_key = claim or "__global__"
            raw_counts.setdefault(claim_key, {})
            raw_trials.setdefault(claim_key, {})
            raw_counts[claim_key][view_type] = raw_counts[claim_key].get(view_type, 0.0) + strength
            raw_trials[claim_key][view_type] = raw_trials[claim_key].get(view_type, 0) + 1
            global_counts[view_type] = global_counts.get(view_type, 0.0) + strength
            global_trials[view_type] = global_trials.get(view_type, 0) + 1

    claim_view_weights: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    for claim, per_view in raw_counts.items():
        total = sum(per_view.values())
        if total <= 0.0:
            continue
        mean = total / max(1, len(per_view))
        claim_view_weights[claim] = {}
        counts[claim] = {}
        for view, value in per_view.items():
            claim_view_weights[claim][view] = round(max(0.25, min(3.0, value / max(1e-6, mean))), 4)
            counts[claim][view] = int(raw_trials.get(claim, {}).get(view, 0))

    global_total = sum(global_counts.values())
    global_mean = global_total / max(1, len(global_counts))
    view_type_weights = {
        view: round(max(0.25, min(3.0, value / max(1e-6, global_mean))), 4)
        for view, value in global_counts.items()
    }

    return EvidenceRequirementProfile(
        claim_view_weights=claim_view_weights,
        view_type_weights=view_type_weights,
        counts=counts,
        metadata={
            "source": "assistant_evidence_events",
            "records_seen": records_seen,
            "records_used": records_used,
            "training_target": "claim_to_evidence_view_requirement",
            "contains_robot_view_labels": False,
        },
    )
