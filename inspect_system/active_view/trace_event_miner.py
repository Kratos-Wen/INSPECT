"""Load assistant events and convert them into relative-view training examples."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .ontology import infer_role_from_fields, normalize_claim, normalize_key
from .relative_view_labeler import infer_relative_action_from_payload
from .view_lattice import RelativeAction


@dataclass
class MinedEvent:
    event_id: str
    video: str
    claim_id: str
    evidence_role: str
    relative_action: str
    label: int
    transferable: bool
    transfer_type: str = ""
    transfer_weight: float = 0.0
    label_confidence: float = 1.0
    action_confidence: float = 1.0
    evidence_stability: float = 1.0
    evidence_importance: float = 1.0
    before_frame: int | None = None
    after_frame: int | None = None
    reason: str = ""
    transferability_reason: str = ""
    source_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _read_json_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def discover_event_files(paths: Sequence[str | Path]) -> List[Path]:
    files: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*assistant_evidence_events.jsonl")))
            files.extend(sorted(path.glob("*assistant_evidence_events.json")))
        elif path.exists():
            files.append(path)
    seen = set()
    unique: List[Path] = []
    for path in files:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def is_successful_resolution(payload: Mapping[str, object]) -> bool:
    outcome = normalize_key(payload.get("outcome", ""))
    feedback = normalize_key(payload.get("feedback", ""))
    revealed = payload.get("revealed_evidence")
    contradictory = payload.get("contradictory_evidence")
    has_reveal = bool(revealed) or bool(contradictory)
    return outcome in {"verified", "rejected", "corrected", "supported", "contradicted"} or feedback in {
        "accepted",
        "corrected",
        "rejected",
    } or has_reveal


def is_failed_resolution(payload: Mapping[str, object]) -> bool:
    outcome = normalize_key(payload.get("outcome", ""))
    return outcome in {"uncertain", "still_uncertain", "unresolved", "failed"}


def mine_relative_view_events(paths: Sequence[str | Path]) -> List[MinedEvent]:
    mined: List[MinedEvent] = []
    for path in discover_event_files(paths):
        for index, payload in enumerate(_read_json_records(path)):
            relative_action = infer_relative_action_from_payload(payload)
            meta = payload.get("metadata", {})
            meta_mapping = meta if isinstance(meta, Mapping) else {}
            transfer_type = str(meta_mapping.get("transfer_type", ""))
            transfer_reason = str(meta_mapping.get("transferability_reason", ""))
            transfer_weight = 0.0
            if meta_mapping.get("transfer_weight") not in (None, ""):
                transfer_weight = float(meta_mapping.get("transfer_weight", 0.0))
            legacy_transferable = bool(meta_mapping.get("transferable", False))
            transferable = transfer_weight > 0.0 or legacy_transferable
            label = 1 if is_successful_resolution(payload) else 0
            if is_failed_resolution(payload):
                label = 0
            claim_id = normalize_claim(payload.get("claim_type") or payload.get("claim_id") or payload.get("state"))
            evidence_role = infer_role_from_fields(
                claim_id=claim_id,
                evidence_view_type=payload.get("evidence_view_type", ""),
                reason=payload.get("reason", ""),
                missing_evidence=payload.get("missing_evidence"),
                revealed_evidence=payload.get("revealed_evidence"),
            )
            confidence = 1.0
            if isinstance(meta_mapping, Mapping) and meta_mapping.get("relative_action_confidence") not in (None, ""):
                confidence = float(meta_mapping.get("relative_action_confidence", 1.0))
            action_confidence = confidence
            if transfer_weight <= 0.0 and legacy_transferable:
                transfer_weight = confidence
            evidence_stability = 1.0
            evidence_importance = 1.0
            if isinstance(meta_mapping, Mapping):
                if meta_mapping.get("action_confidence") not in (None, ""):
                    action_confidence = float(meta_mapping.get("action_confidence", action_confidence))
                if meta_mapping.get("label_confidence") not in (None, ""):
                    confidence = float(meta_mapping.get("label_confidence", confidence))
                if meta_mapping.get("evidence_stability") not in (None, ""):
                    evidence_stability = float(meta_mapping.get("evidence_stability", 1.0))
                if meta_mapping.get("evidence_importance") not in (None, ""):
                    evidence_importance = float(meta_mapping.get("evidence_importance", 1.0))
                if meta_mapping.get("transferable") is not None:
                    transferable = bool(meta_mapping.get("transferable")) and transfer_weight > 0.0
            mined.append(
                MinedEvent(
                    event_id=str(payload.get("event_id", f"{path.stem}_{index}")),
                    video=str(payload.get("video", "")),
                    claim_id=claim_id,
                    evidence_role=evidence_role,
                    relative_action=relative_action,
                    label=label,
                    transferable=transferable,
                    transfer_type=transfer_type,
                    transfer_weight=transfer_weight,
                    label_confidence=confidence,
                    action_confidence=action_confidence,
                    evidence_stability=evidence_stability,
                    evidence_importance=evidence_importance,
                    before_frame=int(payload["before_frame"]) if payload.get("before_frame") not in (None, "") else None,
                    after_frame=int(payload["after_frame"]) if payload.get("after_frame") not in (None, "") else None,
                    reason=str(payload.get("reason", "")),
                    transferability_reason=transfer_reason,
                    source_path=str(path),
                    metadata=dict(meta_mapping) if isinstance(meta_mapping, Mapping) else {},
                )
            )
    return mined


def training_events_only(events: Iterable[MinedEvent]) -> List[MinedEvent]:
    return [
        event
        for event in events
        if not event.metadata.get("not_for_view_policy_training", False)
        and not event.metadata.get("timeline_endpoint_transition_diagnostic", False)
        and event.transferable
        and event.transfer_weight > 0.0
        and event.relative_action not in {RelativeAction.UNKNOWN, RelativeAction.NON_TRANSFERABLE, RelativeAction.DEFER}
    ]
