"""Causal transfer gate for human-to-robot view-policy supervision.

Human assistant traces can resolve uncertainty for many reasons: the state may
change, a hand may move away, focus may recover, or the user may change the
view. Only the last case is clean supervision for a robot view policy. This
module assigns each mined event a transfer type and a training weight.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping

from .ontology import normalize_key
from .view_lattice import ORBIT_ACTIONS, RelativeAction


class TransferType:
    VIEW_TRANSFERABLE = "view_transferable"
    OCCLUSION_ONLY = "occlusion_only"
    ZOOM_FOCUS_ONLY = "zoom_focus_only"
    STATE_CHANGED = "state_changed"
    OBJECT_REORIENTED_NONROBOT = "object_reoriented_nonrobot"
    AMBIGUOUS = "ambiguous"


OCCLUSION_KEYS = {
    "hand_occlusion",
    "object_occlusion",
    "occlusion",
}

ZOOM_FOCUS_KEYS = {
    "blur",
    "blur_or_out_of_focus",
    "out_of_focus",
    "focus",
    "glare",
    "zoom",
    "move_closer",
    "scale_change",
    "scale_or_zoom_dominated",
}


@dataclass(frozen=True)
class TransferGateDecision:
    transfer_type: str
    transfer_weight: float
    label_confidence: float
    reasons: List[str] = field(default_factory=list)

    @property
    def transferable(self) -> bool:
        return self.transfer_weight > 0.0 and self.transfer_type == TransferType.VIEW_TRANSFERABLE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = payload.get("metadata", {})
    return meta if isinstance(meta, Mapping) else {}


def _state_changed(payload: Mapping[str, Any]) -> bool:
    meta = _metadata(payload)
    previous = normalize_key(meta.get("previous_state", ""))
    resolved = normalize_key(meta.get("resolved_state", ""))
    return bool(previous and resolved and previous != resolved)


def _base_reason(payload: Mapping[str, Any], flow_result: Mapping[str, Any]) -> str:
    reason = normalize_key(payload.get("reason", ""))
    if reason:
        return reason
    return normalize_key(flow_result.get("reason", ""))


def causal_transfer_gate(
    *,
    payload: Mapping[str, Any],
    relative_action: str,
    flow_result: Mapping[str, Any] | None = None,
    allow_state_change_events: bool = False,
    allow_occlusion_events: bool = False,
    allow_object_rotation_transfer: bool = False,
) -> TransferGateDecision:
    """Classify whether an event is causal supervision for view transfer."""

    flow = flow_result or {}
    reasons: List[str] = []
    action = normalize_key(relative_action)
    reason = _base_reason(payload, flow)
    confidence = float(flow.get("confidence", _metadata(payload).get("relative_action_confidence", 0.0)) or 0.0)
    confidence = max(0.0, min(1.0, confidence))

    if _state_changed(payload) and not allow_state_change_events:
        return TransferGateDecision(
            transfer_type=TransferType.STATE_CHANGED,
            transfer_weight=0.0,
            label_confidence=0.0,
            reasons=["procedural_state_changed"],
        )

    if reason in OCCLUSION_KEYS and not allow_occlusion_events:
        return TransferGateDecision(
            transfer_type=TransferType.OCCLUSION_ONLY,
            transfer_weight=0.0,
            label_confidence=0.0,
            reasons=[reason],
        )

    if reason in ZOOM_FOCUS_KEYS or normalize_key(flow.get("reason", "")) in ZOOM_FOCUS_KEYS:
        return TransferGateDecision(
            transfer_type=TransferType.ZOOM_FOCUS_ONLY,
            transfer_weight=0.0,
            label_confidence=0.0,
            reasons=[reason or normalize_key(flow.get("reason", ""))],
        )

    if action not in ORBIT_ACTIONS:
        return TransferGateDecision(
            transfer_type=TransferType.AMBIGUOUS,
            transfer_weight=0.0,
            label_confidence=0.0,
            reasons=[normalize_key(flow.get("reason", "")) or "missing_relative_action"],
        )

    motion_source = normalize_key(flow.get("motion_source", ""))
    if motion_source == "object_relative_motion":
        if not allow_object_rotation_transfer:
            return TransferGateDecision(
                transfer_type=TransferType.OBJECT_REORIENTED_NONROBOT,
                transfer_weight=0.0,
                label_confidence=0.0,
                reasons=["object_relative_motion"],
            )
        reasons.append("object_relative_motion_allowed")
        return TransferGateDecision(
            transfer_type=TransferType.VIEW_TRANSFERABLE,
            transfer_weight=max(0.05, 0.35 * confidence),
            label_confidence=confidence,
            reasons=reasons,
        )

    if motion_source == "mixed_motion":
        reasons.append("mixed_motion")
        return TransferGateDecision(
            transfer_type=TransferType.VIEW_TRANSFERABLE,
            transfer_weight=max(0.05, 0.55 * confidence),
            label_confidence=confidence,
            reasons=reasons,
        )

    if motion_source in {"weak_motion", "unknown"}:
        return TransferGateDecision(
            transfer_type=TransferType.AMBIGUOUS,
            transfer_weight=0.0,
            label_confidence=0.0,
            reasons=[motion_source],
        )

    # camera_motion, or explicit orbit action without a contradictory source.
    return TransferGateDecision(
        transfer_type=TransferType.VIEW_TRANSFERABLE,
        transfer_weight=max(0.05, confidence),
        label_confidence=confidence,
        reasons=reasons or ["camera_or_effective_view_motion"],
    )
