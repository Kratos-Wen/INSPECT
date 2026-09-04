"""Infer relative view-change pseudo labels from assistant events."""

from __future__ import annotations

from typing import Mapping, Optional

from .ontology import normalize_key
from .view_lattice import RelativeAction


ACTION_ALIASES = {
    "left": RelativeAction.ORBIT_LEFT,
    "orbit_left": RelativeAction.ORBIT_LEFT,
    "right": RelativeAction.ORBIT_RIGHT,
    "orbit_right": RelativeAction.ORBIT_RIGHT,
    "up": RelativeAction.ORBIT_UP,
    "orbit_up": RelativeAction.ORBIT_UP,
    "down": RelativeAction.ORBIT_DOWN,
    "orbit_down": RelativeAction.ORBIT_DOWN,
    "left_up": RelativeAction.ORBIT_LEFT_UP,
    "up_left": RelativeAction.ORBIT_LEFT_UP,
    "orbit_left_up": RelativeAction.ORBIT_LEFT_UP,
    "orbit_up_left": RelativeAction.ORBIT_LEFT_UP,
    "right_up": RelativeAction.ORBIT_RIGHT_UP,
    "up_right": RelativeAction.ORBIT_RIGHT_UP,
    "orbit_right_up": RelativeAction.ORBIT_RIGHT_UP,
    "orbit_up_right": RelativeAction.ORBIT_RIGHT_UP,
    "left_down": RelativeAction.ORBIT_LEFT_DOWN,
    "down_left": RelativeAction.ORBIT_LEFT_DOWN,
    "orbit_left_down": RelativeAction.ORBIT_LEFT_DOWN,
    "orbit_down_left": RelativeAction.ORBIT_LEFT_DOWN,
    "right_down": RelativeAction.ORBIT_RIGHT_DOWN,
    "down_right": RelativeAction.ORBIT_RIGHT_DOWN,
    "orbit_right_down": RelativeAction.ORBIT_RIGHT_DOWN,
    "orbit_down_right": RelativeAction.ORBIT_RIGHT_DOWN,
    "defer": RelativeAction.DEFER,
    "non_transferable": RelativeAction.NON_TRANSFERABLE,
    "unknown": RelativeAction.UNKNOWN,
}


NON_TRANSFERABLE_REASONS = {
    "hand_occlusion",
    "object_occlusion",
    "occlusion",
    "blur",
    "motion_blur",
    "zoom",
    "closer",
    "move_closer",
    "scale_change",
    "crop",
}


def normalize_relative_action(value: object) -> str:
    key = normalize_key(value)
    return ACTION_ALIASES.get(key, RelativeAction.UNKNOWN)


def _metadata(payload: Mapping[str, object]) -> Mapping[str, object]:
    value = payload.get("metadata", {})
    return value if isinstance(value, Mapping) else {}


def infer_relative_action_from_payload(payload: Mapping[str, object]) -> str:
    """Return an action if one is explicitly recorded, otherwise UNKNOWN.

    The final method should not hallucinate a direction from a coarse reason
    such as "bad angle"; if the assistant event lacks a relative direction,
    it remains useful for the requirement profile but is not a positive action
    label.
    """

    meta = _metadata(payload)
    for key in (
        "relative_action",
        "effective_relative_action",
        "view_change",
        "view_delta",
        "action",
    ):
        if key in payload:
            action = normalize_relative_action(payload.get(key))
            if action != RelativeAction.UNKNOWN:
                return action
        if key in meta:
            action = normalize_relative_action(meta.get(key))
            if action != RelativeAction.UNKNOWN:
                return action
    behavior = normalize_key(payload.get("human_observation_behavior", ""))
    if behavior in ACTION_ALIASES:
        return ACTION_ALIASES[behavior]
    return RelativeAction.UNKNOWN


def transferability_reason(payload: Mapping[str, object]) -> Optional[str]:
    meta = _metadata(payload)
    for key in ("non_transferable_reason", "transferability_reason"):
        if key in payload and str(payload.get(key, "")).strip():
            return normalize_key(payload[key])
        if key in meta and str(meta.get(key, "")).strip():
            return normalize_key(meta[key])
    reason = normalize_key(payload.get("reason", ""))
    behavior = normalize_key(payload.get("human_observation_behavior", ""))
    if reason in NON_TRANSFERABLE_REASONS:
        return reason
    if behavior in NON_TRANSFERABLE_REASONS:
        return behavior
    return None
