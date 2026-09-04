"""Robot six-view lattice and relative view actions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple


class RelativeAction:
    ORBIT_LEFT = "orbit_left"
    ORBIT_RIGHT = "orbit_right"
    ORBIT_UP = "orbit_up"
    ORBIT_DOWN = "orbit_down"
    ORBIT_LEFT_UP = "orbit_left_up"
    ORBIT_RIGHT_UP = "orbit_right_up"
    ORBIT_LEFT_DOWN = "orbit_left_down"
    ORBIT_RIGHT_DOWN = "orbit_right_down"
    DEFER = "defer"
    NON_TRANSFERABLE = "non_transferable"
    UNKNOWN = "unknown"


ORBIT_ACTIONS: Tuple[str, ...] = (
    RelativeAction.ORBIT_LEFT,
    RelativeAction.ORBIT_RIGHT,
    RelativeAction.ORBIT_UP,
    RelativeAction.ORBIT_DOWN,
    RelativeAction.ORBIT_LEFT_UP,
    RelativeAction.ORBIT_RIGHT_UP,
    RelativeAction.ORBIT_LEFT_DOWN,
    RelativeAction.ORBIT_RIGHT_DOWN,
)


ACTION_VECTORS: Mapping[str, Tuple[float, float]] = {
    RelativeAction.ORBIT_LEFT: (-1.0, 0.0),
    RelativeAction.ORBIT_RIGHT: (1.0, 0.0),
    RelativeAction.ORBIT_UP: (0.0, 1.0),
    RelativeAction.ORBIT_DOWN: (0.0, -1.0),
    RelativeAction.ORBIT_LEFT_UP: (-1.0, 1.0),
    RelativeAction.ORBIT_RIGHT_UP: (1.0, 1.0),
    RelativeAction.ORBIT_LEFT_DOWN: (-1.0, -1.0),
    RelativeAction.ORBIT_RIGHT_DOWN: (1.0, -1.0),
}


@dataclass(frozen=True)
class ViewNode:
    view_id: str
    yaw: float
    elevation: float
    label: str = ""


# Matches the existing ordered robot dataset:
# V0 left-lower, V1 middle-lower, V2 right-lower,
# V3 right-upper, V4 middle-upper, V5 left-upper.
SIX_VIEWS: Dict[str, ViewNode] = {
    "V0": ViewNode("V0", yaw=-45.0, elevation=30.0, label="left_lower"),
    "V1": ViewNode("V1", yaw=0.0, elevation=30.0, label="middle_lower"),
    "V2": ViewNode("V2", yaw=45.0, elevation=30.0, label="right_lower"),
    "V3": ViewNode("V3", yaw=45.0, elevation=60.0, label="right_upper"),
    "V4": ViewNode("V4", yaw=0.0, elevation=60.0, label="middle_upper"),
    "V5": ViewNode("V5", yaw=-45.0, elevation=60.0, label="left_upper"),
}


def candidate_views(current_view: str, views: Mapping[str, ViewNode] | None = None) -> List[str]:
    table = views or SIX_VIEWS
    return [view_id for view_id in sorted(table) if view_id != current_view]


def delta(src: str, dst: str, views: Mapping[str, ViewNode] | None = None) -> Tuple[float, float]:
    table = views or SIX_VIEWS
    if src not in table:
        raise KeyError(f"Unknown source view: {src}")
    if dst not in table:
        raise KeyError(f"Unknown target view: {dst}")
    return (table[dst].yaw - table[src].yaw, table[dst].elevation - table[src].elevation)


def transition_cost(src: str, dst: str, views: Mapping[str, ViewNode] | None = None) -> float:
    dyaw, delev = delta(src, dst, views)
    return abs(dyaw) / 45.0 + abs(delev) / 30.0


def direction_token(src: str, dst: str, views: Mapping[str, ViewNode] | None = None) -> str:
    dyaw, delev = delta(src, dst, views)
    yaw_token = ""
    elev_token = ""
    if dyaw < -1e-6:
        yaw_token = "left"
    elif dyaw > 1e-6:
        yaw_token = "right"
    if delev > 1e-6:
        elev_token = "up"
    elif delev < -1e-6:
        elev_token = "down"
    if yaw_token and elev_token:
        return f"orbit_{yaw_token}_{elev_token}"
    if yaw_token:
        return f"orbit_{yaw_token}"
    if elev_token:
        return f"orbit_{elev_token}"
    return RelativeAction.DEFER


def _norm(vec: Tuple[float, float]) -> Tuple[float, float]:
    x, y = vec
    length = math.sqrt(x * x + y * y)
    if length <= 1e-9:
        return (0.0, 0.0)
    return (x / length, y / length)


def action_compatibility(action: str, src: str, dst: str, views: Mapping[str, ViewNode] | None = None) -> float:
    """Directional compatibility between an action token and a lattice move."""

    if action not in ACTION_VECTORS:
        return 0.0
    dyaw, delev = delta(src, dst, views)
    if abs(dyaw) <= 1e-9 and abs(delev) <= 1e-9:
        return 0.0
    candidate_vec = _norm((dyaw / 45.0, delev / 30.0))
    action_vec = _norm(ACTION_VECTORS[action])
    dot = candidate_vec[0] * action_vec[0] + candidate_vec[1] * action_vec[1]
    # A one-sided directional kernel prevents orthogonal or opposite actions
    # from creating evidence through probability-mass averaging.  The fourth
    # power keeps adjacent 45-degree tokens compatible while preserving exact
    # lattice modes.
    return max(0.0, dot) ** 4


def serialize_views(views: Mapping[str, ViewNode] | None = None) -> List[dict]:
    table = views or SIX_VIEWS
    return [
        {"view_id": node.view_id, "yaw": node.yaw, "elevation": node.elevation, "label": node.label}
        for node in table.values()
    ]


def load_views(records: Iterable[Mapping[str, object]]) -> Dict[str, ViewNode]:
    table: Dict[str, ViewNode] = {}
    for record in records:
        view_id = str(record.get("view_id", "")).strip()
        if not view_id:
            continue
        table[view_id] = ViewNode(
            view_id=view_id,
            yaw=float(record.get("yaw", 0.0)),
            elevation=float(record.get("elevation", 0.0)),
            label=str(record.get("label", "")),
        )
    return table or dict(SIX_VIEWS)
