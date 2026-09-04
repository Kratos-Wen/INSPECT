"""Audit typed evidence transport using assistant videos only."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspect_system.active_view.decidability_transport import (
    _event_weight,
    _typed_event_claim,
    parent_claim_id,
)
from inspect_system.active_view.ontology import normalize_key
from inspect_system.active_view.view_lattice import ORBIT_ACTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concentration", type=float, default=2.0)
    return parser.parse_args()


def normalize(values: Mapping[str, float], support: Sequence[str]) -> Dict[str, float]:
    total = sum(max(0.0, float(values.get(key, 0.0))) for key in support)
    if total <= 1e-12:
        return {key: 1.0 / len(support) for key in support}
    return {key: max(0.0, float(values.get(key, 0.0))) / total for key in support}


def update(
    prior: Mapping[str, float],
    values: Mapping[str, float],
    support: Sequence[str],
    concentration: float,
) -> Dict[str, float]:
    mass = sum(max(0.0, float(values.get(key, 0.0))) for key in support)
    raw = {
        key: concentration * float(prior.get(key, 0.0))
        + max(0.0, float(values.get(key, 0.0)))
        for key in support
    }
    return normalize(raw, support) if mass > 0.0 else normalize(prior, support)


def weighted_metrics(rows: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    items = list(rows)
    mass = sum(float(row["weight"]) for row in items)
    return {
        key: sum(float(row["weight"]) * float(row[key]) for row in items)
        / max(1e-12, mass)
        for key in ("nll", "top1", "top2", "target_probability")
    } | {"weight": mass, "events": len(items)}


def action_key(event: Mapping[str, Any], typed: bool, role: str) -> str:
    claim = _typed_event_claim(event) if typed else parent_claim_id(event.get("claim_id"))
    return f"{claim}|{role}"


def evaluate_actions(
    events: Sequence[Mapping[str, Any]], concentration: float
) -> Dict[str, Any]:
    results = {"generic": [], "typed_backoff": []}
    videos = sorted({str(event.get("video", "")) for event in events})
    for held_out in videos:
        train = [event for event in events if str(event.get("video", "")) != held_out]
        tables: Dict[str, Dict[str, Dict[str, float]]] = {
            "generic": defaultdict(lambda: defaultdict(float)),
            "typed": defaultdict(lambda: defaultdict(float)),
        }
        for event in train:
            weight = float(event["_weight"])
            action = str(event["_target"])
            role = normalize_key(event.get("evidence_role", ""))
            for typed, name in ((False, "generic"), (True, "typed")):
                for role_key in (role, "__any__"):
                    tables[name][action_key(event, typed, role_key)][action] += weight
        for event in events:
            if str(event.get("video", "")) != held_out:
                continue
            role = normalize_key(event.get("evidence_role", ""))
            target = str(event["_target"])
            uniform = {action: 1.0 / len(ORBIT_ACTIONS) for action in ORBIT_ACTIONS}
            generic_any = update(
                uniform,
                tables["generic"].get(action_key(event, False, "__any__"), {}),
                ORBIT_ACTIONS,
                concentration,
            )
            generic = update(
                generic_any,
                tables["generic"].get(action_key(event, False, role), {}),
                ORBIT_ACTIONS,
                concentration,
            )
            typed_any = update(
                generic,
                tables["typed"].get(action_key(event, True, "__any__"), {}),
                ORBIT_ACTIONS,
                concentration,
            )
            typed = update(
                typed_any,
                tables["typed"].get(action_key(event, True, role), {}),
                ORBIT_ACTIONS,
                concentration,
            )
            for name, distribution in (("generic", generic), ("typed_backoff", typed)):
                ranked = sorted(
                    ORBIT_ACTIONS,
                    key=lambda action: (distribution[action], action),
                    reverse=True,
                )
                probability = max(1e-12, distribution[target])
                results[name].append(
                    {
                        "weight": float(event["_weight"]),
                        "nll": -math.log(probability),
                        "top1": float(target == ranked[0]),
                        "top2": float(target in ranked[:2]),
                        "target_probability": probability,
                    }
                )
    return {name: weighted_metrics(rows) for name, rows in results.items()}


def evaluate_requirements(
    events: Sequence[Mapping[str, Any]], concentration: float
) -> Dict[str, Any]:
    roles = sorted({normalize_key(event.get("evidence_role", "")) for event in events})
    results = {"generic": [], "typed_backoff": []}
    videos = sorted({str(event.get("video", "")) for event in events})
    for held_out in videos:
        train = [event for event in events if str(event.get("video", "")) != held_out]
        tables: Dict[str, Dict[str, Dict[str, float]]] = {
            "generic": defaultdict(lambda: defaultdict(float)),
            "typed": defaultdict(lambda: defaultdict(float)),
        }
        episode_mass: Dict[str, float] = defaultdict(float)
        for event in train:
            episode_mass[str(event["_episode"])] += float(event["_weight"])
        for event in train:
            weight = float(event["_weight"]) / max(
                1e-12, episode_mass[str(event["_episode"])]
            )
            role = normalize_key(event.get("evidence_role", ""))
            tables["generic"][parent_claim_id(event.get("claim_id"))][role] += weight
            tables["typed"][_typed_event_claim(event)][role] += weight
        for event in events:
            if str(event.get("video", "")) != held_out:
                continue
            target = normalize_key(event.get("evidence_role", ""))
            uniform = {role: 1.0 / len(roles) for role in roles}
            generic = update(
                uniform,
                tables["generic"].get(parent_claim_id(event.get("claim_id")), {}),
                roles,
                concentration,
            )
            typed = update(
                generic,
                tables["typed"].get(_typed_event_claim(event), {}),
                roles,
                concentration,
            )
            for name, distribution in (("generic", generic), ("typed_backoff", typed)):
                ranked = sorted(
                    roles,
                    key=lambda role: (distribution[role], role),
                    reverse=True,
                )
                probability = max(1e-12, distribution[target])
                results[name].append(
                    {
                        "weight": float(event["_weight"]),
                        "nll": -math.log(probability),
                        "top1": float(target == ranked[0]),
                        "top2": float(target in ranked[:2]),
                        "target_probability": probability,
                    }
                )
    return {name: weighted_metrics(rows) for name, rows in results.items()}


def main() -> None:
    args = parse_args()
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    if payload.get("uses_robot_view_training"):
        raise RuntimeError("Assistant calibration report uses robot views.")

    action_events = []
    for event in payload.get("trainable_events") or []:
        if not isinstance(event, Mapping):
            continue
        action = normalize_key(event.get("relative_action", ""))
        weight = _event_weight(event, requirement=False)
        if action in ORBIT_ACTIONS and bool(event.get("transferable", True)) and weight > 0.0:
            action_events.append({**event, "_target": action, "_weight": weight})

    requirement_events = []
    for index, event in enumerate(payload.get("requirement_events") or []):
        if not isinstance(event, Mapping):
            continue
        weight = _event_weight(event, requirement=True)
        role = normalize_key(event.get("evidence_role", ""))
        if not role or weight <= 0.0:
            continue
        episode = "|".join(
            (
                str(event.get("video", "")),
                str(event.get("after_frame", "")),
                _typed_event_claim(event),
            )
        ) or str(index)
        requirement_events.append(
            {**event, "_episode": episode, "_weight": weight}
        )

    result = {
        "protocol": {
            "split": "leave-one-assistant-video-out",
            "uses_robot_views": False,
            "uses_robot_utility_labels": False,
            "uses_candidate_view_images": False,
            "concentration": float(args.concentration),
            "source": str(args.report),
        },
        "relative_reveal": evaluate_actions(action_events, float(args.concentration)),
        "evidence_requirement": evaluate_requirements(
            requirement_events, float(args.concentration)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
