"""Random-start evaluation for INSPECT-Active transfer policies."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .active_selector import ActiveSelector
from .evidence_state import state_from_roles
from .ontology import normalize_claim, roles_for_claim
from .view_lattice import SIX_VIEWS, transition_cost


DEFAULT_VIEWS = tuple(sorted(SIX_VIEWS.keys()))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def utility(row: Mapping[str, str]) -> Optional[int]:
    value = str(row.get("human_utility_0_1_2", "")).strip()
    if value in {"0", "1", "2"}:
        return int(value)
    return None


def trial_rows(rows: Iterable[Dict[str, str]], steps: set[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
    by_trial: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if steps and row.get("target_step", "") not in steps:
            continue
        util = utility(row)
        if util is None:
            continue
        by_trial[row["trial_id"]][row["view_id"]] = row
    return dict(by_trial)


def trial_meta(view_rows: Mapping[str, Dict[str, str]]) -> Dict[str, str]:
    for row in view_rows.values():
        return dict(row)
    return {}


def oracle_target(view_rows: Mapping[str, Dict[str, str]], current: str, lambda_cost: float) -> tuple[str, float]:
    current_utility = float(utility(view_rows.get(current, {})) or 0)
    if current_utility >= 2.0:
        return current, current_utility
    best_view = current
    best_score = -1e9
    for view_id, row in view_rows.items():
        if view_id == current:
            continue
        score = float(utility(row) or 0) - lambda_cost * transition_cost(current, view_id)
        if score > best_score:
            best_score = score
            best_view = view_id
    return best_view, float(utility(view_rows.get(best_view, {})) or 0)


def evaluate_trial(
    *,
    selector: ActiveSelector,
    trial_id: str,
    view_rows: Mapping[str, Dict[str, str]],
    current_view: str,
    lambda_cost: float,
    current_decision_source: str = "oracle",
) -> Dict[str, Any]:
    meta = trial_meta(view_rows)
    claim_id = normalize_claim(meta.get("claim_id") or meta.get("target_step"))
    current_util = float(utility(view_rows.get(current_view, {})) or 0)
    observed_score = current_util / 2.0 if current_decision_source == "oracle" else 0.0
    evidence_state = state_from_roles(
        claim_id=claim_id,
        roles=roles_for_claim(claim_id),
        observed_score=observed_score,
        current_utility_proxy=current_util if current_decision_source == "oracle" else None,
    )
    decision = selector.select(current_view=current_view, evidence_state=evidence_state)
    selected = current_view if decision.action in {"stay", "defer"} else decision.selected_view
    selected_util = float(utility(view_rows.get(selected, {})) or 0)
    oracle_view, oracle_util = oracle_target(view_rows, current_view, lambda_cost)
    oracle_score = oracle_util - (0.0 if oracle_view == current_view else lambda_cost * transition_cost(current_view, oracle_view))
    selected_score = selected_util - (0.0 if selected == current_view else lambda_cost * transition_cost(current_view, selected))
    should_stay = current_util >= 2.0
    return {
        "trial_id": trial_id,
        "target_step": meta.get("target_step", ""),
        "claim_id": meta.get("claim_id", ""),
        "claim_outcome": meta.get("claim_outcome", ""),
        "relation_state": meta.get("relation_state", ""),
        "error_type": meta.get("error_type", ""),
        "current_view": current_view,
        "current_utility": current_util,
        "decision_action": decision.action,
        "selected_view": selected,
        "selected_utility": selected_util,
        "oracle_view": oracle_view,
        "oracle_utility": oracle_util,
        "utility_gain": selected_util - current_util,
        "oracle_regret": max(0.0, oracle_score - selected_score),
        "resolve_at_1": 1.0 if selected_util >= 2.0 else 0.0,
        "top1_oracle_hit": 1.0 if selected == oracle_view else 0.0,
        "stop_accuracy": 1.0 if should_stay and decision.action == "stay" else 0.0,
        "false_stop": 1.0 if (not should_stay and decision.action == "stay") else 0.0,
        "false_move": 1.0 if (should_stay and decision.action != "stay") else 0.0,
        "defer": 1.0 if decision.action == "defer" else 0.0,
        "no_improvement": 1.0 if selected_util <= current_util and not should_stay else 0.0,
        "cost_normalized_gain": (selected_util - current_util) / max(1e-9, transition_cost(current_view, selected)) if selected != current_view else 0.0,
        "policy_score": decision.score,
        "reason": decision.reason,
        "missing_evidence": ";".join(decision.missing_evidence),
        "ranked_views_json": json.dumps(decision.ranked_views, sort_keys=True),
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    keys = [
        "current_utility",
        "selected_utility",
        "oracle_utility",
        "utility_gain",
        "oracle_regret",
        "resolve_at_1",
        "top1_oracle_hit",
        "stop_accuracy",
        "false_stop",
        "false_move",
        "defer",
        "no_improvement",
        "cost_normalized_gain",
    ]
    return {"trials": len(rows), **{key: float(mean(float(row.get(key, 0.0)) for row in rows)) for key in keys}}
