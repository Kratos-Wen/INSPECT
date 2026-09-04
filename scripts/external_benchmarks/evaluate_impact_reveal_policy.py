"""Evaluate assistant-derived reveal policies on frozen IMPACT view scores.

Only the four static cameras with known relative orientation are used as the
action lattice. Candidate-view scores are hidden until after selection.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspect_system.active_view.active_selector import ActiveSelector
from inspect_system.active_view.decidability_gated_transport import (
    ClaimStructuredEvidenceTransportSelector,
)
from inspect_system.active_view.evidence_state import EvidenceItem, EvidenceState
from inspect_system.active_view.ontology import normalize_claim, normalize_key
from inspect_system.active_view.ray_visibility import candidate_role_factors
from inspect_system.active_view.requirement_model import (
    RequirementCalibration,
    load_requirement_counts,
    mix_requirement_weights,
)
from inspect_system.active_view.monotone_evidence_transport import (
    MonotoneClaimEvidenceTransportSelector,
)
from inspect_system.active_view.reveal_model import PriorTableRevealModel
from inspect_system.active_view.view_lattice import ORBIT_ACTIONS, ViewNode


VIEWS = {
    "front": ViewNode("front", yaw=0.0, elevation=0.0),
    "left": ViewNode("left", yaw=-45.0, elevation=0.0),
    "right": ViewNode("right", yaw=45.0, elevation=0.0),
    "top": ViewNode("top", yaw=0.0, elevation=30.0),
}
GENERIC_ROLE = "claim_disambiguation_view"

IMPACT_ROLE_ALIASES = {
    "identity": "identity_disambiguation_view",
    "alignment": "slot_relation_view",
    "containment": "insertion_verification_view",
    "boundary_visibility": "gap_visibility_view",
    "occlusion": "occlusion_recovery_view",
}


def semantic_claim_id(row: Mapping[str, Any]) -> str:
    """Remove component identity while retaining the procedural predicate."""
    claim = normalize_claim(row.get("claim_id", ""))
    for suffix in ("installed_correctly", "installed_incorrectly", "not_installed"):
        if claim == suffix or claim.endswith(f"_{suffix}"):
            return f"component_{suffix}"
    return claim


def evidence_roles(row: Mapping[str, Any]) -> List[str]:
    raw = row.get("evidence_roles", "")
    values = raw if isinstance(raw, list) else str(raw).split(";")
    roles: List[str] = []
    for value in values:
        key = normalize_key(value)
        role = IMPACT_ROLE_ALIASES.get(key, key)
        if role and role not in roles:
            roles.append(role)
    return roles or [GENERIC_ROLE]


def evidence_state_from_row(
    row: Mapping[str, Any],
    *,
    cutoff: float,
    semantics: str,
) -> EvidenceState:
    confidence = max(0.0, float(row.get("decision_confidence", 0.0)))
    if semantics == "generic":
        claim_id = "state_validity"
        roles = [GENERIC_ROLE]
    else:
        claim_id = semantic_claim_id(row)
        roles = evidence_roles(row)
    role_importance = 1.0 / len(roles)
    counterfactual_scores = {
        "spatial_relation": max(0.0, float(row.get("p_installed_wrongly", 0.0))),
        "state_absence": max(0.0, float(row.get("p_not_installed", 0.0))),
    }
    counterfactual_family = ""
    if max(counterfactual_scores.values(), default=0.0) > 1e-12:
        counterfactual_family = max(
            counterfactual_scores,
            key=lambda key: (counterfactual_scores[key], key),
        )
    return EvidenceState(
        claim_id=claim_id,
        counterfactual_family=counterfactual_family,
        counterfactual_scores=counterfactual_scores,
        margin=float(row.get("support_margin", 0.0)),
        items=[
            EvidenceItem(
                name=role,
                evidence_role=role,
                score=min(confidence, cutoff),
                threshold=cutoff,
                importance=role_importance,
            )
            for role in roles
        ],
    )


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def event_key(row: Mapping[str, Any]) -> Tuple[str, int, int]:
    return (str(row["recording_id"]), int(row["source_frame"]), int(row["component_id"]))


def correct_direction_margin(row: Mapping[str, Any]) -> float:
    sign = -1.0 if str(row.get("semantic_outcome", "")).lower() == "contradicted" else 1.0
    return sign * float(row.get("support_margin", 0.0))


def aggregate(rows: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    items = list(rows)
    return {
        "trials": len(items),
        "selected_utility": mean(float(row["selected_utility"]) for row in items),
        "gain": mean(float(row["gain"]) for row in items),
        "resolve_at_1": mean(float(row["resolve_at_1"]) for row in items),
        "regret": mean(float(row["regret"]) for row in items),
        "moved": mean(float(row["moved"]) for row in items),
        "selected_counterfactual_margin": mean(
            float(row["selected_counterfactual_margin"]) for row in items
        ),
        "counterfactual_separation_gain": mean(
            float(row["counterfactual_separation_gain"]) for row in items
        ),
    }


def bootstrap_delta(
    candidate: List[Mapping[str, Any]], baseline: List[Mapping[str, Any]], samples: int, seed: int
) -> Dict[str, List[float] | float]:
    base_index = {(event_key(row), row["current_view"]): row for row in baseline}
    paired = [(row, base_index[(event_key(row), row["current_view"])]) for row in candidate]
    by_recording: Dict[str, List[Tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for pair in paired:
        by_recording[str(pair[0]["recording_id"])].append(pair)
    recordings = sorted(by_recording)
    rng = random.Random(seed)
    draws = []
    counterfactual_draws = []
    for _ in range(samples):
        sampled = [rng.choice(recordings) for _ in recordings]
        pairs = [pair for recording in sampled for pair in by_recording[recording]]
        draws.append(mean(float(a["selected_utility"]) - float(b["selected_utility"]) for a, b in pairs))
        counterfactual_draws.append(
            mean(
                float(a["counterfactual_separation_gain"])
                - float(b["counterfactual_separation_gain"])
                for a, b in pairs
            )
        )
    ordered = sorted(draws)
    counterfactual_ordered = sorted(counterfactual_draws)
    point = mean(float(a["selected_utility"]) - float(b["selected_utility"]) for a, b in paired)
    counterfactual_point = mean(
        float(a["counterfactual_separation_gain"])
        - float(b["counterfactual_separation_gain"])
        for a, b in paired
    )
    return {
        "utility_delta": point,
        "ci95": [ordered[int(0.025 * (len(ordered) - 1))], ordered[int(0.975 * (len(ordered) - 1))]],
        "positive_probability": mean(value > 0 for value in draws),
        "counterfactual_separation_delta": counterfactual_point,
        "counterfactual_separation_ci95": [
            counterfactual_ordered[int(0.025 * (len(counterfactual_ordered) - 1))],
            counterfactual_ordered[int(0.975 * (len(counterfactual_ordered) - 1))],
        ],
        "counterfactual_separation_positive_probability": mean(
            value > 0 for value in counterfactual_draws
        ),
    }


def policy_rows(
    grouped: Mapping[Tuple[str, int, int], Mapping[str, Mapping[str, Any]]],
    model: PriorTableRevealModel,
    *,
    cutoff: float,
    lambda_cost: float,
    tau_view: float,
    semantics: str,
    use_evidence_role_ray_kernel: bool = False,
    requirement_counts: Mapping[str, Mapping[str, float]] | None = None,
    requirement_calibration: RequirementCalibration | None = None,
    uniform_requirement: bool = False,
    selector_mode: str = "relative",
) -> List[Dict[str, Any]]:
    if selector_mode == "monotone":
        selector = MonotoneClaimEvidenceTransportSelector(
            model=model,
            views=VIEWS,
            lambda_cost=lambda_cost,
            tau_view=tau_view,
            geometry_loss_cap=1.0,
        )
    elif selector_mode == "relative":
        selector = ActiveSelector(
            model=model,
            views=VIEWS,
            lambda_cost=lambda_cost,
            tau_view=tau_view,
        )
    elif selector_mode == "claim_structured":
        selector = ClaimStructuredEvidenceTransportSelector(
            model=model,
            views=VIEWS,
            lambda_cost=lambda_cost,
            tau_view=tau_view,
            geometry_loss_cap=1.0,
        )
    else:
        raise ValueError(f"Unknown selector mode: {selector_mode}")
    rows = []
    for key, by_view in sorted(grouped.items()):
        oracle = max(int(by_view[view]["utility"]) for view in VIEWS)
        for current_view in VIEWS:
            current = by_view[current_view]
            current_utility = int(current["utility"])
            if str(current["claim_prediction"]) != "insufficient":
                selected_view = current_view
            else:
                state = evidence_state_from_row(
                    current,
                    cutoff=cutoff,
                    semantics=semantics,
                )
                if requirement_counts is not None:
                    calibration = requirement_calibration or RequirementCalibration()
                    state = mix_requirement_weights(
                        state,
                        requirement_counts,
                        beta=calibration.beta,
                        strength=calibration.strength,
                        temperature=calibration.temperature,
                        uniform=uniform_requirement,
                        preserve_scores=True,
                    )
                candidate_context = None
                if use_evidence_role_ray_kernel:
                    candidate_context = {
                        "candidate_role_factors": candidate_role_factors(
                            normal_camera=None,
                            current_view=current_view,
                            views=VIEWS,
                        )
                    }
                result = selector.select(
                    current_view=current_view,
                    evidence_state=state,
                    candidate_context=candidate_context,
                )
                selected_view = result.selected_view if result.action == "move" else current_view
            selected_row = by_view[selected_view]
            selected = int(selected_row["utility"])
            current_margin = correct_direction_margin(current)
            selected_margin = correct_direction_margin(selected_row)
            rows.append(
                {
                    "recording_id": key[0],
                    "source_frame": key[1],
                    "component_id": key[2],
                    "current_view": current_view,
                    "selected_view": selected_view,
                    "current_utility": current_utility,
                    "selected_utility": selected,
                    "gain": selected - current_utility,
                    "resolve_at_1": int(selected == 2),
                    "regret": oracle - selected,
                    "moved": int(selected_view != current_view),
                    "current_counterfactual_margin": current_margin,
                    "selected_counterfactual_margin": selected_margin,
                    "counterfactual_separation_gain": selected_margin - current_margin,
                }
            )
    return rows


def fixed_view_rows(
    grouped: Mapping[Tuple[str, int, int], Mapping[str, Mapping[str, Any]]],
    target_view: str,
) -> List[Dict[str, Any]]:
    """Evaluate a fixed target camera with the same sufficient-view STOP rule."""
    if target_view not in VIEWS:
        raise KeyError(f"Unknown fixed target view: {target_view}")
    rows: List[Dict[str, Any]] = []
    for key, by_view in sorted(grouped.items()):
        oracle = max(int(by_view[view]["utility"]) for view in VIEWS)
        for current_view in VIEWS:
            current = by_view[current_view]
            current_utility = int(current["utility"])
            selected_view = (
                target_view
                if str(current["claim_prediction"]) == "insufficient"
                else current_view
            )
            selected_row = by_view[selected_view]
            selected = int(selected_row["utility"])
            current_margin = correct_direction_margin(current)
            selected_margin = correct_direction_margin(selected_row)
            rows.append(
                {
                    "recording_id": key[0],
                    "source_frame": key[1],
                    "component_id": key[2],
                    "current_view": current_view,
                    "selected_view": selected_view,
                    "current_utility": current_utility,
                    "selected_utility": selected,
                    "gain": selected - current_utility,
                    "resolve_at_1": int(selected == 2),
                    "regret": oracle - selected,
                    "moved": int(selected_view != current_view),
                    "current_counterfactual_margin": current_margin,
                    "selected_counterfactual_margin": selected_margin,
                    "counterfactual_separation_gain": selected_margin - current_margin,
                }
            )
    return rows


def current_rows(grouped: Mapping[Tuple[str, int, int], Mapping[str, Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    for key, by_view in sorted(grouped.items()):
        oracle = max(int(by_view[view]["utility"]) for view in VIEWS)
        for view in VIEWS:
            view_row = by_view[view]
            value = int(view_row["utility"])
            view_margin = correct_direction_margin(view_row)
            rows.append(
                {
                    "recording_id": key[0],
                    "source_frame": key[1],
                    "component_id": key[2],
                    "current_view": view,
                    "selected_view": view,
                    "current_utility": value,
                    "selected_utility": value,
                    "gain": 0,
                    "resolve_at_1": int(value == 2),
                    "regret": oracle - value,
                    "moved": 0,
                    "current_counterfactual_margin": view_margin,
                    "selected_counterfactual_margin": view_margin,
                    "counterfactual_separation_gain": 0.0,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-jsonl", type=Path, required=True)
    parser.add_argument("--original-policy", type=Path, required=True)
    parser.add_argument(
        "--counterfactual-policy",
        "--reflection-policy",
        dest="counterfactual_policy",
        type=Path,
        required=True,
    )
    parser.add_argument("--requirement-report", type=Path, default=None)
    parser.add_argument("--requirement-calibration", type=Path, default=None)
    parser.add_argument("--no-gate-policy", type=Path, default=None)
    parser.add_argument("--no-gate-requirement-report", type=Path, default=None)
    parser.add_argument(
        "--counterfactual-conditioned-requirement",
        action="store_true",
    )
    parser.add_argument(
        "--include-fixed-diagnostics",
        action="store_true",
        help="Report fixed target cameras only as requested diagnostics.",
    )
    parser.add_argument(
        "--include-ray-kernel-diagnostic",
        action="store_true",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, required=True)
    parser.add_argument("--lambda-cost", type=float, default=0.25)
    parser.add_argument("--tau-view", type=float, default=0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument(
        "--report-bootstrap",
        action="store_true",
        help="Optionally report paired bootstrap diagnostics; disabled by default.",
    )
    parser.add_argument("--output-rows-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--claim-semantics",
        choices=("metadata", "generic"),
        default="metadata",
        help="Use frozen claim/evidence-role metadata or the legacy generic state-validity input.",
    )
    args = parser.parse_args()

    grouped: Dict[Tuple[str, int, int], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(args.scores_jsonl):
        view = str(row["view"])
        if view in VIEWS:
            grouped[event_key(row)][view] = row
    grouped = {key: value for key, value in grouped.items() if set(value) == set(VIEWS)}

    original_model = PriorTableRevealModel.load(args.original_policy)
    full_model = PriorTableRevealModel.load(args.counterfactual_policy)
    requirement_counts = load_requirement_counts(
        args.requirement_report,
        counterfactual_conditioned=bool(
            args.counterfactual_conditioned_requirement
        ),
    )
    requirement_calibration = RequirementCalibration.load(
        args.requirement_calibration
    )
    uniform_model = PriorTableRevealModel(
        alpha=2.0,
        default_probability=1.0 / len(ORBIT_ACTIONS),
        counts={},
        metadata={
            "model_type": "uniform_relative_reveal",
            "uses_robot_view_training": False,
        },
    )
    rows = {
        "Current View": current_rows(grouped),
        "Original Assistant Policy": policy_rows(
            grouped,
            original_model,
            cutoff=args.cutoff,
            lambda_cost=args.lambda_cost,
            tau_view=args.tau_view,
            semantics=args.claim_semantics,
        ),
        "R-only": policy_rows(
            grouped,
            uniform_model,
            cutoff=args.cutoff,
            lambda_cost=args.lambda_cost,
            tau_view=args.tau_view,
            semantics=args.claim_semantics,
            requirement_counts=requirement_counts,
            requirement_calibration=requirement_calibration,
        ),
        "pi-only": policy_rows(
            grouped,
            full_model,
            cutoff=args.cutoff,
            lambda_cost=args.lambda_cost,
            tau_view=args.tau_view,
            semantics=args.claim_semantics,
            requirement_counts=requirement_counts,
            requirement_calibration=requirement_calibration,
            uniform_requirement=True,
        ),
        "INSPECT": policy_rows(
            grouped,
            full_model,
            cutoff=args.cutoff,
            lambda_cost=args.lambda_cost,
            tau_view=args.tau_view,
            semantics=args.claim_semantics,
            requirement_counts=requirement_counts,
            requirement_calibration=requirement_calibration,
        ),
    }
    if args.no_gate_policy is not None:
        no_gate_counts = load_requirement_counts(
            args.no_gate_requirement_report or args.requirement_report,
            counterfactual_conditioned=bool(
                args.counterfactual_conditioned_requirement
            ),
        )
        rows["R+pi without Causal Weighting"] = policy_rows(
            grouped,
            PriorTableRevealModel.load(args.no_gate_policy),
            cutoff=args.cutoff,
            lambda_cost=args.lambda_cost,
            tau_view=args.tau_view,
            semantics=args.claim_semantics,
            requirement_counts=no_gate_counts,
            requirement_calibration=requirement_calibration,
        )
    if args.include_fixed_diagnostics:
        for target_view in VIEWS:
            rows[f"Fixed View: {target_view}"] = fixed_view_rows(
                grouped, target_view
            )
    if args.include_ray_kernel_diagnostic:
        rows["INSPECT + Evidence-Role Ray Kernel"] = policy_rows(
            grouped,
            full_model,
            cutoff=args.cutoff,
            lambda_cost=args.lambda_cost,
            tau_view=args.tau_view,
            semantics=args.claim_semantics,
            use_evidence_role_ray_kernel=True,
            requirement_counts=requirement_counts,
            requirement_calibration=requirement_calibration,
        )
    payload = {
        "protocol": {
            "dataset": "IMPACT-v1.1 official test",
            "static_views": list(VIEWS),
            "events": len(grouped),
            "start_view_trials": len(grouped) * len(VIEWS),
            "candidate_view_scores_hidden": True,
            "candidate_view_images_hidden": True,
            "robot_or_impact_view_labels_used_for_policy_training": False,
            "online_policy_inputs": [
                "current evidence roles",
                "active claim semantics",
                "current view id",
                "known static-camera geometry",
            ],
            "policy_input_semantics": args.claim_semantics,
            "bootstrap_unit": "recording",
            "evidence_cutoff": float(args.cutoff),
            "lambda_cost": float(args.lambda_cost),
            "tau_view": float(args.tau_view),
            "original_policy_checkpoint": args.original_policy.name,
            "counterfactual_policy_checkpoint": args.counterfactual_policy.name,
            "requirement_report": (
                args.requirement_report.name if args.requirement_report else None
            ),
            "requirement_calibration": (
                args.requirement_calibration.name
                if args.requirement_calibration
                else None
            ),
            "requirement_backoff": "episode-balanced assistant global role prior",
            "canonical_role_adapter": dict(IMPACT_ROLE_ALIASES),
        },
        "results": {name: aggregate(value) for name, value in rows.items()},
    }
    if args.report_bootstrap:
        payload["paired_bootstrap"] = {
            "inspect_vs_current": bootstrap_delta(
                rows["INSPECT"],
                rows["Current View"],
                args.bootstrap_samples,
                args.seed,
            ),
            "inspect_vs_original": bootstrap_delta(
                rows["INSPECT"],
                rows["Original Assistant Policy"],
                args.bootstrap_samples,
                args.seed,
            ),
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_rows_json is not None:
        args.output_rows_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_rows_json.write_text(
            json.dumps(rows, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
