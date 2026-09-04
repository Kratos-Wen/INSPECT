"""Evaluate structured decidability transport on the frozen six-view replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_robot_closed_loop as base
import evaluate_robot_closed_loop_gated as gated
from evaluate_robot_policy_ablation import (
    apply_requirement_weights,
    load_requirement_weights,
)
from inspect_system.active_view.active_selector import ActiveSelector
from inspect_system.active_view.decidability_transport import (
    StructuredDecidabilitySelector,
    load_typed_requirement_counts,
    mix_typed_requirement_weights,
)
from inspect_system.active_view.evidence_state import EvidenceState
from inspect_system.active_view.requirement_model import RequirementCalibration
from inspect_system.active_view.reveal_model import PriorTableRevealModel
from inspect_system.active_view.view_lattice import SIX_VIEWS
from inspect_system.evidence_scorer import PrototypeEvidenceScorer


VIEWS = tuple(sorted(SIX_VIEWS))
POLICY_SPEC_KEYS = ("target_step", "claim_id", "product_variant")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--trial-gt", type=Path, required=True)
    parser.add_argument("--evidence-scorer", type=Path, required=True)
    parser.add_argument("--reveal-model", type=Path, required=True)
    parser.add_argument("--requirement-report", type=Path, required=True)
    parser.add_argument("--requirement-calibration", type=Path, required=True)
    parser.add_argument("--current-geometry", type=Path, required=True)
    parser.add_argument("--evaluation-utility-csv", type=Path, required=True)
    parser.add_argument("--evaluation-utility-field", default="human_utility_0_1_2")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-rows-json", type=Path, required=True)
    parser.add_argument("--support-threshold", type=float, default=0.35)
    parser.add_argument("--contradiction-threshold", type=float, default=0.45)
    parser.add_argument("--margin-threshold", type=float, default=0.05)
    parser.add_argument("--role-threshold", type=float, default=0.65)
    parser.add_argument("--partial-threshold", type=float, default=0.35)
    parser.add_argument("--lambda-cost", type=float, default=0.05)
    parser.add_argument("--tau-view", type=float, default=0.02)
    return parser.parse_args()


def fingerprint(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def clone_claim(state: EvidenceState, claim_id: str) -> EvidenceState:
    return EvidenceState(
        claim_id=claim_id,
        items=list(state.items),
        claim_score=state.claim_score,
        contradiction_score=state.contradiction_score,
        margin=state.margin,
        counterfactual_family=state.counterfactual_family,
        counterfactual_scores=dict(state.counterfactual_scores),
        current_utility_proxy=state.current_utility_proxy,
    )


def aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    keys = (
        "current_utility",
        "selected_utility",
        "gain",
        "regret",
        "resolve_at_1",
        "correct_resolution",
        "false_support",
        "false_contradiction",
        "defer",
        "moved",
        "no_improvement",
    )
    return {
        "trials": len(rows),
        **{key: mean(float(row[key]) for row in rows) for key in keys},
        "decision_counts": dict(
            Counter(str(row["selected_decision"]) for row in rows)
        ),
        "selection_counts": dict(
            Counter(str(row["selection_action"]) for row in rows)
        ),
    }


def main() -> None:
    args = parse_args()
    gt_by_trial = {row["trial_id"]: row for row in base.read_csv(args.trial_gt)}
    observations: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for observation in base.read_jsonl(args.observations):
        metadata = dict(observation.get("metadata") or {})
        trial_id = str(metadata.get("trial_id", ""))
        view_id = str(observation.get("view_id", ""))
        if trial_id and view_id:
            observations[trial_id][view_id] = observation

    geometry: Dict[str, Mapping[str, Any]] = {}
    for item in base.read_jsonl(args.current_geometry):
        if item.get("candidate_images_used") or item.get("robot_utility_labels_used"):
            raise RuntimeError("Current-geometry cache contains forbidden supervision.")
        observation_id = str(item.get("observation_id", ""))
        if observation_id and str(item.get("status", "")) == "ok":
            geometry[observation_id] = item

    scorer = PrototypeEvidenceScorer.load(args.evidence_scorer)
    reveal_model = PriorTableRevealModel.load(args.reveal_model)
    calibration = RequirementCalibration.load(args.requirement_calibration)
    parent_requirements = load_requirement_weights(
        args.requirement_report, beta=calibration.beta
    )
    typed_requirements = load_typed_requirement_counts(args.requirement_report)

    predictions: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    states: Dict[str, Dict[str, EvidenceState]] = defaultdict(dict)
    utilities: Dict[str, Dict[str, int]] = defaultdict(dict)
    for trial_id, view_map in sorted(observations.items()):
        gt = gt_by_trial.get(trial_id, {})
        if str(gt.get("target_step", "")).lower() not in base.ACTIVE_STEPS:
            continue
        policy_spec = {key: str(gt.get(key, "")) for key in POLICY_SPEC_KEYS}
        truth = str(gt.get("claim_outcome", "")).strip().lower()
        for view_id, observation in sorted(view_map.items()):
            state, prediction = gated.gated_claim_evidence(
                observation,
                gt=policy_spec,
                scorer=scorer,
                support_threshold=float(args.support_threshold),
                contradiction_threshold=float(args.contradiction_threshold),
                margin_threshold=float(args.margin_threshold),
                role_threshold=float(args.role_threshold),
            )
            predictions[trial_id][view_id] = prediction
            states[trial_id][view_id] = state
            utilities[trial_id][view_id] = base.evaluation_utility(
                prediction, truth, float(args.partial_threshold)
            )

    loaded = 0
    for row in base.read_csv(args.evaluation_utility_csv):
        trial_id = str(row.get("trial_id", ""))
        view_id = str(row.get("view_id", ""))
        value = str(row.get(args.evaluation_utility_field, "")).strip()
        if trial_id in predictions and view_id in predictions[trial_id] and value in {"0", "1", "2"}:
            utilities[trial_id][view_id] = int(value)
            loaded += 1
    expected = sum(len(view_map) for view_map in predictions.values())
    if loaded != expected:
        raise RuntimeError(f"Expected {expected} utility labels, loaded {loaded}.")

    selectors = {
        "INSPECT v1 (frozen selector)": ActiveSelector(
            model=reveal_model,
            lambda_cost=float(args.lambda_cost),
            tau_view=float(args.tau_view),
        ),
        "Structured Transport (uniform R)": StructuredDecidabilitySelector(
            model=reveal_model,
            lambda_cost=float(args.lambda_cost),
            tau_view=float(args.tau_view),
            geometry_loss_cap=1.0,
        ),
        "Structured Transport (parent R)": StructuredDecidabilitySelector(
            model=reveal_model,
            lambda_cost=float(args.lambda_cost),
            tau_view=float(args.tau_view),
            geometry_loss_cap=1.0,
        ),
        "Structured Transport w/o Evidence-Loss Risk": StructuredDecidabilitySelector(
            model=reveal_model,
            lambda_cost=float(args.lambda_cost),
            tau_view=float(args.tau_view),
            geometry_loss_cap=0.0,
        ),
        "INSPECT": StructuredDecidabilitySelector(
            model=reveal_model,
            lambda_cost=float(args.lambda_cost),
            tau_view=float(args.tau_view),
            geometry_loss_cap=1.0,
        ),
    }

    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    all_results: Dict[str, Any] = {}
    for variant, selector in selectors.items():
        rows: List[Dict[str, Any]] = []
        for trial_id, view_predictions in sorted(predictions.items()):
            gt = gt_by_trial[trial_id]
            truth = str(gt.get("claim_outcome", "")).strip().lower()
            oracle_utility = max(utilities[trial_id].values(), default=0)
            for current_view, current_prediction in sorted(view_predictions.items()):
                generic_state = states[trial_id][current_view]
                if variant == "INSPECT v1 (frozen selector)":
                    state = apply_requirement_weights(
                        generic_state,
                        parent_requirements,
                        beta=calibration.beta,
                        uniform=False,
                        preserve_scores=True,
                        strength=calibration.strength,
                        temperature=calibration.temperature,
                    )
                elif variant == "Structured Transport (uniform R)":
                    typed_state = clone_claim(generic_state, str(gt.get("claim_id", "")))
                    state = apply_requirement_weights(
                        typed_state,
                        parent_requirements,
                        beta=calibration.beta,
                        uniform=True,
                        preserve_scores=True,
                        strength=0.0,
                        temperature=1.0,
                    )
                elif variant == "Structured Transport (parent R)":
                    typed_state = clone_claim(generic_state, str(gt.get("claim_id", "")))
                    state = mix_typed_requirement_weights(
                        typed_state,
                        parent_requirements,
                        {},
                        beta=calibration.beta,
                        temperature=calibration.temperature,
                    )
                else:
                    typed_state = clone_claim(generic_state, str(gt.get("claim_id", "")))
                    state = mix_typed_requirement_weights(
                        typed_state,
                        parent_requirements,
                        typed_requirements,
                        beta=calibration.beta,
                        temperature=calibration.temperature,
                    )

                triggered = str(current_prediction["decision"]) == "insufficient"
                selection = None
                if triggered:
                    item = geometry.get(f"{trial_id}_{current_view}", {})
                    context = {
                        "candidate_role_factors": dict(
                            item.get("candidate_role_factors") or {}
                        ),
                        "surface_normal_world": item.get("surface_normal_world"),
                        "role_surface_normals_world": dict(
                            item.get("role_surface_normals_world") or {}
                        ),
                        "role_relation_frames_world": dict(
                            item.get("role_relation_frames_world") or {}
                        ),
                    }
                    selection = selector.select(
                        current_view=current_view,
                        evidence_state=state,
                        candidate_context=context,
                    )
                    selected_view = (
                        selection.selected_view
                        if selection.action == "move"
                        else current_view
                    )
                else:
                    selected_view = current_view
                selected_prediction = view_predictions[selected_view]
                selected_decision = str(selected_prediction["decision"])
                current_utility = utilities[trial_id][current_view]
                selected_utility = utilities[trial_id][selected_view]
                rows.append(
                    {
                        "trial_id": trial_id,
                        "target_step": str(gt.get("target_step", "")),
                        "claim_id": str(gt.get("claim_id", "")),
                        "current_view": current_view,
                        "selected_view": selected_view,
                        "triggered": triggered,
                        "current_utility": current_utility,
                        "selected_utility": selected_utility,
                        "gain": selected_utility - current_utility,
                        "regret": oracle_utility - selected_utility,
                        "resolve_at_1": int(selected_utility == 2),
                        "correct_resolution": int(selected_decision == truth),
                        "false_support": int(
                            selected_decision == "supported" and truth != "supported"
                        ),
                        "false_contradiction": int(
                            selected_decision == "contradicted"
                            and truth != "contradicted"
                        ),
                        "defer": int(selected_decision == "insufficient"),
                        "moved": int(selected_view != current_view),
                        "no_improvement": int(
                            selected_utility <= current_utility and current_utility < 2
                        ),
                        "selected_decision": selected_decision,
                        "selection_action": (
                            selection.action if selection else "not_triggered"
                        ),
                        "selection_reason": (
                            selection.reason if selection else "claim_already_decidable"
                        ),
                        "selection_score": (
                            float(selection.score) if selection else 0.0
                        ),
                        "ranked_views": (
                            list(selection.ranked_views) if selection else []
                        ),
                    }
                )
        all_rows[variant] = rows
        all_results[variant] = aggregate(rows)

    output = {
        "protocol": {
            "candidate_view_images_hidden": True,
            "robot_labels_used_for_policy": False,
            "robot_utility_used_for_policy": False,
            "current_geometry_uses_candidate_images": False,
            "policy_spec_keys": list(POLICY_SPEC_KEYS),
            "policy_forbidden_gt_keys": [
                "claim_outcome",
                "error_type",
                "component_identity",
                "relation_state",
            ],
            "setups": len(predictions),
            "start_views": expected,
            "lambda_cost": float(args.lambda_cost),
            "tau_view": float(args.tau_view),
            "requirement_beta": calibration.beta,
            "requirement_temperature": calibration.temperature,
            "input_fingerprints": {
                "observations": fingerprint(args.observations),
                "trial_gt": fingerprint(args.trial_gt),
                "evidence_scorer": fingerprint(args.evidence_scorer),
                "reveal_model": fingerprint(args.reveal_model),
                "requirement_report": fingerprint(args.requirement_report),
                "requirement_calibration": fingerprint(args.requirement_calibration),
                "current_geometry": fingerprint(args.current_geometry),
                "evaluation_utility": fingerprint(args.evaluation_utility_csv),
            },
        },
        "results": all_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output_rows_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_rows_json.write_text(
        json.dumps(all_rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
