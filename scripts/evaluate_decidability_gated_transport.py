"""Evaluate transport proposals with claim-structured move verification."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_robot_closed_loop as base
import evaluate_robot_closed_loop_gated as gated
from evaluate_decidability_transport_policy import (
    POLICY_SPEC_KEYS,
    aggregate,
    clone_claim,
    fingerprint,
)
from evaluate_robot_policy_ablation import (
    apply_requirement_weights,
    load_requirement_weights,
    uniform_reveal_model,
)
from inspect_system.active_view.active_selector import ActiveSelector
from inspect_system.active_view.decidability_gated_transport import (
    ClaimStructuredEvidenceTransportSelector,
)
from inspect_system.active_view.decidability_transport import (
    TypedTraceBackoffRevealModel,
    load_typed_requirement_counts,
    mix_typed_requirement_weights,
)
from inspect_system.active_view.evidence_state import EvidenceState
from inspect_system.active_view.requirement_model import RequirementCalibration
from inspect_system.active_view.monotone_evidence_transport import (
    MonotoneClaimEvidenceTransportSelector,
)
from inspect_system.active_view.reveal_model import PriorTableRevealModel
from inspect_system.active_view.view_lattice import SIX_VIEWS
from inspect_system.evidence_scorer import PrototypeEvidenceScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--trial-gt", type=Path, required=True)
    parser.add_argument("--evidence-scorer", type=Path, required=True)
    parser.add_argument("--reveal-model", type=Path, required=True)
    parser.add_argument("--requirement-report", type=Path, required=True)
    parser.add_argument("--requirement-calibration", type=Path, required=True)
    parser.add_argument("--current-geometry", type=Path, required=True)
    parser.add_argument(
        "--ungated-reveal-model",
        type=Path,
        help=(
            "Optional architecture-matched reveal model trained from all "
            "geometry transitions without causal transfer filtering."
        ),
    )
    parser.add_argument("--evaluation-utility-csv", type=Path, required=True)
    parser.add_argument("--evaluation-utility-field", default="human_utility_0_1_2")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-rows-json", type=Path, required=True)
    parser.add_argument(
        "--only-variant",
        action="append",
        default=[],
        help=(
            "Evaluate only the named policy variant. Repeat for multiple "
            "variants; the default evaluates every registered variant."
        ),
    )
    parser.add_argument("--support-threshold", type=float, default=0.35)
    parser.add_argument("--contradiction-threshold", type=float, default=0.45)
    parser.add_argument("--margin-threshold", type=float, default=0.05)
    parser.add_argument("--role-threshold", type=float, default=0.65)
    parser.add_argument("--partial-threshold", type=float, default=0.35)
    parser.add_argument("--lambda-cost", type=float, default=0.05)
    parser.add_argument("--tau-view", type=float, default=0.02)
    parser.add_argument("--commit-identity-confidence", type=float)
    parser.add_argument("--commit-identity-margin", type=float)
    parser.add_argument("--commit-alternative-identity-confidence", type=float)
    parser.add_argument("--commit-counterfactual-margin", type=float)
    parser.add_argument("--commit-relation-threshold", type=float)
    parser.add_argument(
        "--commit-explicit-contradiction-only",
        action="store_true",
        help="Do not treat absent positive relation evidence as contradiction.",
    )
    return parser.parse_args()


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
    discrete_reveal_model = getattr(reveal_model, "base_model", reveal_model)
    ungated_reveal_model = (
        PriorTableRevealModel.load(args.ungated_reveal_model)
        if args.ungated_reveal_model is not None
        else None
    )
    typed_reveal_model = TypedTraceBackoffRevealModel.from_report(
        reveal_model,
        args.requirement_report,
    )
    calibration = RequirementCalibration.load(args.requirement_calibration)
    parent_requirements = load_requirement_weights(
        args.requirement_report, beta=calibration.beta
    )
    typed_requirements = load_typed_requirement_counts(args.requirement_report)

    predictions: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    commit_predictions: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
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
            if args.commit_identity_confidence is None:
                commit_predictions[trial_id][view_id] = dict(prediction)
            else:
                commit_predictions[trial_id][view_id] = (
                    gated.apply_evidence_availability_gate(
                        prediction,
                        identity_confidence_threshold=float(
                            args.commit_identity_confidence
                        ),
                        identity_margin_floor=float(
                            args.commit_identity_margin
                            if args.commit_identity_margin is not None
                            else gated.IDENTITY_MARGIN_THRESHOLD
                        ),
                        counterfactual_identity_threshold=(
                            float(args.commit_alternative_identity_confidence)
                            if args.commit_alternative_identity_confidence is not None
                            else None
                        ),
                        counterfactual_margin_floor=(
                            float(args.commit_counterfactual_margin)
                            if args.commit_counterfactual_margin is not None
                            else None
                        ),
                        relation_evidence_threshold=float(
                            args.commit_relation_threshold
                            if args.commit_relation_threshold is not None
                            else gated.RELATION_EVIDENCE_THRESHOLD
                        ),
                        allow_relation_absence_contradiction=not bool(
                            args.commit_explicit_contradiction_only
                        ),
                    )
                )
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
        "R-only": ClaimStructuredEvidenceTransportSelector(
            model=uniform_reveal_model(),
            lambda_cost=float(args.lambda_cost),
            tau_view=float(args.tau_view),
            geometry_loss_cap=1.0,
        ),
        "pi-only": ClaimStructuredEvidenceTransportSelector(
            model=reveal_model,
            lambda_cost=float(args.lambda_cost),
            tau_view=float(args.tau_view),
            geometry_loss_cap=1.0,
        ),
        "Discrete Transport (no object-centric calibration)": (
            ClaimStructuredEvidenceTransportSelector(
                model=discrete_reveal_model,
                lambda_cost=float(args.lambda_cost),
                tau_view=float(args.tau_view),
                geometry_loss_cap=1.0,
            )
        ),
        "Counterfactual-Agnostic Transport": (
            ClaimStructuredEvidenceTransportSelector(
                model=reveal_model,
                lambda_cost=float(args.lambda_cost),
                tau_view=float(args.tau_view),
                geometry_loss_cap=1.0,
                use_counterfactual_relevance_for_proposal=False,
            )
        ),
        "Relative Transport (parent R)": ActiveSelector(
            model=reveal_model,
            lambda_cost=float(args.lambda_cost),
            tau_view=float(args.tau_view),
        ),
        "Relative Transport (typed R)": ActiveSelector(
            model=reveal_model,
            lambda_cost=float(args.lambda_cost),
            tau_view=float(args.tau_view),
        ),
        "Typed-Backoff Reveal (diagnostic)": ActiveSelector(
            model=typed_reveal_model,
            lambda_cost=float(args.lambda_cost),
            tau_view=float(args.tau_view),
        ),
        "Claim-Structured Transport (parent R)": ClaimStructuredEvidenceTransportSelector(
            model=reveal_model,
            lambda_cost=float(args.lambda_cost),
            tau_view=float(args.tau_view),
            geometry_loss_cap=1.0,
        ),
        "INSPECT": ClaimStructuredEvidenceTransportSelector(
            model=reveal_model,
            lambda_cost=float(args.lambda_cost),
            tau_view=float(args.tau_view),
            geometry_loss_cap=1.0,
        ),
        "Claim-Structured Transport + Projected Roles": ClaimStructuredEvidenceTransportSelector(
            model=reveal_model,
            lambda_cost=float(args.lambda_cost),
            tau_view=float(args.tau_view),
            geometry_loss_cap=1.0,
            use_projected_role_factors_for_proposal=True,
        ),
        "Typed-Backoff Claim Structured (diagnostic)": ClaimStructuredEvidenceTransportSelector(
            model=typed_reveal_model,
            lambda_cost=float(args.lambda_cost),
            tau_view=float(args.tau_view),
            geometry_loss_cap=1.0,
        ),
        "Monotone Evidence Transport": MonotoneClaimEvidenceTransportSelector(
            model=reveal_model,
            lambda_cost=float(args.lambda_cost),
            tau_view=float(args.tau_view),
            geometry_loss_cap=1.0,
        ),
    }
    if ungated_reveal_model is not None:
        selectors["Ungated Geometric Transport"] = (
            ClaimStructuredEvidenceTransportSelector(
                model=ungated_reveal_model,
                lambda_cost=float(args.lambda_cost),
                tau_view=float(args.tau_view),
                geometry_loss_cap=1.0,
            )
        )
    if args.only_variant:
        requested = set(args.only_variant)
        missing = sorted(requested.difference(selectors))
        if missing:
            raise ValueError(f"Unknown --only-variant value(s): {missing}")
        selectors = {
            name: selector
            for name, selector in selectors.items()
            if name in requested
        }

    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    all_results: Dict[str, Any] = {}
    for variant, selector in selectors.items():
        rows: List[Dict[str, Any]] = []
        selection_times_ms: List[float] = []
        for trial_id, view_predictions in sorted(predictions.items()):
            gt = gt_by_trial[trial_id]
            truth = str(gt.get("claim_outcome", "")).strip().lower()
            oracle_utility = max(utilities[trial_id].values(), default=0)
            for current_view, current_prediction in sorted(view_predictions.items()):
                typed_state = clone_claim(
                    states[trial_id][current_view], str(gt.get("claim_id", ""))
                )
                if variant == "pi-only":
                    state = apply_requirement_weights(
                        typed_state,
                        {},
                        beta=calibration.beta,
                        uniform=True,
                        preserve_scores=True,
                        strength=0.0,
                        temperature=calibration.temperature,
                    )
                elif variant in {
                    "Relative Transport (parent R)",
                    "Claim-Structured Transport (parent R)",
                }:
                    state = mix_typed_requirement_weights(
                        typed_state,
                        parent_requirements,
                        {},
                        beta=calibration.beta,
                        temperature=calibration.temperature,
                    )
                else:
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
                        "candidate_role_factors": {},
                        "preservation_role_factors": dict(
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
                    started = time.perf_counter()
                    selection = selector.select(
                        current_view=current_view,
                        evidence_state=state,
                        candidate_context=context,
                    )
                    selection_times_ms.append(
                        1000.0 * (time.perf_counter() - started)
                    )
                    selected_view = (
                        selection.selected_view
                        if selection.action == "move"
                        else current_view
                    )
                else:
                    selected_view = current_view
                discovery_prediction = view_predictions[selected_view]
                selected_prediction = commit_predictions[trial_id][selected_view]
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
                            selected_decision == "contradicted" and truth != "contradicted"
                        ),
                        "defer": int(selected_decision == "insufficient"),
                        "moved": int(selected_view != current_view),
                        "no_improvement": int(
                            selected_utility <= current_utility and current_utility < 2
                        ),
                        "selected_decision": selected_decision,
                        "discovery_decision": str(discovery_prediction["decision"]),
                        "commit_gate_changed_decision": int(
                            selected_decision != str(discovery_prediction["decision"])
                        ),
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
        ordered_times = sorted(selection_times_ms)
        if ordered_times:
            p95_index = min(
                len(ordered_times) - 1,
                max(0, int(0.95 * len(ordered_times))),
            )
            all_results[variant]["selector_runtime_ms"] = {
                "calls": len(ordered_times),
                "mean": sum(ordered_times) / len(ordered_times),
                "median": ordered_times[len(ordered_times) // 2],
                "p95": ordered_times[p95_index],
            }

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
            "evaluated_variants": list(selectors),
            "requirement_beta": calibration.beta,
            "requirement_temperature": calibration.temperature,
            "full_policy_reveal_model": dict(
                getattr(reveal_model, "metadata", {}) or {}
            ),
            "ungated_geometric_transport_model": (
                dict(getattr(ungated_reveal_model, "metadata", {}) or {})
                if ungated_reveal_model is not None
                else None
            ),
            "typed_backoff_diagnostic_model": dict(typed_reveal_model.metadata),
            "asymmetric_evidence_channels": bool(
                args.commit_identity_confidence is not None
            ),
            "commit_gate": {
                "identity_confidence": args.commit_identity_confidence,
                "identity_margin_floor": args.commit_identity_margin,
                "counterfactual_identity_threshold": (
                    args.commit_alternative_identity_confidence
                ),
                "counterfactual_margin_floor": args.commit_counterfactual_margin,
                "relation_evidence": args.commit_relation_threshold,
                "explicit_contradiction_only": bool(
                    args.commit_explicit_contradiction_only
                ),
            },
            "input_fingerprints": {
                "observations": fingerprint(args.observations),
                "trial_gt": fingerprint(args.trial_gt),
                "evidence_scorer": fingerprint(args.evidence_scorer),
                "reveal_model": fingerprint(args.reveal_model),
                "ungated_reveal_model": (
                    fingerprint(args.ungated_reveal_model)
                    if args.ungated_reveal_model is not None
                    else None
                ),
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
