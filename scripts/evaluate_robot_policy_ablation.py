"""Ablate R, pi, transfer gating, and robot-supervised view selection.

All policies reuse one frozen robot perception cache. Assistant-supervised
policies never consume candidate-view observations or robot utility labels.
The diagnostic kNN may use robot utility labels, but its evaluation is
strictly leave-one-setup-out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_robot_closed_loop as base
import evaluate_robot_closed_loop_gated as gated
from inspect_system.active_view.active_selector import ActiveSelector
from inspect_system.active_view.evidence_state import EvidenceItem, EvidenceState
from inspect_system.active_view.ontology import normalize_claim
from inspect_system.active_view.requirement_model import (
    RequirementCalibration,
    load_requirement_counts,
    mix_requirement_weights,
)
from inspect_system.active_view.reveal_model import PriorTableRevealModel
from inspect_system.active_view.view_lattice import (
    ORBIT_ACTIONS,
    SIX_VIEWS,
    action_compatibility,
    transition_cost,
)
from inspect_system.evidence_scorer import PrototypeEvidenceScorer


VIEWS = tuple(sorted(SIX_VIEWS))


def file_fingerprint(path: Path | None) -> Dict[str, Any] | None:
    """Return a reproducible input fingerprint for protocol auditing."""

    if path is None:
        return None
    resolved = path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": resolved.stat().st_size,
    }


def load_requirement_weights(
    path: Path,
    beta: float,
    *,
    counterfactual_conditioned: bool = False,
) -> Dict[str, Dict[str, float]]:
    del beta
    return load_requirement_counts(
        path,
        episode_balanced=True,
        counterfactual_conditioned=counterfactual_conditioned,
    )


def calibrated_requirement_settings(
    path: Path | None,
    *,
    fallback_beta: float,
    fallback_strength: float,
    fallback_temperature: float,
) -> RequirementCalibration:
    return RequirementCalibration.load(
        path,
        beta=fallback_beta,
        strength=fallback_strength,
        temperature=fallback_temperature,
    )


def apply_requirement_weights(
    state: EvidenceState,
    learned: Mapping[str, Mapping[str, float]],
    *,
    beta: float,
    uniform: bool,
    preserve_scores: bool = False,
    strength: float = 1.0,
    temperature: float = 1.0,
) -> EvidenceState:
    return mix_requirement_weights(
        state,
        learned,
        beta=beta,
        strength=strength,
        temperature=temperature,
        uniform=uniform,
        preserve_scores=preserve_scores,
    )


def uniform_reveal_model() -> PriorTableRevealModel:
    return PriorTableRevealModel(
        alpha=2.0, default_probability=1.0 / len(ORBIT_ACTIONS), counts={}
    )


def claim_prior_model(full: PriorTableRevealModel) -> PriorTableRevealModel:
    claim_copy = getattr(full, "claim_conditioned_copy", None)
    if callable(claim_copy):
        return claim_copy()
    counts: Dict[str, Dict[str, float]] = {}
    for key, value in full.counts.items():
        parts = key.split("|")
        if (
            len(parts) == 3
            and parts[0] not in {"__global__", "global"}
            and parts[1] in {"__any__", "any"}
        ):
            counts[key] = dict(value)
    return PriorTableRevealModel(
        alpha=full.alpha,
        default_probability=full.default_probability,
        counts=counts,
        metadata={
            "model_type": "claim_conditioned_relative_prior",
            "uses_robot_view_training": False,
        },
    )


def feature_vector(state: EvidenceState, current_view: str) -> np.ndarray:
    role_map = state.evidence_vector()
    role_names = sorted(
        {
            "identity_disambiguation_view",
            "insertion_verification_view",
            "containment_verification_view",
            "slot_relation_view",
            "gap_visibility_view",
            "boundary_alignment_view",
            "contact_verification_view",
            "claim_disambiguation_view",
        }
    )
    claim_names = ("gear_inserted", "cover_seated", "state_validity")
    return np.asarray(
        [state.claim_score, state.contradiction_score, state.margin]
        + [float(role_map.get(name, 0.0)) for name in role_names]
        + [
            1.0 if normalize_claim(state.claim_id) == name else 0.0
            for name in claim_names
        ]
        + [1.0 if current_view == view else 0.0 for view in VIEWS],
        dtype=np.float64,
    )


def knn_select(
    *,
    test_trial: str,
    current_view: str,
    state: EvidenceState,
    states: Mapping[str, Mapping[str, EvidenceState]],
    utilities: Mapping[str, Mapping[str, int]],
    k: int,
    lambda_cost: float,
) -> str:
    train_x: List[np.ndarray] = []
    train_y: List[np.ndarray] = []
    for trial_id, view_states in states.items():
        if trial_id == test_trial:
            continue
        target = np.asarray(
            [float(utilities[trial_id].get(view, 0)) for view in VIEWS],
            dtype=np.float64,
        )
        for train_current, train_state in view_states.items():
            train_x.append(feature_vector(train_state, train_current))
            train_y.append(target)
    if not train_x:
        return current_view
    matrix = np.stack(train_x)
    query = feature_vector(state, current_view)
    scale = np.std(matrix, axis=0)
    scale[scale < 1e-6] = 1.0
    distances = np.linalg.norm((matrix - query) / scale, axis=1)
    indices = np.argsort(distances)[: max(1, min(k, len(distances)))]
    weights = 1.0 / np.maximum(1e-6, distances[indices])
    predicted = np.average(
        np.stack([train_y[index] for index in indices]), axis=0, weights=weights
    )
    scores = {
        view: float(predicted[index])
        - lambda_cost * transition_cost(current_view, view)
        for index, view in enumerate(VIEWS)
    }
    return max(
        VIEWS,
        key=lambda view: (scores[view], -transition_cost(current_view, view), view),
    )


def fixed_action_select(
    *,
    action: str,
    current_view: str,
    lambda_cost: float,
) -> str:
    candidates = [view for view in VIEWS if view != current_view]
    scores = {
        view: action_compatibility(action, current_view, view)
        - float(lambda_cost) * transition_cost(current_view, view)
        for view in candidates
    }
    return max(
        candidates,
        key=lambda view: (
            scores[view],
            -transition_cost(current_view, view),
            view,
        ),
    )


def aggregate(
    rows: Sequence[Mapping[str, Any]], *, triggered_only: bool
) -> Dict[str, Any]:
    subset = [
        row for row in rows if not triggered_only or bool(row.get("triggered", False))
    ]
    if not subset:
        return {"trials": 0}
    keys = [
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
    ]
    return {
        "trials": len(subset),
        **{key: mean(float(row.get(key, 0.0)) for row in subset) for key in keys},
        "decision_counts": dict(
            Counter(str(row.get("selected_decision", "")) for row in subset)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--trial-gt", type=Path, required=True)
    parser.add_argument("--evidence-scorer", type=Path, required=True)
    parser.add_argument("--full-reveal-model", type=Path, required=True)
    parser.add_argument("--no-gate-reveal-model", type=Path, required=True)
    parser.add_argument("--requirement-report", type=Path, required=True)
    parser.add_argument("--no-gate-requirement-report", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-rows-json", type=Path, default=None)
    parser.add_argument("--support-threshold", type=float, default=0.35)
    parser.add_argument("--contradiction-threshold", type=float, default=0.45)
    parser.add_argument("--margin-threshold", type=float, default=0.05)
    parser.add_argument("--role-threshold", type=float, default=0.65)
    parser.add_argument("--partial-threshold", type=float, default=0.35)
    parser.add_argument("--evaluation-utility-csv", type=Path, default=None)
    parser.add_argument("--evaluation-utility-field", default="human_utility_0_1_2")
    parser.add_argument("--lambda-cost", type=float, default=0.25)
    parser.add_argument("--tau-view", type=float, default=0.02)
    parser.add_argument("--requirement-beta", type=float, default=2.0)
    parser.add_argument("--requirement-strength", type=float, default=1.0)
    parser.add_argument("--requirement-temperature", type=float, default=1.0)
    parser.add_argument("--requirement-calibration-json", type=Path, default=None)
    parser.add_argument(
        "--counterfactual-conditioned-requirement",
        action="store_true",
        help=(
            "Condition R(c,q) on the evidence-derived counterfactual family. "
            "The default preserves the original claim-only requirement model."
        ),
    )
    parser.add_argument("--knn-k", type=int, default=7)
    parser.add_argument(
        "--candidate-affordance-jsonl",
        type=Path,
        default=None,
        help="Optional current-view-only candidate role factors.",
    )
    parser.add_argument(
        "--preserve-partial-evidence",
        action="store_true",
        help="Preserve continuous evidence scores when applying learned role weights.",
    )
    args = parser.parse_args()

    gt_by_trial = {row["trial_id"]: row for row in base.read_csv(args.trial_gt)}
    observations: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for observation in base.read_jsonl(args.observations):
        metadata = dict(observation.get("metadata") or {})
        trial_id = str(metadata.get("trial_id", ""))
        view_id = str(observation.get("view_id", ""))
        if trial_id and view_id:
            observations[trial_id][view_id] = observation
    incomplete = {
        trial_id: sorted(set(VIEWS) - set(view_map))
        for trial_id, view_map in observations.items()
        if set(view_map) != set(VIEWS)
    }
    if incomplete:
        raise RuntimeError(f"Incomplete six-view observations: {incomplete}")

    ray_affordances: Dict[str, Mapping[str, Any]] = {}
    if args.candidate_affordance_jsonl is not None:
        for item in base.read_jsonl(args.candidate_affordance_jsonl):
            if bool(item.get("candidate_images_used", False)):
                raise RuntimeError("Candidate-affordance cache used candidate images.")
            if bool(item.get("robot_utility_labels_used", False)):
                raise RuntimeError(
                    "Candidate-affordance cache used robot utility labels."
                )
            observation_id = str(item.get("observation_id", ""))
            if observation_id and str(item.get("status", "")) == "ok":
                ray_affordances[observation_id] = item

    scorer = PrototypeEvidenceScorer.load(args.evidence_scorer)
    full_model = PriorTableRevealModel.load(args.full_reveal_model)
    no_gate_model = PriorTableRevealModel.load(args.no_gate_reveal_model)
    context_free_model = getattr(full_model, "base_model", full_model)
    requirement_calibration = calibrated_requirement_settings(
        args.requirement_calibration_json,
        fallback_beta=float(args.requirement_beta),
        fallback_strength=float(args.requirement_strength),
        fallback_temperature=float(args.requirement_temperature),
    )
    requirement_beta = requirement_calibration.beta
    requirement_strength = requirement_calibration.strength
    requirement_temperature = requirement_calibration.temperature
    learned_r = load_requirement_weights(
        args.requirement_report,
        beta=requirement_beta,
        counterfactual_conditioned=bool(
            args.counterfactual_conditioned_requirement
        ),
    )
    no_gate_r = load_requirement_weights(
        args.no_gate_requirement_report or args.requirement_report,
        beta=requirement_beta,
        counterfactual_conditioned=bool(
            args.counterfactual_conditioned_requirement
        ),
    )

    predictions: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    states: Dict[str, Dict[str, EvidenceState]] = defaultdict(dict)
    utilities: Dict[str, Dict[str, int]] = defaultdict(dict)
    for trial_id, view_map in sorted(observations.items()):
        gt = gt_by_trial.get(trial_id, {})
        if str(gt.get("target_step", "")).lower() not in base.ACTIVE_STEPS:
            continue
        truth = str(gt.get("claim_outcome", "")).strip().lower()
        for view_id, observation in sorted(view_map.items()):
            state, prediction = gated.gated_claim_evidence(
                observation,
                gt=gt,
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

    utility_source = "frozen_verifier_correctness_and_partial_evidence"
    if args.evaluation_utility_csv is not None:
        loaded = 0
        for row in base.read_csv(args.evaluation_utility_csv):
            trial_id = str(row.get("trial_id", ""))
            view_id = str(row.get("view_id", ""))
            value = str(row.get(args.evaluation_utility_field, "")).strip()
            if (
                trial_id not in predictions
                or view_id not in predictions[trial_id]
                or value not in {"0", "1", "2"}
            ):
                continue
            utilities[trial_id][view_id] = int(value)
            loaded += 1
        expected = sum(len(values) for values in predictions.values())
        if loaded != expected:
            raise RuntimeError(
                f"Expected {expected} evaluation utilities, loaded {loaded}."
            )
        utility_source = (
            f"{args.evaluation_utility_csv}:{args.evaluation_utility_field}"
        )

    variants = {
        "Claim-Conditioned Prior": (
            claim_prior_model(context_free_model),
            False,
            learned_r,
            "none",
        ),
        "R-only": (uniform_reveal_model(), False, learned_r, "none"),
        "pi-only": (full_model, True, learned_r, "object"),
        "pi-only w/o Object-Centric Calibration": (
            context_free_model,
            True,
            learned_r,
            "none",
        ),
        "R+pi without Causal Weighting": (no_gate_model, False, no_gate_r, "object"),
        "INSPECT": (full_model, False, learned_r, "object"),
    }
    if ray_affordances:
        variants["INSPECT + Evidence-Role Ray Kernel"] = (
            full_model,
            False,
            learned_r,
            "ray",
        )
    all_results: Dict[str, Any] = {}
    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    for variant, (model, uniform_r, variant_r, context_mode) in variants.items():
        selector = ActiveSelector(
            model=model,
            lambda_cost=float(args.lambda_cost),
            tau_view=float(args.tau_view),
        )
        rows: List[Dict[str, Any]] = []
        for trial_id, view_predictions in sorted(predictions.items()):
            gt = gt_by_trial[trial_id]
            truth = str(gt.get("claim_outcome", "")).strip().lower()
            oracle_utility = max(utilities[trial_id].values(), default=0)
            for current_view, current_prediction in sorted(view_predictions.items()):
                current_decision = str(current_prediction["decision"])
                state = apply_requirement_weights(
                    states[trial_id][current_view],
                    variant_r,
                    beta=requirement_beta,
                    uniform=uniform_r,
                    preserve_scores=bool(args.preserve_partial_evidence),
                    strength=requirement_strength,
                    temperature=requirement_temperature,
                )
                triggered = current_decision == "insufficient"
                selection_result = None
                if triggered:
                    ray_item = ray_affordances.get(f"{trial_id}_{current_view}", {})
                    result = selector.select(
                        current_view=current_view,
                        evidence_state=state,
                        candidate_context=(
                            {
                                **(
                                    {
                                        "candidate_role_factors": dict(
                                            ray_item.get("candidate_role_factors") or {}
                                        )
                                    }
                                    if context_mode == "ray"
                                    else {}
                                ),
                                "surface_normal_world": ray_item.get(
                                    "surface_normal_world"
                                ),
                                "role_surface_normals_world": dict(
                                    ray_item.get("role_surface_normals_world") or {}
                                ),
                                "role_relation_frames_world": dict(
                                    ray_item.get("role_relation_frames_world") or {}
                                ),
                            }
                            if context_mode in {"object", "ray"}
                            else None
                        ),
                    )
                    selection_result = result
                    selected_view = (
                        result.selected_view
                        if result.action == "move"
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
                        "selection_action": (
                            selection_result.action
                            if selection_result
                            else "not_triggered"
                        ),
                        "selection_reason": (
                            selection_result.reason
                            if selection_result
                            else "claim_already_decidable"
                        ),
                        "selection_score": (
                            float(selection_result.score) if selection_result else 0.0
                        ),
                        "ranked_views": (
                            list(selection_result.ranked_views)
                            if selection_result
                            else []
                        ),
                        "no_improvement": int(
                            selected_utility <= current_utility and current_utility < 2
                        ),
                        "selected_decision": selected_decision,
                        "counterfactual_family": states[trial_id][
                            current_view
                        ].counterfactual_family,
                        "counterfactual_scores": dict(
                            states[trial_id][current_view].counterfactual_scores
                        ),
                        "counterfactual_risk": max(
                            states[trial_id][
                                current_view
                            ].counterfactual_scores.values(),
                            default=0.0,
                        ),
                    }
                )
        all_rows[variant] = rows
        all_results[variant] = {
            "all_start_views": aggregate(rows, triggered_only=False),
            "inspection_trigger_subset": aggregate(rows, triggered_only=True),
        }

    for action in ORBIT_ACTIONS:
        rows = []
        for trial_id, view_predictions in sorted(predictions.items()):
            gt = gt_by_trial[trial_id]
            truth = str(gt.get("claim_outcome", "")).strip().lower()
            oracle_utility = max(utilities[trial_id].values(), default=0)
            for current_view, current_prediction in sorted(view_predictions.items()):
                current_decision = str(current_prediction["decision"])
                triggered = current_decision == "insufficient"
                selected_view = (
                    fixed_action_select(
                        action=action,
                        current_view=current_view,
                        lambda_cost=float(args.lambda_cost),
                    )
                    if triggered
                    else current_view
                )
                selected_prediction = view_predictions[selected_view]
                selected_decision = str(selected_prediction["decision"])
                current_utility = utilities[trial_id][current_view]
                selected_utility = utilities[trial_id][selected_view]
                rows.append(
                    {
                        "trial_id": trial_id,
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
                        "fixed_action": action,
                    }
                )
        name = f"Fixed Action: {action}"
        all_rows[name] = rows
        all_results[name] = {
            "all_start_views": aggregate(rows, triggered_only=False),
            "inspection_trigger_subset": aggregate(rows, triggered_only=True),
        }

    # Select a single fixed action using robot-utility calibration setups only.
    # The held-out setup never contributes to its own action choice.
    calibrated_fixed_rows: List[Dict[str, Any]] = []
    fixed_rows_by_action = {
        action: all_rows[f"Fixed Action: {action}"] for action in ORBIT_ACTIONS
    }
    for test_trial in sorted(predictions):
        ranked_actions: List[tuple[float, float, str]] = []
        for action in ORBIT_ACTIONS:
            training_rows = [
                row
                for row in fixed_rows_by_action[action]
                if str(row["trial_id"]) != test_trial and bool(row["triggered"])
            ]
            selected_utility = mean(
                float(row["selected_utility"]) for row in training_rows
            )
            gain = mean(float(row["gain"]) for row in training_rows)
            ranked_actions.append((selected_utility, gain, action))
        selected_action = max(
            ranked_actions, key=lambda item: (item[0], item[1], item[2])
        )[2]
        for row in fixed_rows_by_action[selected_action]:
            if str(row["trial_id"]) == test_trial:
                calibrated_fixed_rows.append(
                    {
                        **row,
                        "fixed_action": selected_action,
                        "selection_protocol": "leave-one-setup-out",
                    }
                )
    all_results["Calibrated Fixed Action (setup-disjoint)"] = {
        "all_start_views": aggregate(calibrated_fixed_rows, triggered_only=False),
        "inspection_trigger_subset": aggregate(
            calibrated_fixed_rows, triggered_only=True
        ),
    }
    all_rows["Calibrated Fixed Action (setup-disjoint)"] = calibrated_fixed_rows

    knn_rows: List[Dict[str, Any]] = []
    for trial_id, view_predictions in sorted(predictions.items()):
        gt = gt_by_trial[trial_id]
        truth = str(gt.get("claim_outcome", "")).strip().lower()
        oracle_utility = max(utilities[trial_id].values(), default=0)
        for current_view, current_prediction in sorted(view_predictions.items()):
            current_decision = str(current_prediction["decision"])
            triggered = current_decision == "insufficient"
            if triggered:
                selected_view = knn_select(
                    test_trial=trial_id,
                    current_view=current_view,
                    state=states[trial_id][current_view],
                    states=states,
                    utilities=utilities,
                    k=int(args.knn_k),
                    lambda_cost=float(args.lambda_cost),
                )
            else:
                selected_view = current_view
            selected_prediction = view_predictions[selected_view]
            selected_decision = str(selected_prediction["decision"])
            current_utility = utilities[trial_id][current_view]
            selected_utility = utilities[trial_id][selected_view]
            knn_rows.append(
                {
                    "trial_id": trial_id,
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
                    "counterfactual_family": states[trial_id][
                        current_view
                    ].counterfactual_family,
                    "counterfactual_scores": dict(
                        states[trial_id][current_view].counterfactual_scores
                    ),
                    "counterfactual_risk": max(
                        states[trial_id][current_view].counterfactual_scores.values(),
                        default=0.0,
                    ),
                }
            )
    all_results["Robot-Utility kNN (setup-disjoint)"] = {
        "all_start_views": aggregate(knn_rows, triggered_only=False),
        "inspection_trigger_subset": aggregate(knn_rows, triggered_only=True),
    }
    all_rows["Robot-Utility kNN (setup-disjoint)"] = knn_rows

    payload = {
        "protocol": {
            "candidate_view_images_hidden": True,
            "assistant_policy_uses_robot_labels": False,
            "robot_knn_uses_robot_utility_labels": True,
            "robot_knn_split": "leave-one-setup-out",
            "calibrated_fixed_action_uses_robot_utility_labels": True,
            "calibrated_fixed_action_split": "leave-one-setup-out",
            "per_action_fixed_rows_are_post_hoc_diagnostics": True,
            "inspection_trigger": "current claim decision is insufficient",
            "evaluation_utility_source": utility_source,
            "evaluation_utility_used_by_policy": False,
            "setups": len(predictions),
            "start_views": sum(len(values) for values in predictions.values()),
            "input_fingerprints": {
                "observations": file_fingerprint(args.observations),
                "trial_gt": file_fingerprint(args.trial_gt),
                "evidence_scorer": file_fingerprint(args.evidence_scorer),
                "full_reveal_model": file_fingerprint(args.full_reveal_model),
                "no_gate_reveal_model": file_fingerprint(args.no_gate_reveal_model),
                "requirement_report": file_fingerprint(args.requirement_report),
                "no_gate_requirement_report": file_fingerprint(
                    args.no_gate_requirement_report or args.requirement_report
                ),
                "candidate_affordance": file_fingerprint(
                    args.candidate_affordance_jsonl
                ),
                "evaluation_utility": file_fingerprint(args.evaluation_utility_csv),
            },
            "decision_thresholds": {
                "support": float(args.support_threshold),
                "contradiction": float(args.contradiction_threshold),
                "margin": float(args.margin_threshold),
                "role": float(args.role_threshold),
                "partial": float(args.partial_threshold),
                "tau_view": float(args.tau_view),
            },
            "lambda_cost": float(args.lambda_cost),
            "requirement_beta": requirement_beta,
            "requirement_strength": requirement_strength,
            "requirement_temperature": requirement_temperature,
            "requirement_conditioning": (
                "claim_and_counterfactual_family"
                if args.counterfactual_conditioned_requirement
                else "claim_only"
            ),
            "requirement_calibration": (
                file_fingerprint(args.requirement_calibration_json)
                if args.requirement_calibration_json is not None
                else None
            ),
            "requirement_normalization": (
                "matched missing-role mass across ablations; strength=0 assigns "
                "uniform role mass, while strength=1 uses the episode-balanced "
                "ontology-plus-trace posterior selected on held-out assistant videos"
            ),
            "preserve_partial_evidence": bool(args.preserve_partial_evidence),
            "current_view_ray_affordance_cache": (
                str(args.candidate_affordance_jsonl)
                if args.candidate_affordance_jsonl is not None
                else None
            ),
            "current_view_ray_records": len(ray_affordances),
            "ray_affordance_uses_candidate_images": False,
            "ray_affordance_uses_robot_utility_labels": False,
        },
        "requirement_supervision": {
            "source": str(args.requirement_report),
            "no_gate_source": str(
                args.no_gate_requirement_report or args.requirement_report
            ),
            "learned_claims": learned_r,
            "beta": requirement_beta,
            "strength": requirement_strength,
            "temperature": requirement_temperature,
            "conditioning": (
                "claim_and_counterfactual_family"
                if args.counterfactual_conditioned_requirement
                else "claim_only"
            ),
        },
        "results": all_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.output_rows_json is not None:
        args.output_rows_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_rows_json.write_text(
            json.dumps(all_rows, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
