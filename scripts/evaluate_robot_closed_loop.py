"""Evaluate one-step fixed-lattice inspection from frozen RGB observations.

The online policy receives only the active claim, the current-view evidence
state, and lattice geometry. Candidate observations and ground-truth outcomes
are accessed only after a view has been selected, for re-verification and
evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspect_system.active_view.active_selector import ActiveSelector
from inspect_system.active_view.evidence_state import EvidenceItem, EvidenceState
from inspect_system.active_view.ontology import normalize_claim, roles_for_claim
from inspect_system.active_view.reveal_model import PriorTableRevealModel
from inspect_system.evidence_scorer import (
    HOUSING_CLASSES,
    PrototypeEvidenceScorer,
    expected_class,
    extract_relation_features,
    target_role_for,
)


OUTCOMES = ("supported", "contradicted", "insufficient")
ACTIVE_STEPS = {"step2", "step3", "step4"}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(dict(json.loads(line)))
    return rows


def load_requirement_weights(path: Path | None) -> Dict[str, Dict[str, float]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_claim_role: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for event in payload.get("trainable_events") or []:
        if not isinstance(event, Mapping):
            continue
        claim = normalize_claim(event.get("claim_id", ""))
        role = str(event.get("evidence_role", ""))
        weight = (
            max(0.0, float(event.get("transfer_weight", 0.0) or 0.0))
            * max(0.0, float(event.get("label_confidence", 0.0) or 0.0))
            * max(0.0, float(event.get("evidence_stability", 0.0) or 0.0))
            * max(0.0, float(event.get("evidence_importance", 0.0) or 0.0))
        )
        if claim and role and weight > 0.0:
            by_claim_role[claim][role] += weight
    return {claim: dict(values) for claim, values in by_claim_role.items()}


def apply_requirement_weights(
    state: EvidenceState,
    learned: Mapping[str, Mapping[str, float]],
    *,
    beta: float,
) -> EvidenceState:
    if not learned:
        return state
    base_total = sum(max(0.0, float(item.importance)) for item in state.items)
    learned_values = learned.get(normalize_claim(state.claim_id), {})
    event_mass = sum(max(0.0, float(value)) for value in learned_values.values())
    items: List[EvidenceItem] = []
    for item in state.items:
        prior = max(0.0, float(item.importance)) / max(1e-9, base_total)
        learned_value = max(
            0.0,
            float(learned_values.get(item.evidence_role or item.name, 0.0)),
        )
        importance = (beta * prior + learned_value) / max(1e-9, beta + event_mass)
        items.append(
            replace(
                item,
                importance=importance,
                score=(item.threshold if item.observed else 0.0),
            )
        )
    return EvidenceState(
        claim_id=state.claim_id,
        items=items,
        claim_score=state.claim_score,
        contradiction_score=state.contradiction_score,
        margin=state.margin,
        current_utility_proxy=None,
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _step_state(step: object) -> str:
    text = str(step or "").strip().lower()
    return {"step1": "S1", "step2": "S2", "step3": "S3", "step4": "S4"}.get(text, text.upper())


def _relation_score(
    observation: Mapping[str, Any],
    target: str,
    housing: str,
    predicate_weights: Mapping[str, float],
) -> float:
    metadata = dict(observation.get("metadata") or {})
    scene = dict(metadata.get("scene_evidence") or {})
    best = 0.0
    housing_names = set(HOUSING_CLASSES)
    if housing:
        housing_names.add(housing)
    for relation in scene.get("relations") or []:
        if not isinstance(relation, Mapping):
            continue
        subject = str(relation.get("subject", ""))
        obj = str(relation.get("object", ""))
        predicate = str(relation.get("predicate", ""))
        if not ((subject == target and obj in housing_names) or (obj == target and subject in housing_names)):
            continue
        weight = float(predicate_weights.get(predicate, 0.0))
        best = max(best, weight * float(relation.get("score", 0.0) or 0.0))
    return _clamp(best)


def claim_evidence(
    observation: Mapping[str, Any],
    *,
    gt: Mapping[str, str],
    scorer: PrototypeEvidenceScorer,
    support_threshold: float,
    contradiction_threshold: float,
    margin_threshold: float,
    role_threshold: float,
) -> Tuple[EvidenceState, Dict[str, Any]]:
    metadata = dict(observation.get("metadata") or {})
    detections = list(metadata.get("detections") or [])
    frame_shape = list(metadata.get("frame_shape") or [1, 1])
    step = str(gt.get("target_step", ""))
    state_id = _step_state(step)
    claim = normalize_claim(gt.get("claim_id") or step)
    product = str(gt.get("product_variant", ""))
    features = extract_relation_features(
        detections,
        claim_id=gt.get("claim_id", ""),
        step_id=state_id,
        product=product,
        image_shape=(int(frame_shape[0]), int(frame_shape[1])),
        role_detections=list(metadata.get("role_detections") or []),
    )
    scores = scorer.score_features(features, claim_id=gt.get("claim_id", ""), step_id=state_id)
    support = float(scores.get("support_score", 0.0))
    contradiction = float(scores.get("contradiction_score", 0.0))
    margin = support - contradiction

    role = target_role_for(gt.get("claim_id", ""), state_id)
    target = expected_class(product, role)
    housing = expected_class(product, "housing")
    insertion_relation = _relation_score(
        observation,
        target,
        housing,
        {"inside": 1.0, "overlapping": 0.75, "contacting": 0.65, "near": 0.25},
    )
    alignment_relation = _relation_score(
        observation,
        target,
        housing,
        {"aligned_with": 1.0, "inside": 0.85, "overlapping": 0.65, "contacting": 0.55, "near": 0.20},
    )
    contact_relation = _relation_score(
        observation,
        target,
        housing,
        {"contacting": 1.0, "overlapping": 0.80, "near": 0.45, "inside": 0.70},
    )
    identity_score = _clamp(
        float(features.get("target_conf", 0.0))
        * _clamp((float(features.get("identity_margin", 0.0)) + 0.20) / 0.40)
    )
    visibility = _clamp(float(scores.get("visibility_score", 0.0)))
    containment = max(_clamp(float(features.get("containment_score", 0.0))), insertion_relation)
    gap_visibility = _clamp(
        min(float(features.get("target_conf", 0.0)), float(features.get("housing_role_conf", 0.0)))
        * (1.0 - _clamp(float(features.get("edge_gap_norm", 1.0))))
    )
    disambiguation = max(support, contradiction, 0.5 * visibility)

    role_scores = {
        "identity_disambiguation_view": identity_score,
        "insertion_verification_view": max(containment, insertion_relation),
        "containment_verification_view": containment,
        "slot_relation_view": max(alignment_relation, insertion_relation),
        "gap_visibility_view": max(gap_visibility, contact_relation),
        "boundary_alignment_view": max(alignment_relation, gap_visibility),
        "contact_verification_view": contact_relation,
        "claim_disambiguation_view": disambiguation,
    }
    role_importance = roles_for_claim(claim)
    if claim == "gear_inserted":
        role_importance = {"identity_disambiguation_view": 1.0, **role_importance}
    items = [
        EvidenceItem(
            name=name,
            evidence_role=name,
            score=float(role_scores.get(name, 0.0)),
            threshold=float(role_threshold),
            importance=float(importance),
        )
        for name, importance in role_importance.items()
    ]
    if contradiction >= contradiction_threshold and contradiction >= support + margin_threshold:
        decision = "contradicted"
    elif support >= support_threshold and support >= contradiction + margin_threshold:
        decision = "supported"
    else:
        decision = "insufficient"
    wrong_identity_score = _clamp(float(features.get("wrong_same_role_conf", 0.0)))
    relation_counterfactual_score = _clamp(
        min(float(features.get("target_conf", 0.0)), float(features.get("housing_role_conf", 0.0)))
        * (1.0 - max(insertion_relation, alignment_relation, contact_relation))
    )
    relation_family = "seating_contact" if claim == "cover_seated" else "spatial_relation"
    counterfactual_scores = {
        "identity": wrong_identity_score,
        relation_family: relation_counterfactual_score,
    }
    counterfactual_family = ""
    if max(counterfactual_scores.values(), default=0.0) > 1e-12:
        counterfactual_family = max(
            counterfactual_scores,
            key=lambda key: (counterfactual_scores[key], key),
        )
    state = EvidenceState(
        claim_id=claim,
        items=items,
        claim_score=support,
        contradiction_score=contradiction,
        margin=margin,
        counterfactual_family=counterfactual_family,
        counterfactual_scores=counterfactual_scores,
        current_utility_proxy=None,
    )
    return state, {
        "decision": decision,
        "support_score": support,
        "contradiction_score": contradiction,
        "counterfactual_margin": margin,
        "counterfactual_family": counterfactual_family,
        "counterfactual_scores": counterfactual_scores,
        "visibility_score": visibility,
        "features": features,
        "role_scores": role_scores,
        "missing_roles": [item.evidence_role for item in items if not item.observed],
        "policy_input_keys": ["active_claim", "current_view_evidence", "lattice_geometry"],
        "uses_candidate_observations": False,
        "uses_trial_outcome": False,
    }


def evaluation_utility(prediction: Mapping[str, Any], truth: str, partial_threshold: float) -> int:
    decision = str(prediction.get("decision", "insufficient"))
    if decision in {"supported", "contradicted"}:
        return 2 if decision == truth else 0
    role_scores = dict(prediction.get("role_scores") or {})
    return 1 if max((float(value) for value in role_scores.values()), default=0.0) >= partial_threshold else 0


def summarize_trials(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    numeric = [
        "current_utility",
        "selected_utility",
        "oracle_utility",
        "gain",
        "regret",
        "resolve_at_1",
        "correct_resolution",
        "false_support",
        "false_contradiction",
        "defer",
        "moved",
        "stop_accuracy",
        "false_stay",
        "false_move",
        "no_improvement",
    ]
    return {
        "trials": len(rows),
        **{key: mean(float(row.get(key, 0.0)) for row in rows) for key in numeric},
        "decision_counts": dict(Counter(str(row.get("selected_decision", "")) for row in rows)),
    }


def cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> Dict[str, List[float]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("trial_id", ""))].append(row)
    keys = sorted(grouped)
    rng = random.Random(seed)
    values: Dict[str, List[float]] = {metric: [] for metric in metrics}
    for _ in range(max(1, samples)):
        sampled = [item for _key in (rng.choice(keys) for _ in keys) for item in grouped[_key]]
        for metric in metrics:
            values[metric].append(mean(float(item.get(metric, 0.0)) for item in sampled))
    intervals: Dict[str, List[float]] = {}
    for metric, metric_values in values.items():
        metric_values.sort()
        lo = metric_values[int(0.025 * (len(metric_values) - 1))]
        hi = metric_values[int(0.975 * (len(metric_values) - 1))]
        intervals[metric] = [float(lo), float(hi)]
    return intervals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--trial-gt", type=Path, required=True)
    parser.add_argument("--evidence-scorer", type=Path, required=True)
    parser.add_argument("--reveal-model", type=Path, required=True)
    parser.add_argument("--requirement-report", type=Path, default=None)
    parser.add_argument("--requirement-beta", type=float, default=2.0)
    parser.add_argument(
        "--current-geometry-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional current-view-only MoGe/ray-affordance cache. The cache "
            "must not contain candidate-view images or robot utility labels."
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--view-output-csv", type=Path, required=True)
    parser.add_argument("--support-threshold", type=float, default=0.35)
    parser.add_argument("--contradiction-threshold", type=float, default=0.45)
    parser.add_argument("--margin-threshold", type=float, default=0.05)
    parser.add_argument("--role-threshold", type=float, default=0.65)
    parser.add_argument("--partial-threshold", type=float, default=0.35)
    parser.add_argument("--lambda-cost", type=float, default=0.25)
    parser.add_argument("--tau-view", type=float, default=0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    gt_by_trial = {row["trial_id"]: row for row in read_csv(args.trial_gt)}
    observations: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for observation in read_jsonl(args.observations):
        metadata = dict(observation.get("metadata") or {})
        trial_id = str(metadata.get("trial_id", ""))
        view_id = str(observation.get("view_id", ""))
        if trial_id and view_id:
            observations[trial_id][view_id] = observation

    scorer = PrototypeEvidenceScorer.load(args.evidence_scorer)
    reveal_model = PriorTableRevealModel.load(args.reveal_model)
    learned_r = load_requirement_weights(args.requirement_report)
    current_geometry: Dict[str, Dict[str, Any]] = {}
    if args.current_geometry_jsonl is not None:
        for item in read_jsonl(args.current_geometry_jsonl):
            if bool(item.get("candidate_images_used", False)):
                raise ValueError("Current-view geometry cache contains candidate images.")
            if bool(item.get("robot_utility_labels_used", False)):
                raise ValueError("Current-view geometry cache contains robot utility labels.")
            key = str(item.get("observation_id", "")).strip()
            if not key:
                key = (
                    f"{str(item.get('trial_id', '')).strip()}_"
                    f"{str(item.get('current_view', '')).strip()}"
                )
            if key.strip("_"):
                current_geometry[key] = dict(item)
    selector = ActiveSelector(
        model=reveal_model,
        lambda_cost=float(args.lambda_cost),
        tau_view=float(args.tau_view),
    )

    view_rows: List[Dict[str, Any]] = []
    predictions: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    utilities: Dict[str, Dict[str, int]] = defaultdict(dict)
    for trial_id, view_map in sorted(observations.items()):
        gt = gt_by_trial.get(trial_id, {})
        if str(gt.get("target_step", "")).lower() not in ACTIVE_STEPS:
            continue
        truth = str(gt.get("claim_outcome", "")).strip().lower()
        for view_id, observation in sorted(view_map.items()):
            state, prediction = claim_evidence(
                observation,
                gt=gt,
                scorer=scorer,
                support_threshold=float(args.support_threshold),
                contradiction_threshold=float(args.contradiction_threshold),
                margin_threshold=float(args.margin_threshold),
                role_threshold=float(args.role_threshold),
            )
            prediction["evidence_state"] = state.to_dict()
            utility = evaluation_utility(prediction, truth, float(args.partial_threshold))
            predictions[trial_id][view_id] = prediction
            utilities[trial_id][view_id] = utility
            view_rows.append(
                {
                    "trial_id": trial_id,
                    "view_id": view_id,
                    "target_step": gt.get("target_step", ""),
                    "claim_id": gt.get("claim_id", ""),
                    "truth": truth,
                    "prediction": prediction["decision"],
                    "support_score": prediction["support_score"],
                    "contradiction_score": prediction["contradiction_score"],
                    "counterfactual_margin": prediction["counterfactual_margin"],
                    "missing_roles": ";".join(prediction["missing_roles"]),
                    "evaluation_utility": utility,
                    "uses_candidate_observations": 0,
                    "uses_trial_outcome": 0,
                }
            )

    trial_rows: List[Dict[str, Any]] = []
    for trial_id, view_predictions in sorted(predictions.items()):
        gt = gt_by_trial[trial_id]
        truth = str(gt.get("claim_outcome", "")).strip().lower()
        oracle_utility = max(utilities[trial_id].values(), default=0)
        for current_view, current_prediction in sorted(view_predictions.items()):
            current_state = EvidenceState.from_dict(current_prediction["evidence_state"])
            current_state = apply_requirement_weights(
                current_state,
                learned_r,
                beta=float(args.requirement_beta),
            )
            current_decision = str(current_prediction["decision"])
            if current_decision != "insufficient":
                action = "stay"
                selected_view = current_view
                reason = "current_claim_resolved"
                policy_score = 0.0
            else:
                geometry = current_geometry.get(f"{trial_id}_{current_view}", {})
                candidate_context = None
                if geometry:
                    candidate_context = {
                        "candidate_role_factors": dict(
                            geometry.get("candidate_role_factors") or {}
                        ),
                        "surface_normal_world": geometry.get("surface_normal_world"),
                        "role_surface_normals_world": dict(
                            geometry.get("role_surface_normals_world") or {}
                        ),
                        "role_relation_frames_world": dict(
                            geometry.get("role_relation_frames_world") or {}
                        ),
                    }
                selection = selector.select(
                    current_view=current_view,
                    evidence_state=current_state,
                    candidate_context=candidate_context,
                )
                action = selection.action
                selected_view = selection.selected_view if selection.action == "move" else current_view
                reason = selection.reason
                policy_score = float(selection.score)
            selected_prediction = view_predictions[selected_view]
            selected_decision = str(selected_prediction["decision"])
            current_utility = utilities[trial_id][current_view]
            selected_utility = utilities[trial_id][selected_view]
            should_stay = current_utility == 2
            better_exists = oracle_utility > current_utility
            trial_rows.append(
                {
                    "trial_id": trial_id,
                    "target_step": gt.get("target_step", ""),
                    "claim_id": gt.get("claim_id", ""),
                    "truth": truth,
                    "current_view": current_view,
                    "current_decision": current_decision,
                    "action": action,
                    "selected_view": selected_view,
                    "selected_decision": selected_decision,
                    "current_utility": current_utility,
                    "selected_utility": selected_utility,
                    "oracle_utility": oracle_utility,
                    "gain": selected_utility - current_utility,
                    "regret": oracle_utility - selected_utility,
                    "resolve_at_1": int(selected_utility == 2),
                    "correct_resolution": int(selected_decision == truth),
                    "false_support": int(selected_decision == "supported" and truth != "supported"),
                    "false_contradiction": int(selected_decision == "contradicted" and truth != "contradicted"),
                    "defer": int(selected_decision == "insufficient"),
                    "moved": int(selected_view != current_view),
                    "stop_accuracy": int(should_stay and selected_view == current_view),
                    "false_stay": int(better_exists and selected_view == current_view),
                    "false_move": int(should_stay and selected_view != current_view),
                    "no_improvement": int(not should_stay and selected_utility <= current_utility),
                    "policy_score": policy_score,
                    "reason": reason,
                    "current_missing_roles": ";".join(current_prediction["missing_roles"]),
                    "selected_missing_roles": ";".join(selected_prediction["missing_roles"]),
                }
            )

    summary = summarize_trials(trial_rows)
    ci_metrics = ["selected_utility", "gain", "resolve_at_1", "correct_resolution", "false_support", "defer"]
    summary["setup_cluster_bootstrap_95ci"] = cluster_bootstrap(
        trial_rows,
        ci_metrics,
        samples=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    summary.update(
        {
            "setups": len(predictions),
            "views": len(view_rows),
            "protocol": "one-step closed-loop RGB replay with selected-view re-verification",
            "policy_inputs": [
                "active_claim",
                "current_view_evidence",
                "lattice_geometry",
                *(["current_view_geometry"] if current_geometry else []),
            ],
            "candidate_view_images_hidden": True,
            "ground_truth_used_by_policy": False,
            "utility_definition": "2=correct resolved claim, 1=insufficient with partial role evidence, 0=wrong or no usable evidence",
            "thresholds": {
                "support": args.support_threshold,
                "contradiction": args.contradiction_threshold,
                "counterfactual_margin": args.margin_threshold,
                "role": args.role_threshold,
                "partial": args.partial_threshold,
            },
            "observations": str(args.observations),
            "evidence_scorer": str(args.evidence_scorer),
            "reveal_model": str(args.reveal_model),
            "requirement_report": (
                str(args.requirement_report) if args.requirement_report is not None else None
            ),
            "requirement_beta": args.requirement_beta,
            "lambda_cost": args.lambda_cost,
            "tau_view": args.tau_view,
            "current_geometry_jsonl": (
                str(args.current_geometry_jsonl)
                if args.current_geometry_jsonl is not None
                else None
            ),
            "current_geometry_records": len(current_geometry),
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    trial_fields = list(trial_rows[0].keys()) if trial_rows else []
    view_fields = list(view_rows[0].keys()) if view_rows else []
    write_csv(args.output_csv, trial_rows, trial_fields)
    write_csv(args.view_output_csv, view_rows, view_fields)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
