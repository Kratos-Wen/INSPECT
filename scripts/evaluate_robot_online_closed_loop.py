"""Evaluate causal online view adaptation in a two-move robot replay.

Every start view is an independent cold-start session. The policy may update
only after moving to a selected view and re-running the frozen verifier.
Candidate images and robot utility annotations are never policy inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_robot_closed_loop as base
import evaluate_robot_closed_loop_gated as gated
from evaluate_decidability_transport_policy import POLICY_SPEC_KEYS
from inspect_system.learned_view_lattice_policy import LearnedViewLatticePolicy
from inspect_system.types import RobotObservation, VerificationResult


VIEWS = tuple(f"V{index}" for index in range(6))
GEOMETRY_KEYS = (
    "surface_normal_world",
    "role_surface_normals_world",
    "role_relation_frames_world",
)


def fingerprint(path: Path | None) -> Dict[str, Any] | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def runtime_inputs(
    raw: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    session_id: str,
    role_threshold: float,
    geometry: Mapping[str, Any] | None,
) -> tuple[RobotObservation, VerificationResult]:
    role_scores = {
        str(key): float(value)
        for key, value in dict(prediction.get("role_scores") or {}).items()
    }
    missing = [str(value) for value in prediction.get("missing_roles") or []]
    observed = [role for role, score in role_scores.items() if score >= role_threshold]
    decision = str(prediction.get("decision", "insufficient"))
    metadata = {
        **dict(raw.get("metadata") or {}),
        "session_id": session_id,
        "claim_id": str(prediction.get("claim_id", "")),
        "support_score": float(prediction.get("support_score", 0.0)),
        "contradiction_score": float(prediction.get("contradiction_score", 0.0)),
        "counterfactual_margin": float(prediction.get("counterfactual_margin", 0.0)),
        "counterfactual_family": str(prediction.get("counterfactual_family", "")),
        "counterfactual_scores": dict(prediction.get("counterfactual_scores") or {}),
        "role_scores": role_scores,
        "missing_roles": missing,
        "decision": decision,
        "identity_available": bool(prediction.get("identity_available", False)),
        "relation_available": bool(prediction.get("relation_available", False)),
        "wrong_identity_visible": bool(
            prediction.get("wrong_identity_visible", False)
        ),
        "incomplete_relation_visible": bool(
            prediction.get("incomplete_relation_visible", False)
        ),
        "support_evidence_available": bool(
            prediction.get("support_evidence_available", False)
        ),
        "contradiction_evidence_available": bool(
            prediction.get("contradiction_evidence_available", False)
        ),
        "role_threshold": float(role_threshold),
        "uses_candidate_view_images": False,
        "uses_robot_utility_labels": False,
    }
    for key in GEOMETRY_KEYS:
        value = (geometry or {}).get(key)
        if value not in (None, {}, []):
            metadata[key] = value
    role_factors = dict((geometry or {}).get("candidate_role_factors") or {})
    if role_factors:
        metadata["preservation_role_factors"] = role_factors
    observation_payload = {**dict(raw), "metadata": metadata}
    observation = RobotObservation.from_dict(observation_payload)
    confidence = max(
        float(prediction.get("support_score", 0.0)),
        float(prediction.get("contradiction_score", 0.0)),
    )
    coverage = float(prediction.get("visibility_score", 0.0))
    verification = VerificationResult(
        observation_id=observation.observation_id,
        predicted_state=str(prediction.get("claim_id", "")),
        verified=decision in {"supported", "contradicted"},
        confidence=confidence,
        evidence_coverage=coverage,
        observed_evidence=observed,
        missing_evidence=missing,
        contradicted_evidence=(missing if decision == "contradicted" else []),
        next_step_admissible=decision == "supported",
        anomaly=decision == "contradicted",
        recommended_action=(
            "observe"
            if decision == "insufficient"
            else "continue" if decision == "supported" else "pause"
        ),
        metadata=metadata,
    )
    return observation, verification


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metrics = (
        "current_utility",
        "utility_at_1",
        "final_utility",
        "gain_at_1",
        "final_gain",
        "retained_utility",
        "retained_gain",
        "retained_resolve",
        "retained_regret",
        "resolve_at_1",
        "resolve_at_2",
        "regret_at_2",
        "moves",
        "online_updates",
        "no_improvement",
        "correct_resolution",
        "false_support",
        "false_contradiction",
    )
    return {
        "trials": len(rows),
        **{metric: mean(float(row[metric]) for row in rows) for metric in metrics},
        "defer": mean(float(row["final_decision"] == "insufficient") for row in rows),
        "final_decisions": dict(Counter(str(row["final_decision"]) for row in rows)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--trial-gt", type=Path, required=True)
    parser.add_argument("--evidence-scorer", type=Path, required=True)
    parser.add_argument("--reveal-model", type=Path, required=True)
    parser.add_argument("--requirement-report", type=Path, required=True)
    parser.add_argument("--requirement-calibration", type=Path, required=True)
    parser.add_argument(
        "--counterfactual-conditioned-requirement",
        action="store_true",
        help="Condition R(c,q) on the evidence-derived counterfactual family.",
    )
    parser.add_argument("--current-geometry-jsonl", type=Path, default=None)
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
    parser.add_argument("--max-moves", type=int, default=2)
    parser.add_argument("--online-update-weight", type=float, default=1.0)
    parser.add_argument("--online-requirement-scale", type=float, default=1.0)
    parser.add_argument("--online-requirement-blend", type=float, default=1.0)
    parser.add_argument("--online-requirement-prior-strength", type=float, default=0.05)
    parser.add_argument("--online-gain-deadband", type=float, default=0.05)
    parser.add_argument("--include-claim-belief", action="store_true")
    parser.add_argument("--claim-structured", action="store_true")
    parser.add_argument("--typed-requirement", action="store_true")
    parser.add_argument("--unified-claim-evidence-ledger", action="store_true")
    parser.add_argument("--pareto-evidence-checkpoint", action="store_true")
    parser.add_argument("--belief-conflict-threshold", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_by_trial = {row["trial_id"]: row for row in base.read_csv(args.trial_gt)}
    raw_observations: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for raw in base.read_jsonl(args.observations):
        metadata = dict(raw.get("metadata") or {})
        trial_id = str(metadata.get("trial_id", ""))
        view_id = str(raw.get("view_id", ""))
        if trial_id and view_id:
            raw_observations[trial_id][view_id] = raw

    geometry: Dict[str, Mapping[str, Any]] = {}
    if args.current_geometry_jsonl is not None:
        for row in base.read_jsonl(args.current_geometry_jsonl):
            if bool(row.get("candidate_images_used", False)):
                raise RuntimeError("Current-geometry cache used candidate images.")
            if bool(row.get("robot_utility_labels_used", False)):
                raise RuntimeError("Current-geometry cache used robot utility labels.")
            observation_id = str(row.get("observation_id", ""))
            if observation_id and str(row.get("status", "")) == "ok":
                geometry[observation_id] = row

    scorer = base.PrototypeEvidenceScorer.load(args.evidence_scorer)
    predictions: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    utilities: Dict[str, Dict[str, int]] = defaultdict(dict)
    for trial_id, view_map in sorted(raw_observations.items()):
        gt = gt_by_trial.get(trial_id, {})
        if str(gt.get("target_step", "")).lower() not in base.ACTIVE_STEPS:
            continue
        truth = str(gt.get("claim_outcome", "")).strip().lower()
        policy_spec = {key: str(gt.get(key, "")) for key in POLICY_SPEC_KEYS}
        for view_id, raw in sorted(view_map.items()):
            state, prediction = gated.gated_claim_evidence(
                raw,
                gt=policy_spec,
                scorer=scorer,
                support_threshold=float(args.support_threshold),
                contradiction_threshold=float(args.contradiction_threshold),
                margin_threshold=float(args.margin_threshold),
                role_threshold=float(args.role_threshold),
            )
            prediction = dict(prediction)
            prediction["claim_id"] = (
                str(policy_spec.get("claim_id", "")).strip() or state.claim_id
            )
            predictions[trial_id][view_id] = prediction
            utilities[trial_id][view_id] = base.evaluation_utility(
                prediction,
                truth,
                float(args.partial_threshold),
            )

    loaded = 0
    for row in base.read_csv(args.evaluation_utility_csv):
        trial_id = str(row.get("trial_id", ""))
        view_id = str(row.get("view_id", ""))
        value = str(row.get(args.evaluation_utility_field, "")).strip()
        if (
            trial_id in predictions
            and view_id in predictions[trial_id]
            and value in {"0", "1", "2"}
        ):
            utilities[trial_id][view_id] = int(value)
            loaded += 1
    expected = sum(len(view_map) for view_map in predictions.values())
    if loaded != expected:
        raise RuntimeError(
            f"Expected {expected} evaluation utilities, loaded {loaded}."
        )

    variant_specs = {
        "pi-only frozen": {
            "strength": 0.0,
            "calibration": None,
            "online": False,
            "requirement_scale": 0.0,
            "requirement_blend": 0.0,
            "destination_strength": 0.0,
        },
        "R+pi frozen": {
            "strength": 1.0,
            "calibration": args.requirement_calibration,
            "online": False,
            "requirement_scale": args.online_requirement_scale,
            "requirement_blend": args.online_requirement_blend,
            "destination_strength": 0.0,
        },
        "pi-only relative-online": {
            "strength": 0.0,
            "calibration": None,
            "online": True,
            "requirement_scale": 0.0,
            "requirement_blend": 0.0,
            "destination_strength": 0.0,
        },
        "pi cold-start + causal online R": {
            "strength": 0.0,
            "calibration": None,
            "online": True,
            "requirement_scale": args.online_requirement_scale,
            "requirement_blend": args.online_requirement_blend,
            "destination_strength": 0.0,
        },
        "R+pi relative-online": {
            "strength": 1.0,
            "calibration": args.requirement_calibration,
            "online": True,
            "requirement_scale": args.online_requirement_scale,
            "requirement_blend": args.online_requirement_blend,
            "destination_strength": 0.0,
        },
        "INSPECT online": {
            "strength": 1.0,
            "calibration": args.requirement_calibration,
            "online": True,
            "requirement_scale": args.online_requirement_scale,
            "requirement_blend": args.online_requirement_blend,
            "destination_strength": 1.0,
        },
        "INSPECT adaptive online": {
            "strength": 0.0,
            "calibration": None,
            "online": True,
            "requirement_scale": args.online_requirement_scale,
            "requirement_blend": args.online_requirement_blend,
            "destination_strength": 1.0,
        },
    }
    if args.include_claim_belief:
        variant_specs["INSPECT online + counterfactual evidence belief"] = {
            "strength": 1.0,
            "calibration": args.requirement_calibration,
            "online": True,
            "requirement_scale": args.online_requirement_scale,
            "requirement_blend": args.online_requirement_blend,
            "destination_strength": 1.0,
            "claim_belief": True,
        }
    if args.unified_claim_evidence_ledger:
        variant_specs["INSPECT online + unified claim ledger"] = {
            "strength": 1.0,
            "calibration": args.requirement_calibration,
            "online": True,
            "requirement_scale": args.online_requirement_scale,
            "requirement_blend": args.online_requirement_blend,
            "destination_strength": 1.0,
            "unified_claim_evidence_ledger": True,
        }
        if args.pareto_evidence_checkpoint:
            variant_specs[
                "INSPECT online + unified claim ledger + evidence dominance"
            ] = {
                "strength": 1.0,
                "calibration": args.requirement_calibration,
                "online": True,
                "requirement_scale": args.online_requirement_scale,
                "requirement_blend": args.online_requirement_blend,
                "destination_strength": 1.0,
                "unified_claim_evidence_ledger": True,
                "pareto_evidence_checkpoint": True,
            }
    results: Dict[str, Any] = {}
    rows_by_variant: Dict[str, list[Dict[str, Any]]] = {}
    for variant, spec in variant_specs.items():
        policy = LearnedViewLatticePolicy.from_paths(
            reveal_model=args.reveal_model,
            requirement_report=args.requirement_report,
            requirement_calibration=spec["calibration"],
            requirement_strength=float(spec["strength"]),
            requirement_temperature=0.0,
            requirement_beta=0.25,
            counterfactual_conditioned_requirement=bool(
                args.counterfactual_conditioned_requirement
            ),
            lambda_cost=float(args.lambda_cost),
            tau_view=float(args.tau_view),
            online_update_enabled=bool(spec["online"]),
            online_update_weight=float(args.online_update_weight),
            online_requirement_scale=float(spec["requirement_scale"]),
            online_requirement_blend=float(spec["requirement_blend"]),
            online_requirement_prior_strength=float(
                args.online_requirement_prior_strength
            ),
            online_gain_deadband=float(args.online_gain_deadband),
            claim_belief_enabled=bool(spec.get("claim_belief", False)),
            belief_support_threshold=float(args.support_threshold),
            belief_contradiction_threshold=float(args.contradiction_threshold),
            belief_margin_threshold=float(args.margin_threshold),
            belief_role_threshold=float(args.role_threshold),
            belief_conflict_threshold=float(args.belief_conflict_threshold),
            selector_mode=(
                "claim_structured" if args.claim_structured else "legacy"
            ),
            typed_requirement_enabled=bool(
                args.typed_requirement and float(spec["strength"]) > 0.0
            ),
            unified_claim_evidence_ledger=bool(
                spec.get("unified_claim_evidence_ledger", False)
            ),
            pareto_evidence_checkpoint=bool(
                spec.get("pareto_evidence_checkpoint", False)
            ),
        )
        session_ray_field = getattr(
            getattr(policy.selector, "model", None),
            "session_ray_field",
            None,
        )
        if session_ray_field is None:
            raise RuntimeError("Reveal model does not expose a session ray field.")
        session_ray_field.strength = float(spec["destination_strength"])
        variant_rows: list[Dict[str, Any]] = []
        for trial_id, view_predictions in sorted(predictions.items()):
            truth = str(gt_by_trial[trial_id].get("claim_outcome", "")).strip().lower()
            oracle = max(utilities[trial_id].values(), default=0)
            for start_view in sorted(view_predictions):
                session_id = f"{variant}|{trial_id}|{start_view}"
                current_view = start_view
                visited = {start_view}
                path = [start_view]
                utility_path = [utilities[trial_id][start_view]]
                update_log: list[Dict[str, Any]] = []
                first_utility = utility_path[0]
                final_prediction = view_predictions[start_view]
                final_verification: VerificationResult | None = None
                for _move_index in range(max(0, int(args.max_moves))):
                    raw = raw_observations[trial_id][current_view]
                    prediction = view_predictions[current_view]
                    if (
                        str(prediction.get("decision", "insufficient"))
                        != "insufficient"
                    ):
                        break
                    observation_id = str(
                        raw.get("observation_id", f"{trial_id}_{current_view}")
                    )
                    before_observation, before_verification = runtime_inputs(
                        raw,
                        prediction,
                        session_id=session_id,
                        role_threshold=float(args.role_threshold),
                        geometry=geometry.get(observation_id),
                    )
                    decision = policy.select_view(
                        before_observation, before_verification
                    )
                    final_verification = policy.accumulate_verification(
                        before_observation, before_verification
                    )
                    selected_view = str(decision.selected_view)
                    if not selected_view or selected_view in visited:
                        break
                    visited.add(selected_view)
                    after_raw = raw_observations[trial_id][selected_view]
                    after_prediction = view_predictions[selected_view]
                    after_id = str(
                        after_raw.get("observation_id", f"{trial_id}_{selected_view}")
                    )
                    after_observation, after_verification = runtime_inputs(
                        after_raw,
                        after_prediction,
                        session_id=session_id,
                        role_threshold=float(args.role_threshold),
                        geometry=geometry.get(after_id),
                    )
                    update = policy.update_after_observation(
                        before_observation=before_observation,
                        before_verification=before_verification,
                        decision=decision,
                        after_observation=after_observation,
                        after_verification=after_verification,
                    )
                    update_log.append(update)
                    final_verification = policy.accumulate_verification(
                        after_observation, after_verification
                    )
                    policy.remember_observation_evidence(
                        after_observation, final_verification
                    )
                    current_view = selected_view
                    final_prediction = after_prediction
                    path.append(current_view)
                    utility_path.append(utilities[trial_id][current_view])
                    if len(utility_path) == 2:
                        first_utility = utility_path[-1]

                final_utility = utility_path[-1]
                claim_id = str(final_prediction.get("claim_id", ""))
                counterfactual_family = str(
                    final_prediction.get("counterfactual_family", "")
                )
                retained_view = (
                    policy.best_evidence_view(
                        session_id,
                        claim_id,
                        counterfactual_family,
                    )
                    or current_view
                )
                retained_utility = utilities[trial_id][retained_view]
                retained_prediction = view_predictions[retained_view]
                final_decision = str(
                    (
                        final_verification.metadata.get("decision")
                        if final_verification is not None
                        else final_prediction.get("decision")
                    )
                    or "insufficient"
                )
                belief_payload = (
                    dict(
                        final_verification.metadata.get(
                            "counterfactual_evidence_belief", {}
                        )
                    )
                    if final_verification is not None
                    else {}
                )
                variant_rows.append(
                    {
                        "trial_id": trial_id,
                        "start_view": start_view,
                        "truth": truth,
                        "path": path,
                        "utility_path": utility_path,
                        "current_utility": utility_path[0],
                        "utility_at_1": first_utility,
                        "final_utility": final_utility,
                        "gain_at_1": first_utility - utility_path[0],
                        "final_gain": final_utility - utility_path[0],
                        "retained_view": retained_view,
                        "retained_utility": retained_utility,
                        "retained_gain": retained_utility - utility_path[0],
                        "retained_resolve": int(retained_utility == 2),
                        "retained_regret": oracle - retained_utility,
                        "retained_decision": str(
                            retained_prediction.get("decision", "insufficient")
                        ),
                        "resolve_at_1": int(first_utility == 2),
                        "resolve_at_2": int(final_utility == 2),
                        "regret_at_2": oracle - final_utility,
                        "moves": len(path) - 1,
                        "online_updates": sum(
                            bool(row.get("applied", False)) for row in update_log
                        ),
                        "no_improvement": int(
                            final_utility <= utility_path[0] and utility_path[0] < 2
                        ),
                        "final_decision": final_decision,
                        "correct_resolution": int(final_decision == truth),
                        "false_support": int(
                            final_decision == "supported" and truth != "supported"
                        ),
                        "false_contradiction": int(
                            final_decision == "contradicted" and truth != "contradicted"
                        ),
                        "counterfactual_evidence_belief": belief_payload,
                        "update_log": update_log,
                    }
                )
        rows_by_variant[variant] = variant_rows
        results[variant] = summarize(variant_rows)

    payload = {
        "protocol": {
            "name": "causal two-move closed-loop replay",
            "cold_start_per_start_view": True,
            "max_moves": int(args.max_moves),
            "online_update_timing": "selected-view observation and frozen re-verification, then update",
            "retained_evidence_selection": (
                "highest claim-decidability score among observed views; "
                "robot utility is read only after the retained view is frozen"
            ),
            "selector_mode": (
                "claim_structured" if args.claim_structured else "legacy"
            ),
            "typed_requirement": bool(args.typed_requirement),
            "policy_spec_keys": list(POLICY_SPEC_KEYS),
            "policy_forbidden_gt_keys": [
                "claim_outcome",
                "error_type",
                "component_identity",
                "relation_state",
            ],
            "online_gain_deadband": float(args.online_gain_deadband),
            "online_requirement_fusion": {
                "scale": float(args.online_requirement_scale),
                "blend": float(args.online_requirement_blend),
                "prior_strength": float(args.online_requirement_prior_strength),
                "selection_source": "assistant-only causal prequential calibration",
            },
            "requirement_conditioning": (
                "claim_and_counterfactual_family"
                if args.counterfactual_conditioned_requirement
                else "claim_only"
            ),
            "online_update_variants": {
                "pi-only relative-online": (
                    "session-local relative reveal updates with uniform evidence-role mass"
                ),
                "pi cold-start + causal online R": (
                    "uniform cold-start role mass followed by causal session-local requirement updates"
                ),
                "R+pi relative-online": (
                    "session-local relative reveal and requirement updates only"
                ),
                "INSPECT online": (
                    "relative updates plus a session-local spherical destination-evidence field"
                ),
                "INSPECT adaptive online": (
                    "pi cold start followed by causal requirement and spherical destination updates"
                ),
            },
            "strict_session_isolation": True,
            "candidate_view_images_hidden": True,
            "robot_utility_used_by_policy": False,
            "utility_used_only_after_selection": True,
            "session_updates_shared_across_trials": False,
            "counterfactual_evidence_belief": {
                "enabled_variant": bool(args.include_claim_belief),
                "fusion": (
                    "conjunctive support/contradiction/ignorance with retained conflict"
                ),
                "conflict_threshold": float(args.belief_conflict_threshold),
                "selection_or_training_uses_robot_utility": False,
            },
            "setups": len(predictions),
            "start_views": sum(len(value) for value in predictions.values()),
            "input_fingerprints": {
                "observations": fingerprint(args.observations),
                "trial_gt": fingerprint(args.trial_gt),
                "evidence_scorer": fingerprint(args.evidence_scorer),
                "reveal_model": fingerprint(args.reveal_model),
                "requirement_report": fingerprint(args.requirement_report),
                "requirement_calibration": fingerprint(args.requirement_calibration),
                "current_geometry": fingerprint(args.current_geometry_jsonl),
                "evaluation_utility": fingerprint(args.evaluation_utility_csv),
            },
        },
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output_rows_json.write_text(
        json.dumps(rows_by_variant, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
