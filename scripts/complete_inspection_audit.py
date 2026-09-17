"""Propagate query-time truth and test a causal feedback extraction repair.

Old results and checkpoints are never overwritten. Robot thresholds, view
models, and evaluation sets come from the frozen reference, not a new search.
"""

import argparse
import os
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Frozen evaluation artifacts from the paper runs, released separately from the
# code. Point INSPECT_ARTIFACTS_ROOT at your local copy (default: artifacts/).
# Missing inputs raise FileNotFoundError with the exact expected path.
ARTIFACTS = Path(os.environ.get("INSPECT_ARTIFACTS_ROOT", "artifacts"))

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from audit_inspection_evidence_chain import fingerprint, read_json, read_jsonl, write_report
from external_benchmarks.impact_query_truth import relabel_query
from inspect_system.evidence_scorer import PrototypeEvidenceScorer
from train_assistance_evidence_scorer import build_samples, write_jsonl

VIEWS = ("front", "left", "right", "top")


def key(row):
    return row["recording_id"], int(row["source_frame"]), int(row["component_id"])


def rescore_trial(trial, scores):
    event = key(trial)
    current = scores[event + (trial["current_view"],)]
    selected = scores[event + (trial["selected_view"],)]
    oracle = max(scores[event + (view,)]["utility"] for view in VIEWS)
    truth = selected["semantic_outcome"]
    sign = 1 if truth == "supported" else -1
    current_margin = sign * current["support_margin"]
    selected_margin = sign * selected["support_margin"]
    return {
        **trial, "current_utility": current["utility"], "selected_utility": selected["utility"],
        "gain": selected["utility"] - current["utility"],
        "resolve_at_1": int(selected["claim_prediction"] == truth),
        "regret": oracle - selected["utility"],
        "current_counterfactual_margin": current_margin,
        "selected_counterfactual_margin": selected_margin,
        "counterfactual_separation_gain": selected_margin - current_margin,
        "truth": truth, "selected_prediction": selected["claim_prediction"],
    }


def summarize(rows):
    result = {name: mean(r[name] for r in rows) for name in (
        "selected_utility", "gain", "resolve_at_1", "regret", "moved",
        "selected_counterfactual_margin", "counterfactual_separation_gain")}
    result.update(trials=len(rows), correct=sum(r["resolve_at_1"] for r in rows),
                  incorrect=sum(r["selected_prediction"] not in (r["truth"], "insufficient") for r in rows),
                  insufficient=sum(r["selected_prediction"] == "insufficient" for r in rows))
    return result


def correct_impact(output):
    folder = ARTIFACTS / "external_comparison"
    source = ARTIFACTS / "impact/multiview_claim_scores.jsonl"
    rows_path = folder / "impact_zero_shot_rerun_rows.json"
    raw = read_jsonl(source)
    annotations = {r["source_annotation"]: read_json(r["source_annotation"]) for r in raw}
    corrected = [relabel_query(r, annotations[r["source_annotation"]]) for r in raw]
    scores = {}
    for row in corrected:
        index = key(row) + (row["view"],)
        if index in scores:
            raise ValueError(f"Duplicate view: {index}")
        if row["view"] in VIEWS and row["query_frame"] != row["reference_query_frame"]:
            raise ValueError(f"Static frame alignment mismatch: {index}")
        scores[index] = row
    policies = read_json(rows_path)
    expected = {key(r) + (r["current_view"],) for r in policies["INSPECT"]}
    if len(expected) != 1068:
        raise ValueError("Unexpected trial set")
    results = {}
    for name, trials in policies.items():
        if len(trials) != len(expected) or {key(r) + (r["current_view"],) for r in trials} != expected:
            raise ValueError(f"Unmatched trials: {name}")
        results[name] = [rescore_trial(r, scores) for r in trials]
    matched_uniform = []
    for row in policies["INSPECT"]:
        candidates = [v for v in VIEWS if v != row["current_view"]] if row["moved"] else [row["current_view"]]
        matched_uniform.append(mean(
            rescore_trial({**row, "selected_view": v}, scores)["resolve_at_1"] for v in candidates))
    changes = sorted({key(r) for r in corrected if r["semantic_outcome"] != r["source_event_semantic_outcome"]})
    report = {
        "protocol": {
            "frozen_actions_and_predictions": True, "fitting_or_threshold_selection": False,
            "truth": "Front ASR world state at reference_query_frame; not visibility annotation",
            "static_alignment": "Equal cached query indices; capture synchronization not independently verified",
            "changed_events": changes, "all_policies_share_identical_trial_keys": True,
            "source_receipts": [fingerprint(source), fingerprint(rows_path),
                                *[fingerprint(p) for p in sorted(annotations)]],
            "script_receipts": [fingerprint(__file__), fingerprint(ROOT / "scripts/external_benchmarks/impact_query_truth.py")],
        },
        "results": {name: summarize(rows) for name, rows in results.items()},
        "same_move_mask_uniform_diagnostic": {
            "expected_correct": sum(matched_uniform), "resolve_at_1": mean(matched_uniform),
            "trials": len(matched_uniform), "sampled_run": False,
        },
    }
    write_jsonl(output / "impact/multiview_claim_scores.jsonl", corrected)
    write_report(output / "impact/policy_rows.json", results)
    write_report(output / "impact/summary.json", report)
    structured_path = ARTIFACTS / "impact/claim_structured_zero_shot_rows.json"
    structured = {}
    for name, trials in read_json(structured_path).items():
        if len(trials) != len(expected) or {key(r) + (r["current_view"],) for r in trials} != expected:
            raise ValueError(f"Unmatched claim-structured trials: {name}")
        structured[name] = [rescore_trial(r, scores) for r in trials]
    structured_report = {
        "protocol": {**report["protocol"], "policy_rows_source": fingerprint(structured_path)},
        "results": {name: summarize(rows) for name, rows in structured.items()},
    }
    write_report(output / "impact/claim_structured_policy_rows.json", structured)
    write_report(output / "impact/claim_structured_summary.json", structured_report)
    verify_impact_actions(raw, corrected, policies, output)
    print(json.dumps({"IMPACT": report["results"]}, indent=2), flush=True)
    print(json.dumps({"IMPACT_claim_structured": structured_report["results"]}, indent=2), flush=True)


def verify_impact_actions(original, corrected, frozen, output):
    from collections import defaultdict
    from external_benchmarks.evaluate_impact_reveal_policy import policy_rows
    from inspect_system.active_view.requirement_model import RequirementCalibration, load_requirement_counts
    from inspect_system.active_view.reveal_model import PriorTableRevealModel

    config = ARTIFACTS / "robot_policy"
    model_path = config / "adopted_active_selector_object_centric_online_v2.json"
    requirement_path = config / "asymmetric_requirement_reveal_report.json"
    calibration_path = config / "expanded_joint_action_calibration_v2.json"
    model = PriorTableRevealModel.load(model_path)
    arguments = dict(cutoff=0.33679987490177155, lambda_cost=0.05, tau_view=0.02,
                     semantics="metadata", requirement_counts=load_requirement_counts(requirement_path),
                     requirement_calibration=RequirementCalibration.load(calibration_path))
    action_maps = []
    for rows in (original, corrected):
        grouped = defaultdict(dict)
        for row in rows:
            if row["view"] in VIEWS:
                grouped[key(row)][row["view"]] = row
        replay = policy_rows(grouped, model, **arguments)
        action_maps.append({key(r) + (r["current_view"],): r["selected_view"] for r in replay})
    expected = {key(r) + (r["current_view"],): r["selected_view"] for r in frozen["INSPECT"]}
    if action_maps[0] != expected or action_maps[1] != expected:
        raise ValueError("Replayed actions differ from the frozen relative-policy choices")
    write_report(output / "impact/action_invariance.json", {
        "trials": len(expected), "all_actions_match_frozen": True,
        "query_truth_does_not_change_actions": True,
        "input_receipts": [fingerprint(p) for p in (model_path, requirement_path, calibration_path)],
        "code_receipt": fingerprint(ROOT / "scripts/external_benchmarks/evaluate_impact_reveal_policy.py"),
    })


def repair_scorer(output):
    reference_path = ARTIFACTS / "robot/strict_final_policy_component_ablation_132.json"
    reference = read_json(reference_path)
    sources = reference["protocol"]["input_fingerprints"]
    for value in sources.values():
        if value and fingerprint(value["path"])["sha256"] != value["sha256"]:
            raise ValueError(f"Changed frozen input: {value['path']}")
    old = PrototypeEvidenceScorer.load(Path(sources["evidence_scorer"]["path"]))
    metadata = old.metadata
    samples = build_samples(summary_csv=Path(metadata["online_summary_csv"]),
                            timeline_csv=Path(metadata["timeline_csv"]),
                            window=int(metadata["window"]), min_conf=float(metadata["min_conf"]))
    model = PrototypeEvidenceScorer.train(samples)
    model.metadata.update(metadata)
    model.metadata.update(num_samples=len(samples), num_usable_samples=len(samples),
                          feature_contract="explicit fused detections; causal within-interval feedback",
                          threshold_selection=False, previous_checkpoint_sha256=sources["evidence_scorer"]["sha256"])
    checkpoint = output / "robot/causal_feedback_scorer.json"
    model.save(checkpoint)
    write_jsonl(output / "robot/causal_feedback_samples.jsonl", samples)
    receipt = {
        "training": "Assistant feedback replay only; no new threshold or hyperparameter selection",
        "robot": "Frozen predicted detector/MoGe/scene graph; view actions recomputed",
        "old_samples": old.metadata["num_samples"], "new_samples": len(samples),
        "by_claim_label": dict(Counter(s["claim_id"] + ":" + s["label"] for s in samples)),
        "source_receipts": [fingerprint(reference_path), fingerprint(metadata["timeline_csv"]),
                            fingerprint(metadata["online_summary_csv"])],
        "training_log_receipts": [
            fingerprint(Path(directory) / name)
            for directory in sorted({s["run_dir"] for s in samples})
            for name in ("iterations.jsonl", "feedback.jsonl")
        ],
        "code_receipts": [fingerprint(ROOT / "scripts/train_assistance_evidence_scorer.py"),
                          fingerprint(ROOT / "inspect_system/evidence_scorer.py"), fingerprint(__file__)],
        "checkpoint": fingerprint(checkpoint),
    }
    write_report(output / "robot/training_receipt.json", receipt)
    cmd = [sys.executable, str(ROOT / "scripts/evaluate_decidability_gated_transport.py")]
    for flag, source in (
        ("observations", "observations"), ("trial-gt", "trial_gt"),
        ("reveal-model", "reveal_model"), ("requirement-report", "requirement_report"),
        ("requirement-calibration", "requirement_calibration"), ("current-geometry", "current_geometry"),
        ("evaluation-utility-csv", "evaluation_utility"),
    ):
        cmd += ["--" + flag, sources[source]["path"]]
    cmd += ["--evidence-scorer", str(checkpoint), "--only-variant", "INSPECT",
            "--output-json", str(output / "robot/summary.json"),
            "--output-rows-json", str(output / "robot/policy_rows.json")]
    fixed = {"support-threshold": .35, "contradiction-threshold": .45,
             "margin-threshold": .05, "role-threshold": .65, "partial-threshold": .35,
             "lambda-cost": reference["protocol"]["lambda_cost"], "tau-view": reference["protocol"]["tau_view"]}
    gate = reference["protocol"]["commit_gate"]
    for flag, field in (
        ("commit-identity-confidence", "identity_confidence"),
        ("commit-identity-margin", "identity_margin_floor"),
        ("commit-alternative-identity-confidence", "counterfactual_identity_threshold"),
        ("commit-counterfactual-margin", "counterfactual_margin_floor"),
        ("commit-relation-threshold", "relation_evidence"),
    ):
        if gate[field] is not None:
            fixed[flag] = gate[field]
    for flag, value in fixed.items():
        cmd += ["--" + flag, str(value)]
    if gate["explicit_contradiction_only"]:
        cmd.append("--commit-explicit-contradiction-only")
    write_report(output / "robot/command.json", cmd)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/inspection_protocol_repair")
    parser.add_argument("--stage", choices=("all", "impact", "robot"), default="all")
    args = parser.parse_args()
    if args.stage in ("all", "impact"):
        correct_impact(args.output)
    if args.stage in ("all", "robot"):
        repair_scorer(args.output)


if __name__ == "__main__":
    main()
