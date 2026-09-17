"""Audit frozen inspection decisions without fitting or changing a policy."""

from __future__ import annotations

import argparse
import os
import csv
import hashlib
import json
import sys
from bisect import bisect_right
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Frozen evaluation artifacts from the paper runs, released separately from the
# code. Point INSPECT_ARTIFACTS_ROOT at your local copy (default: artifacts/).
# Missing inputs raise FileNotFoundError with the exact expected path.
ARTIFACTS = Path(os.environ.get("INSPECT_ARTIFACTS_ROOT", "artifacts"))

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fingerprint(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_report(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def category(prediction, truth):
    if prediction == "insufficient":
        return "insufficient"
    if prediction == truth:
        return "correct_resolution"
    return "incorrect_" + prediction


def scorer_observation(observation):
    metadata = observation["metadata"]
    return {"metadata": {
        key: metadata[key] for key in
        ("detections", "role_detections", "frame_shape", "scene_evidence")
        if key in metadata
    }}


def state_at(annotation, frame, component):
    sequence = annotation["state_sequence"]
    index = bisect_right([int(row["frame"]) for row in sequence], int(frame)) - 1
    if index < 0:
        return int(annotation["initial_state_vector"][component])
    return int(sequence[index]["state"][component])


def summarize_robot(rows):
    return {
        "trials": len(rows),
        "raw_scorer": dict(Counter(category(r["raw_decision"], r["truth"]) for r in rows)),
        "discovery_gate": dict(Counter(category(r["discovery_decision"], r["truth"]) for r in rows)),
        "commit_gate": dict(Counter(category(r["final_decision"], r["truth"]) for r in rows)),
        "first_abstention_stage": dict(Counter(r["first_abstention_stage"] for r in rows)),
        "overlapping_evidence_failures": {
            name: sum(not r[name] for r in rows)
            for name in ("identity_available", "relation_available",
                         "support_evidence_available", "contradiction_evidence_available")
        },
        "target_identity_evidence_absent": sum(r["target_conf"] == 0 for r in rows),
        "housing_detection_absent": sum(r["housing_role_conf"] == 0 for r in rows),
        "correct_raw_decisions_blocked": sum(
            r["raw_decision"] == r["truth"] and r["final_decision"] == "insufficient"
            for r in rows
        ),
        "incorrect_raw_decisions_blocked": sum(
            r["raw_decision"] not in (r["truth"], "insufficient")
            and r["final_decision"] == "insufficient" for r in rows
        ),
    }


def audit_robot(root, output):
    import evaluate_robot_closed_loop_gated as gated
    from inspect_system.evidence_scorer import PrototypeEvidenceScorer

    source = ARTIFACTS / "robot"
    report_path = source / "strict_final_policy_component_ablation_132.json"
    rows_path = source / "strict_final_policy_component_ablation_rows_132.json"
    protocol = read_json(report_path)["protocol"]
    sources = protocol["input_fingerprints"]
    for key in ("observations", "evidence_scorer", "trial_gt"):
        assert fingerprint(sources[key]["path"])["sha256"] == sources[key]["sha256"], key
    scorer = PrototypeEvidenceScorer.load(Path(sources["evidence_scorer"]["path"]))
    with Path(sources["trial_gt"]["path"]).open(encoding="utf-8-sig", newline="") as stream:
        truth_rows = {row["trial_id"]: row for row in csv.DictReader(stream)}
    observations = {
        (o["metadata"]["trial_id"], o["view_id"]): o
        for o in read_jsonl(sources["observations"]["path"])
    }
    frozen_rows = read_json(rows_path)["INSPECT"]
    settings = protocol["commit_gate"]
    predictions = {}
    for trial in frozen_rows:
        for view in (trial["current_view"], trial["selected_view"]):
            key = (trial["trial_id"], view)
            if key in predictions:
                continue
            gt = truth_rows[key[0]]
            policy_spec = {name: gt[name] for name in protocol["policy_spec_keys"]}
            observation = observations[key]
            # Only the explicit policy-spec whitelist enters the claim scorer.
            _, raw = gated._ORIGINAL(
                scorer_observation(observation), gt=policy_spec, scorer=scorer, support_threshold=0.35,
                contradiction_threshold=0.45, margin_threshold=0.05, role_threshold=0.65,
            )
            discovery = gated.apply_evidence_availability_gate(raw)
            final = gated.apply_evidence_availability_gate(
                discovery, identity_confidence_threshold=settings["identity_confidence"],
                identity_margin_floor=settings["identity_margin_floor"],
                counterfactual_identity_threshold=settings["counterfactual_identity_threshold"],
                counterfactual_margin_floor=settings["counterfactual_margin_floor"],
                relation_evidence_threshold=settings["relation_evidence"],
                allow_relation_absence_contradiction=not settings["explicit_contradiction_only"],
            )
            stage = (
                "scorer" if raw["decision"] == "insufficient" else
                "discovery_gate" if discovery["decision"] == "insufficient" else
                "commit_gate" if final["decision"] == "insufficient" else "committed"
            )
            metadata = observation["metadata"]
            predictions[key] = {
                "truth": gt["claim_outcome"].strip().lower(),
                "raw_decision": raw["decision"], "discovery_decision": discovery["decision"],
                "final_decision": final["decision"], "first_abstention_stage": stage,
                "support_score": raw["support_score"],
                "contradiction_score": raw["contradiction_score"],
                "role_scores": raw["role_scores"],
                "detection_count": len(metadata.get("detections", [])),
                "raw_detection_count": len(metadata.get("raw_detections", [])),
                "minimum_detection_confidence": min(
                    (float(d["confidence"]) for d in metadata.get("detections", [])),
                    default=None,
                ),
                **{k: final[k] for k in (
                    "identity_available", "relation_available", "support_evidence_available",
                    "contradiction_evidence_available", "wrong_identity_visible",
                    "incomplete_relation_visible",
                )},
                **{k: float(raw["features"].get(k, 0)) for k in (
                    "target_conf", "housing_role_conf", "wrong_same_role_conf", "identity_margin",
                )},
            }
    rows = []
    for trial in frozen_rows:
        pred = predictions[(trial["trial_id"], trial["selected_view"])]
        assert pred["final_decision"] == trial["selected_decision"], trial
        assert int(pred["final_decision"] == pred["truth"]) == trial["correct_resolution"]
        rows.append({
            **{k: trial[k] for k in (
                "trial_id", "current_view", "selected_view", "moved", "current_utility",
                "selected_utility", "correct_resolution",
            )}, **pred,
        })
    summary = {
        "protocol": "Frozen predicted-evidence replay; no neural inference, fitting, or threshold selection.",
        "evidence_failure_counts_overlap": True,
        "evidence_availability_is_not_gt_detection_accuracy": True,
        "all_decisions_match_frozen_rows": True,
        "annotation_metadata_removed_before_scoring": True,
        "audit_script": fingerprint(__file__),
        "sources": [fingerprint(report_path), fingerprint(rows_path), *sources.values()],
        "all_selected": summarize_robot(rows),
        "moved": summarize_robot([r for r in rows if r["moved"]]),
        "moved_by_truth": {
            truth: summarize_robot([r for r in rows if r["moved"] and r["truth"] == truth])
            for truth in sorted({r["truth"] for r in rows})
        },
        "selected_utility_2": summarize_robot([r for r in rows if r["selected_utility"] == 2]),
        "full_verifiability_gained": sum(r["current_utility"] < 2 and r["selected_utility"] == 2 for r in rows),
        "full_verifiability_lost": sum(r["current_utility"] == 2 and r["selected_utility"] < 2 for r in rows),
        "all_used_views_minimum_detection_confidence": min(
            r["minimum_detection_confidence"] for r in predictions.values()
            if r["minimum_detection_confidence"] is not None
        ),
    }
    write_report(output / "robot_summary.json", summary)
    write_report(output / "robot_trial_diagnostics.json", rows)
    return summary


def audit_impact(root, output):
    source = ARTIFACTS / "impact/multiview_claim_scores.jsonl"
    policy_path = ARTIFACTS / "impact/zero_shot_rerun_rows.json"
    score_rows = read_jsonl(source)
    scores = {}
    annotations = {}
    alignment = []
    for row in score_rows:
        key = (row["recording_id"], int(row["source_frame"]), int(row["component_id"]), row["view"])
        assert key not in scores, key
        scores[key] = row
        ann_path = Path(row["source_annotation"])
        if str(ann_path) not in annotations:
            annotations[str(ann_path)] = read_json(ann_path)
        front = annotations[str(ann_path)]
        local_path = ann_path.with_name(ann_path.name.replace("_front_asr", "_" + row["view"] + "_asr"))
        if not local_path.exists():
            alignment.append({"key": list(key), "status": "missing_view_annotation"})
            continue
        if str(local_path) not in annotations:
            annotations[str(local_path)] = read_json(local_path)
        local = annotations[str(local_path)]
        reference, query = int(row["reference_query_frame"]), int(row["query_frame"])
        offset_front = int(front.get("view_start", 0))
        offset_local = int(local.get("view_start", 0))
        front_time = (reference - offset_front) / float(front["fps"])
        local_time = (query - offset_local) / float(local["fps"])
        expected = int(row["world_value"])
        alignment.append({
            "key": list(key), "status": "checked", "reference_frame": reference,
            "query_frame": query, "time_difference_seconds": local_time - front_time,
            "source_truth_matches_reference_frame": state_at(front, reference, key[2]) == expected,
            "source_truth_matches_query_frame": state_at(local, query, key[2]) == expected,
        })
    rows = []
    uniform_total = 0.0
    uniform_new = 0.0
    for trial in read_json(policy_path)["INSPECT"]:
        base = (trial["recording_id"], int(trial["source_frame"]), int(trial["component_id"]))
        current = scores[base + (trial["current_view"],)]
        selected = scores[base + (trial["selected_view"],)]
        assert current["utility"] == trial["current_utility"]
        assert selected["utility"] == trial["selected_utility"]
        truth = selected["semantic_outcome"]
        correct = int(selected["claim_prediction"] == truth)
        assert correct == trial["resolve_at_1"]
        alternatives = [
            scores[base + (v,)] for v in ("front", "left", "right", "top")
            if v != trial["current_view"]
        ]
        uniform = (
            sum(v["claim_prediction"] == v["semantic_outcome"] for v in alternatives) / 3
            if trial["moved"] else int(current["claim_prediction"] == truth)
        )
        uniform_total += uniform
        uniform_new += uniform if trial["moved"] else 0
        rows.append({
            **trial, "truth": truth, "current_prediction": current["claim_prediction"],
            "selected_prediction": selected["claim_prediction"],
            "result_category": category(selected["claim_prediction"], truth),
            "uniform_same_move_mask_expected_correct": uniform,
        })
    checked = [r for r in alignment if r["status"] == "checked"]
    static = [r for r in checked if r["key"][-1] != "ego"]
    # ASR labels are front-view temporal world states. The four static caches
    # use identical frame indices; this does not independently certify capture sync.
    static_scores = [r for r in score_rows if r["view"] != "ego"]
    same_static_index = all(
        int(r["query_frame"]) == int(r["reference_query_frame"]) for r in static_scores
    )
    corrected_truth = {}
    for key, row in scores.items():
        if row["view"] == "ego":
            continue
        assert row["claim_id"].endswith(".installed_correctly"), row["claim_id"]
        annotation = annotations[str(Path(row["source_annotation"]))]
        value = state_at(annotation, int(row["reference_query_frame"]), key[2])
        corrected_truth[key[:3]] = "supported" if value == 1 else "contradicted"
    corrected = {}
    for policy, trials in read_json(policy_path).items():
        counts = Counter()
        moved_counts = Counter()
        for trial in trials:
            base = (trial["recording_id"], int(trial["source_frame"]), int(trial["component_id"]))
            selected = scores[base + (trial["selected_view"],)]
            result = category(selected["claim_prediction"], corrected_truth[base])
            counts[result] += 1
            if trial["moved"]:
                moved_counts[result] += 1
        corrected[policy] = {"trials": len(trials), "all_outcomes": dict(counts),
                             "moved_outcomes": dict(moved_counts)}
    corrected_uniform = sum(
        (
            sum(scores[(r["recording_id"], int(r["source_frame"]), int(r["component_id"]), v)]["claim_prediction"]
                == corrected_truth[(r["recording_id"], int(r["source_frame"]), int(r["component_id"]))]
                for v in ("front", "left", "right", "top") if v != r["current_view"]) / 3
            if r["moved"] else int(r["current_prediction"] == corrected_truth[
                (r["recording_id"], int(r["source_frame"]), int(r["component_id"]))
            ])
        ) for r in rows
    )
    summary = {
        "protocol": "Frozen head predictions and frozen move mask; candidate labels are evaluated only after action selection.",
        "sources": [fingerprint(source), fingerprint(policy_path)],
        "annotation_sources": [fingerprint(p) for p in sorted(annotations)],
        "audit_script": fingerprint(__file__),
        "trials": len(rows), "moved": sum(r["moved"] for r in rows),
        "moved_outcomes": dict(Counter(r["result_category"] for r in rows if r["moved"])),
        "all_outcomes": dict(Counter(r["result_category"] for r in rows)),
        "same_move_mask_uniform": {
            "expected_correct_total": uniform_total,
            "expected_correct_on_moved": uniform_new,
            "selection": "Exact uniform expectation over three non-current static views on INSPECT's fixed move-start subset.",
        },
        "alignment": {
            "all_static_query_indices_match_reference": same_static_index,
            "all_static_score_rows": len(static_scores),
            "checked_rows": len(checked), "static_rows": len(static),
            "missing_annotations": len(alignment) - len(checked),
            "static_source_truth_mismatch_at_reference": sum(not r["source_truth_matches_reference_frame"] for r in static),
            "static_source_truth_mismatch_at_query": sum(not r["source_truth_matches_query_frame"] for r in static),
            "static_max_abs_time_difference_seconds": max((abs(r["time_difference_seconds"]) for r in static), default=None),
            "timing_basis": "Per-view official annotation FPS and view_start; check capture synchronization separately.",
        },
        "query_time_truth_diagnostic": {
            "source_event_truth_changed_at_query_events": sum(
                scores[k + ("front",)]["semantic_outcome"] != truth
                for k, truth in corrected_truth.items()
            ),
            "fixed_policy_choices_no_refitting": corrected,
            "uniform_same_move_mask_expected_correct_total": corrected_uniform,
            "scope": "Front ASR world truth at reference_query_frame, shared across synchronized static views; not per-view visibility GT.",
        },
    }
    write_report(output / "impact_summary.json", summary)
    write_report(output / "impact_trial_diagnostics.json", rows)
    write_report(output / "impact_alignment.json", alignment)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/inspection_audit")
    args = parser.parse_args()
    robot = audit_robot(args.root, args.output)
    print(json.dumps({"robot": robot}, indent=2))
    impact = audit_impact(args.root, args.output)
    print(json.dumps({"impact": impact}, indent=2))


if __name__ == "__main__":
    main()
