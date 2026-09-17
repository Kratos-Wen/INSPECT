"""Calibrate score thresholds on disjoint assistant videos, never robot labels."""

import argparse
import os
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Frozen evaluation artifacts from the paper runs, released separately from the
# code. Point INSPECT_ARTIFACTS_ROOT at your local copy (default: artifacts/).
# Missing inputs raise FileNotFoundError with the exact expected path.
ARTIFACTS = Path(os.environ.get("INSPECT_ARTIFACTS_ROOT", "artifacts"))

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from audit_inspection_evidence_chain import fingerprint, read_json, write_report
import evaluate_robot_closed_loop_gated as gated
from inspect_system.evidence_scorer import PrototypeEvidenceScorer
from train_assistance_evidence_scorer import group_timeline, iter_jsonl, read_csv, timeline_at, video_key


def split_videos(samples):
    groups = defaultdict(set)
    for sample in samples:
        groups[sample["claim_id"], sample["label"]].add(sample["video"])
    validation = set()
    mandatory_train = set().union(*(videos for videos in groups.values() if len(videos) < 2))
    for key, videos in sorted(groups.items()):
        ordered = sorted(videos, key=lambda s: hashlib.sha256(video_key(s).encode()).hexdigest())
        if len(ordered) < 2:
            continue
        validation.update(ordered[:max(1, len(ordered) // 5)])
    validation -= mandatory_train
    training = {s["video"] for s in samples} - validation
    for key, videos in groups.items():
        if not videos & training:
            raise ValueError(f"No training video for {key}")
    return training, validation


def metrics(rows, support, contradiction, gate):
    counts = defaultdict(int)
    for row in rows:
        prediction = deepcopy(row["prediction"])
        pos, neg = prediction["support_score"], prediction["contradiction_score"]
        prediction["decision"] = (
            "contradicted" if neg >= contradiction and neg >= pos + .05 else
            "supported" if pos >= support and pos >= neg + .05 else "insufficient")
        first = gated.apply_evidence_availability_gate(prediction)
        final = gated.apply_evidence_availability_gate(
            first, identity_confidence_threshold=gate["identity_confidence"],
            identity_margin_floor=gate["identity_margin_floor"],
            counterfactual_identity_threshold=gate["counterfactual_identity_threshold"],
            counterfactual_margin_floor=gate["counterfactual_margin_floor"],
            relation_evidence_threshold=gate["relation_evidence"],
            allow_relation_absence_contradiction=not gate["explicit_contradiction_only"])
        decision, truth = final["decision"], row["truth"]
        counts["n"] += 1
        counts["correct_commitments"] += int(decision == truth and decision != "insufficient")
        counts["wrong_commitments"] += int(decision != truth and decision != "insufficient")
        counts["false_support"] += int(decision == "supported" and truth != "supported")
        counts["abstentions"] += int(decision == "insufficient")
        counts["triage_correct"] += int(decision == truth)
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-dir", type=Path, default=ROOT / "outputs/inspection_protocol_repair")
    args = parser.parse_args()
    output = args.repair_dir / "source_calibration"
    samples_path = args.repair_dir / "robot/causal_feedback_samples.jsonl"
    samples = list(iter_jsonl(samples_path))
    train_videos, val_videos = split_videos(samples)
    training = [s for s in samples if s["video"] in train_videos]
    model = PrototypeEvidenceScorer.train(training)
    model.metadata.update(training_source="assistant feedback replay",
                          uses_robot_view_training=False,
                          training_videos=sorted(train_videos),
                          validation_videos=sorted(val_videos),
                          feature_contract="explicit fused detections; causal within-interval feedback")
    model_path = output / "scorer.json"
    model.save(model_path)
    manifest = {"train": sorted(train_videos), "validation": sorted(val_videos),
                "test": "unchanged 22 robot setups / 132 start views",
                "split_unit": "source video; stratified by claim and outcome",
                "split_selection": "stable SHA256 ordering; singleton claim/label groups remain in training",
                "training_sample_count": len(training)}
    write_report(output / "split.json", manifest)
    old = read_json(args.repair_dir / "robot/causal_feedback_scorer.json")["metadata"]
    timeline = group_timeline(read_csv(Path(old["timeline_csv"])))
    rows = []
    unknown = 0
    for entry in read_csv(Path(old["online_summary_csv"])):
        if entry["video"] not in val_videos:
            continue
        intervals = timeline.get(video_key(entry["video"]), [])
        for record in iter_jsonl(Path(entry["run_dir"]) / "iterations.jsonl"):
            frame = int(record.get("frame_index", -1))
            annotation = timeline_at(intervals, frame)
            if not annotation:
                continue
            if annotation["assembly_set"] not in ("A", "B"):
                unknown += 1
                continue
            truth = annotation["outcome"]
            truth = "insufficient" if truth == "unresolved" else truth
            if truth not in ("supported", "contradicted", "insufficient"):
                raise ValueError(truth)
            detection_key = "fused_detections" if "fused_detections" in record else "raw_detections"
            obs = {"metadata": {
                "detections": record.get(detection_key) or [],
                "role_detections": record.get("role_detections") or [],
                "frame_shape": record.get("frame_shape") or [1080, 1920],
                "scene_evidence": record.get("scene_evidence") or {},
            }}
            spec = {"target_step": annotation["step_id"], "claim_id": annotation["claim_id"],
                    "product_variant": annotation["assembly_set"]}
            _, prediction = gated._ORIGINAL(obs, gt=spec, scorer=model, support_threshold=.35,
                                            contradiction_threshold=.45, margin_threshold=.05, role_threshold=.65)
            rows.append({"video": entry["video"], "frame": frame, "truth": truth,
                         "claim_id": annotation["claim_id"], "prediction": prediction})
    if not rows:
        raise ValueError("No validation observations")
    reference = read_json(ARTIFACTS / "robot/strict_final_policy_component_ablation_132.json")
    gate = reference["protocol"]["commit_gate"]
    baseline = metrics(rows, .35, .45, gate)
    candidates = []
    for support in (.35, .50, .65, .80, .90):
        for contradiction in (.45, .60, .75, .90):
            result = metrics(rows, support, contradiction, gate)
            if result["false_support"] <= baseline["false_support"] and result["wrong_commitments"] <= baseline["wrong_commitments"]:
                candidates.append({"support": support, "contradiction": contradiction, "metrics": result})
    selected = max(candidates, key=lambda c: (
        c["metrics"]["correct_commitments"], c["metrics"]["triage_correct"],
        -c["metrics"]["wrong_commitments"], -abs(c["support"]-.35)-abs(c["contradiction"]-.45)))
    report = {
        "protocol": manifest, "baseline": baseline, "selected": selected,
        "selection": "maximize correct commitments without increasing validation false support or wrong commitments; tie-break by triage accuracy and proximity to frozen thresholds",
        "fixed_identity_relation_gate": gate, "validation_observations": len(rows),
        "unknown_family_excluded": unknown,
        "validation_claim_counts": {c: sum(r["claim_id"] == c for r in rows)
                                   for c in sorted({r["claim_id"] for r in rows})},
        "truth_counts": {s: sum(r["truth"] == s for r in rows) for s in ("supported", "contradicted", "insufficient")},
        "eligible_candidates": candidates,
        "sources": [fingerprint(samples_path), fingerprint(old["timeline_csv"]), fingerprint(old["online_summary_csv"])],
        "code": fingerprint(__file__),
    }
    # Persist the unique operating point before the robot evaluator is started.
    write_report(output / "calibration.json", report)
    command = read_json(args.repair_dir / "robot/command.json")
    replacements = {"--evidence-scorer": str(model_path),
                    "--support-threshold": str(selected["support"]),
                    "--contradiction-threshold": str(selected["contradiction"]),
                    "--output-json": str(output / "robot_summary.json"),
                    "--output-rows-json": str(output / "robot_rows.json")}
    for flag, value in replacements.items():
        command[command.index(flag) + 1] = value
    write_report(output / "command.json", command)
    print(json.dumps({"validation": report}, indent=2), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
