"""Audit detector checkpoint and threshold consistency across INSPECT evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("observation_id", "")).strip()
            if not key:
                raise ValueError(f"Missing observation_id at line {line_number}")
            if key in rows:
                raise ValueError(f"Duplicate observation_id: {key}")
            rows[key] = row
    return rows


def _canonical_path(path: str | Path) -> str:
    return Path(path).resolve().as_posix().casefold()


def _detection_signature(detections: Iterable[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return sorted(
        (
            str(item["name"]),
            round(float(item["confidence"]), 8),
            tuple(round(float(value), 4) for value in item["xyxy"]),
        )
        for item in detections
    )


def _raw_detections(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(row.get("metadata", {}).get("raw_detections", []) or [])


def _correct_f1(result: dict[str, Any]) -> float:
    precision = float(result["precision_correct_class"])
    recall = float(result["correct_class_recall"])
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def build_audit(
    *,
    repository_root: Path,
    detector_summary_path: Path,
    assistant_replay_summary_path: Path,
    robot_proposal_observations_path: Path,
    robot_commit_observations_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    checkpoint = checkpoint_path.resolve()
    checkpoint_digest = sha256(checkpoint)

    detector_rows = _load_json(detector_summary_path)
    if not isinstance(detector_rows, list):
        raise ValueError("Detector summary must contain a list of model results")
    matches = [row for row in detector_rows if row.get("model") == "YOLO26s + PI-TwinSwap"]
    if len(matches) != 1:
        raise ValueError("Expected exactly one PI-TwinSwap detector row")
    detector = matches[0]
    proposal_threshold = float(detector["conf"])
    if proposal_threshold != 0.1:
        raise ValueError(f"Unexpected detector operating point: {proposal_threshold}")
    if float(detector["iou_match"]) != 0.5:
        raise ValueError("Detector table does not use IoU=0.50")
    if _canonical_path(detector["weights"]) != _canonical_path(checkpoint):
        raise ValueError("Detector table checkpoint differs from the adopted checkpoint")
    baseline_names = ("YOLO26s Plain", "YOLO26s + Synthetic-Aug")
    baseline_rows = {
        name: next((row for row in detector_rows if row.get("model") == name), None)
        for name in baseline_names
    }
    if any(row is None for row in baseline_rows.values()):
        raise ValueError("Detector summary is missing a required matched baseline")
    detector_f1 = _correct_f1(detector)
    baseline_f1 = {
        name: _correct_f1(row)
        for name, row in baseline_rows.items()
        if row is not None
    }
    if detector_f1 <= max(baseline_f1.values()):
        raise ValueError("PI-TwinSwap does not improve correct-evidence F1")

    replay_rows = _load_json(assistant_replay_summary_path)
    if not isinstance(replay_rows, list) or not replay_rows:
        raise ValueError("Assistant replay summary must contain session rows")
    assistant_configs = []
    for replay in replay_rows:
        run_dir = Path(str(replay["run_dir"]))
        if not run_dir.is_absolute():
            run_dir = root / run_dir
        meta = _load_json(run_dir / "meta.json")
        component = meta["components"]["detector"]
        assistant_configs.append(
            {
                "checkpoint": _canonical_path(component["weights"]),
                "proposal_threshold": float(component["default_conf"]),
                "identity_commit_threshold": float(component["identity_commit_conf"]),
            }
        )
    expected_checkpoint = _canonical_path(checkpoint)
    if any(item["checkpoint"] != expected_checkpoint for item in assistant_configs):
        raise ValueError("Assistant sessions do not all use the adopted detector checkpoint")
    proposal_thresholds = {item["proposal_threshold"] for item in assistant_configs}
    commit_thresholds = {item["identity_commit_threshold"] for item in assistant_configs}
    if proposal_thresholds != {proposal_threshold}:
        raise ValueError("Assistant proposal threshold differs from detector evaluation")
    if len(commit_thresholds) != 1:
        raise ValueError("Assistant identity-commit threshold changed across sessions")
    identity_commit_threshold = next(iter(commit_thresholds))

    proposal_rows = _load_jsonl(robot_proposal_observations_path)
    commit_rows = _load_jsonl(robot_commit_observations_path)
    if set(proposal_rows) != set(commit_rows):
        raise ValueError("Robot observation caches do not contain identical frames")
    if len(proposal_rows) != 360:
        raise ValueError(f"Expected 360 robot observations, found {len(proposal_rows)}")

    subset_matches = 0
    commit_checkpoint_digests: set[str] = set()
    proposal_confidences: list[float] = []
    commit_confidences: list[float] = []
    for observation_id in sorted(proposal_rows):
        proposal_detections = _raw_detections(proposal_rows[observation_id])
        commit_detections = _raw_detections(commit_rows[observation_id])
        proposal_confidences.extend(float(item["confidence"]) for item in proposal_detections)
        commit_confidences.extend(float(item["confidence"]) for item in commit_detections)
        expected_commit = [
            item
            for item in proposal_detections
            if float(item["confidence"]) >= identity_commit_threshold
        ]
        if _detection_signature(expected_commit) != _detection_signature(commit_detections):
            raise ValueError(
                "Commit-threshold detections are not an exact subset at "
                f"{observation_id}"
            )
        subset_matches += 1
        checkpoint_meta = commit_rows[observation_id].get("metadata", {}).get(
            "detector_checkpoint", {}
        )
        digest = str(checkpoint_meta.get("sha256", "")).lower()
        if digest:
            commit_checkpoint_digests.add(digest)
    if commit_checkpoint_digests != {checkpoint_digest}:
        raise ValueError("Robot observation checkpoint fingerprint does not match")
    if not proposal_confidences or min(proposal_confidences) < proposal_threshold:
        raise ValueError("Robot proposal cache contains detections below its threshold")
    if not any(value < identity_commit_threshold for value in proposal_confidences):
        raise ValueError("Robot proposal cache does not preserve lower-confidence evidence")
    if commit_confidences and min(commit_confidences) < identity_commit_threshold:
        raise ValueError("Robot commit cache contains detections below commit threshold")

    return {
        "schema": "inspect.detector-protocol-audit.v1",
        "status": "pass",
        "checkpoint": {
            "model": "YOLO26s + PI-TwinSwap",
            "bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_digest,
        },
        "operating_points": {
            "evidence_proposal_confidence": proposal_threshold,
            "identity_commit_confidence": identity_commit_threshold,
            "detector_evaluation_iou": float(detector["iou_match"]),
        },
        "assistant_replay": {
            "sessions": len(replay_rows),
            "same_checkpoint": True,
            "same_thresholds": True,
        },
        "robot_observations": {
            "frames": len(proposal_rows),
            "same_frame_keys": True,
            "commit_is_exact_threshold_subset": True,
            "exact_subset_frames": subset_matches,
            "minimum_proposal_confidence": min(proposal_confidences),
            "minimum_commit_confidence": min(commit_confidences),
        },
        "detector_test": {
            "images": int(detector["images"]),
            "ground_truth_boxes": int(detector["gt_boxes"]),
            "box_recall": float(detector["box_oracle_recall"]),
            "correct_class_recall": float(detector["correct_class_recall"]),
            "matched_class_accuracy": float(detector["matched_class_accuracy"]),
            "hard_pair_error_per_gt": float(detector["hard_pair_error_per_gt"]),
            "identity_margin": float(detector["avg_identity_margin"]),
            "detections_per_frame": float(detector["avg_detections_per_frame"]),
            "correct_evidence_f1": detector_f1,
            "matched_baseline_correct_evidence_f1": baseline_f1,
            "improves_over_matched_baselines": True,
        },
        "interpretation": (
            "The 0.10 operating point preserves candidate object evidence; "
            "identity commitment remains separately gated at 0.50."
        ),
        "source_paths_anonymized": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--detector-summary", type=Path, required=True)
    parser.add_argument("--assistant-replay-summary", type=Path, required=True)
    parser.add_argument("--robot-proposal-observations", type=Path, required=True)
    parser.add_argument("--robot-commit-observations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_audit(
        repository_root=args.root,
        detector_summary_path=args.detector_summary,
        assistant_replay_summary_path=args.assistant_replay_summary,
        robot_proposal_observations_path=args.robot_proposal_observations,
        robot_commit_observations_path=args.robot_commit_observations,
        checkpoint_path=args.checkpoint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)
    print(f"status={audit['status']}")
    print(f"checkpoint_sha256={audit['checkpoint']['sha256']}")


if __name__ == "__main__":
    main()
