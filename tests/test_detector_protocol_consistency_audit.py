import json
from pathlib import Path

import pytest

from scripts.audit_detector_protocol_consistency import build_audit, sha256


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _observation(index: int, detections: list[dict], digest: str | None = None) -> dict:
    metadata = {"raw_detections": detections}
    if digest is not None:
        metadata["detector_checkpoint"] = {"name": "best.pt", "sha256": digest}
    return {"observation_id": f"view-{index:03d}", "metadata": metadata}


def _build_fixture(tmp_path: Path) -> dict[str, Path]:
    checkpoint = tmp_path / "models" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"pi-twinswap")
    digest = sha256(checkpoint)

    detector_summary = tmp_path / "detector.json"
    _write_json(
        detector_summary,
        [
            {
                "model": "YOLO26s Plain",
                "precision_correct_class": 0.12,
                "correct_class_recall": 0.09,
            },
            {
                "model": "YOLO26s + Synthetic-Aug",
                "precision_correct_class": 0.18,
                "correct_class_recall": 0.09,
            },
            {
                "model": "YOLO26s + PI-TwinSwap",
                "weights": str(checkpoint),
                "conf": 0.1,
                "iou_match": 0.5,
                "images": 304,
                "gt_boxes": 905,
                "box_oracle_recall": 0.45,
                "correct_class_recall": 0.21,
                "matched_class_accuracy": 0.46,
                "hard_pair_error_per_gt": 0.11,
                "avg_identity_margin": 0.04,
                "avg_detections_per_frame": 3.6,
                "precision_correct_class": 0.17,
            }
        ],
    )

    replay_summary = tmp_path / "replay.json"
    replay_rows = []
    for index in range(2):
        run_dir = tmp_path / "runs" / str(index)
        _write_json(
            run_dir / "meta.json",
            {
                "components": {
                    "detector": {
                        "weights": str(checkpoint),
                        "default_conf": 0.1,
                        "identity_commit_conf": 0.5,
                    }
                }
            },
        )
        replay_rows.append({"run_dir": str(run_dir)})
    _write_json(replay_summary, replay_rows)

    low_rows = []
    commit_rows = []
    low = [
        {"name": "gear", "confidence": 0.2, "xyxy": [1, 2, 3, 4]},
        {"name": "housing", "confidence": 0.8, "xyxy": [5, 6, 7, 8]},
    ]
    high = [low[1]]
    for index in range(360):
        low_rows.append(_observation(index, low))
        commit_rows.append(_observation(index, high, digest))
    proposal_observations = tmp_path / "robot-low.jsonl"
    commit_observations = tmp_path / "robot-high.jsonl"
    _write_jsonl(proposal_observations, low_rows)
    _write_jsonl(commit_observations, commit_rows)
    return {
        "checkpoint": checkpoint,
        "detector_summary": detector_summary,
        "replay_summary": replay_summary,
        "proposal_observations": proposal_observations,
        "commit_observations": commit_observations,
    }


def test_detector_protocol_audit_is_path_free_and_complete(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    audit = build_audit(
        repository_root=tmp_path,
        detector_summary_path=paths["detector_summary"],
        assistant_replay_summary_path=paths["replay_summary"],
        robot_proposal_observations_path=paths["proposal_observations"],
        robot_commit_observations_path=paths["commit_observations"],
        checkpoint_path=paths["checkpoint"],
    )

    assert audit["status"] == "pass"
    assert audit["assistant_replay"]["sessions"] == 2
    assert audit["robot_observations"]["exact_subset_frames"] == 360
    assert audit["detector_test"]["improves_over_matched_baselines"]
    assert audit["operating_points"] == {
        "evidence_proposal_confidence": 0.1,
        "identity_commit_confidence": 0.5,
        "detector_evaluation_iou": 0.5,
    }
    assert str(tmp_path) not in json.dumps(audit)


def test_detector_protocol_audit_rejects_non_subset_predictions(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in paths["commit_observations"].read_text().splitlines()
    ]
    rows[0]["metadata"]["raw_detections"][0]["confidence"] = 0.7
    _write_jsonl(paths["commit_observations"], rows)

    with pytest.raises(ValueError, match="exact subset"):
        build_audit(
            repository_root=tmp_path,
            detector_summary_path=paths["detector_summary"],
            assistant_replay_summary_path=paths["replay_summary"],
            robot_proposal_observations_path=paths["proposal_observations"],
            robot_commit_observations_path=paths["commit_observations"],
            checkpoint_path=paths["checkpoint"],
        )
