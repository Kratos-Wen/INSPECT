from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
from inspect_runtime.core_types import Detection
from inspect_runtime.inspect_system.robot_observation_builder import (
    _checkpoint_identity,
    _serialize_detection,
)


def test_detection_serialization_preserves_identity_gate_metadata() -> None:
    detection = Detection(
        name="Type 3 Gear",
        xyxy=(1.0, 2.0, 10.0, 12.0),
        confidence=0.73,
        meta={
            "identity_safe": False,
            "proposal_only": True,
            "track_class_margin": 0.12,
            "track_class_posterior": 0.61,
            "track_id": 7,
            "non_serializable": {"ignored"},
        },
    )

    payload = _serialize_detection(detection)

    assert payload["name"] == "type_3_gear"
    assert payload["meta"]["identity_safe"] is False
    assert payload["meta"]["proposal_only"] is True
    assert payload["meta"]["track_class_margin"] == 0.12
    assert payload["meta"]["track_id"] == 7
    assert "non_serializable" not in payload["meta"]


def test_checkpoint_identity_is_path_anonymous_and_content_addressed(tmp_path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"frozen detector")

    identity = _checkpoint_identity(str(checkpoint))

    assert identity["name"] == "best.pt"
    assert set(identity) == {"name", "sha256"}
    assert len(identity["sha256"]) == 64
    assert str(tmp_path) not in str(identity)