from inspect_runtime.components.temporal import ByteTrackLiteFusion
from inspect_runtime.core_types import Detection


GROUPS = [["type_7_gear", "type_3_gear"]]


def detection(name: str, confidence: float = 0.9, x: float = 10.0) -> Detection:
    return Detection(name=name, xyxy=(x, 10.0, x + 20.0, 30.0), confidence=confidence)


def tracker(**overrides) -> ByteTrackLiteFusion:
    options = {
        "track_high_thresh": 0.1,
        "track_low_thresh": 0.01,
        "new_track_thresh": 0.1,
        "match_iou_thr": 0.25,
        "min_confirmed_hits": 1,
        "identity_groups": GROUPS,
        "identity_smoothing_alpha": 0.8,
        "identity_switch_margin": 0.12,
        "identity_commit_conf": 0.5,
        "identity_commit_margin": 0.12,
    }
    options.update(overrides)
    return ByteTrackLiteFusion(**options)


def test_single_hard_pair_glitch_keeps_identity_and_track() -> None:
    fusion = tracker()
    first = fusion.update([detection("type_7_gear")])[0]
    second = fusion.update([detection("type_3_gear")])[0]

    assert second.meta["track_id"] == first.meta["track_id"]
    assert second.name == "type_7_gear"
    assert second.meta["raw_name"] == "type_3_gear"


def test_sustained_counterfactual_evidence_can_switch_identity() -> None:
    fusion = tracker(identity_smoothing_alpha=0.6, identity_switch_margin=0.05)
    first = fusion.update([detection("type_7_gear")])[0]
    latest = first
    for _ in range(6):
        latest = fusion.update([detection("type_3_gear")])[0]

    assert latest.meta["track_id"] == first.meta["track_id"]
    assert latest.name == "type_3_gear"
    assert latest.meta["identity_safe"] is True


def test_unrelated_classes_do_not_share_tracks() -> None:
    fusion = tracker()
    first = fusion.update([detection("type_7_gear")])[0]
    second = fusion.update([detection("type_2_gear")])[0]

    assert first.meta["track_id"] != second.meta["track_id"]


def test_evidence_preserving_mode_never_drops_current_detections() -> None:
    fusion = tracker(preserve_current_detections=True, min_confirmed_hits=2)
    current = [
        detection("type_7_gear", confidence=0.9, x=10.0),
        detection("type_2_gear", confidence=0.06, x=80.0),
    ]
    output = fusion.update(current)

    assert len(output) == len(current)
    assert output[0].meta["track_confirmed"] is False
    assert output[0].meta["identity_safe"] is False
    assert output[1].meta["tracked"] is False
