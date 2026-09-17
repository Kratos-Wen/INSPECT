import pytest

from scripts.external_benchmarks.impact_query_truth import relabel_query, world_state_at
from scripts.train_assistance_evidence_scorer import feedback_window_frames, features_from_iteration
from scripts.complete_inspection_audit import rescore_trial
from scripts.calibrate_repaired_feedback_scorer import split_videos


def test_query_time_replaces_stale_source_truth_without_changing_prediction():
    annotation = {"initial_state_vector": [-1], "state_sequence": [
        {"frame": 10, "state": [1]}, {"frame": 20, "state": [0]}]}
    row = {"claim_id": "part.installed_correctly", "component_id": 0,
           "reference_query_frame": 10, "world_value": -1,
           "semantic_outcome": "contradicted", "claim_prediction": "supported",
           "raw_claim_decision": "supported", "utility": 0}
    fixed = relabel_query(row, annotation)
    assert fixed["utility"] == 2
    assert fixed["source_event_world_value"] == -1
    assert row["utility"] == 0
    assert fixed["claim_prediction"] == row["claim_prediction"]
    assert world_state_at(annotation, 9, 0) == -1
    assert world_state_at(annotation, 20, 0) == 0


def test_query_truth_rejects_unknown_semantics():
    with pytest.raises(ValueError):
        relabel_query({"claim_id": "part.other"}, {})


def test_feedback_window_cannot_cross_state_boundary():
    interval = {"start_frame_i": 10, "end_frame_i": 20}
    assert feedback_window_frames([1, 7, 10, 13, 16, 19, 22], 16, interval, 8) == [10, 13, 16]
    assert feedback_window_frames([1, 7, 10, 13, 16, 19, 22], 16, interval, 2) == [13, 16]


def test_empty_fused_detections_do_not_reintroduce_rejected_raw_identity():
    raw = [{"name": "type_3_gear", "confidence": 0.9, "xyxy": [0, 0, 10, 10]}]
    task = {"claim_id": "small_gear_inserted", "step_id": "S2", "assembly_set": "A"}
    assert features_from_iteration({"fused_detections": [], "raw_detections": raw}, task)["target_conf"] == 0
    assert features_from_iteration({"raw_detections": raw}, task)["target_conf"] == 0.9


def test_rescoring_updates_oracle_regret_and_margin_without_changing_action():
    trial = {"recording_id": "r", "source_frame": 0, "component_id": 0,
             "current_view": "front", "selected_view": "left", "moved": 1}
    scores = {("r", 0, 0, view): {"utility": utility, "semantic_outcome": "contradicted",
              "claim_prediction": "contradicted" if utility == 2 else "insufficient",
              "support_margin": -0.4} for view, utility in
              (("front", 0), ("left", 1), ("right", 2), ("top", 0))}
    result = rescore_trial(trial, scores)
    assert result["regret"] == 1
    assert result["gain"] == 1
    assert result["resolve_at_1"] == 0
    assert result["selected_counterfactual_margin"] == 0.4
    assert result["selected_view"] == trial["selected_view"]


def test_calibration_split_is_video_disjoint_and_keeps_singletons_in_training():
    samples = [{"video": "single.mp4", "claim_id": "gear", "label": "support"},
               *[{"video": f"v{i}.mp4", "claim_id": "gear", "label": "contradiction"} for i in range(5)]]
    train, validation = split_videos(samples)
    assert "single.mp4" in train
    assert not train & validation
    assert train | validation == {s["video"] for s in samples}
    assert len(validation) == 1
    assert split_videos(list(reversed(samples))) == (train, validation)
