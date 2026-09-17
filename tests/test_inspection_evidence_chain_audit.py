from scripts.audit_inspection_evidence_chain import category, scorer_observation, state_at, summarize_robot


def test_insufficient_is_not_a_correct_commitment():
    assert category("insufficient", "supported") == "insufficient"
    assert category("supported", "supported") == "correct_resolution"
    assert category("supported", "contradicted") == "incorrect_supported"


def test_prediction_input_drops_annotation_metadata():
    observation = {"metadata": {
        "detections": [], "frame_shape": [720, 1280],
        "claim_outcome": "supported", "actual_parts": ["ground_truth_part"],
        "human_utility_0_1_2": 2,
    }}
    assert scorer_observation(observation) == {
        "metadata": {"detections": [], "frame_shape": [720, 1280]}
    }


def test_state_at_uses_the_query_time_not_the_event_time():
    annotation = {
        "initial_state_vector": [0],
        "state_sequence": [{"frame": 10, "state": [-1]}, {"frame": 32, "state": [1]}],
    }
    assert state_at(annotation, 0, 0) == 0
    assert state_at(annotation, 10, 0) == -1
    assert state_at(annotation, 31, 0) == -1
    assert state_at(annotation, 32, 0) == 1
    assert state_at(annotation, 55, 0) == 1


def test_gate_attribution_separates_blocked_correct_and_incorrect_decisions():
    row = {
        "truth": "supported", "raw_decision": "supported",
        "discovery_decision": "insufficient", "final_decision": "insufficient",
        "first_abstention_stage": "discovery_gate",
        "identity_available": False, "relation_available": False,
        "support_evidence_available": False, "contradiction_evidence_available": False,
        "target_conf": 0.0, "housing_role_conf": 0.0,
    }
    result = summarize_robot([row, {**row, "truth": "contradicted"}])
    assert result["trials"] == 2
    assert result["correct_raw_decisions_blocked"] == 1
    assert result["incorrect_raw_decisions_blocked"] == 1
    assert result["commit_gate"] == {"insufficient": 2}
