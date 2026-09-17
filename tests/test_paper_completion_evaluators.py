"""Tests for matched evaluation metrics and execution-path accounting."""
from types import SimpleNamespace

from scripts.evaluate_frozen_policy_commitment import action_candidates, score
from scripts.evaluate_runtime_verifier_matched import InstrumentedVerifier, metrics


def test_conditional_recall_and_false_support_have_distinct_denominators():
    rows = [
        {"truth": "supported", "prediction": "supported"},
        {"truth": "supported", "prediction": "insufficient"},
        {"truth": "contradicted", "prediction": "supported"},
        {"truth": "insufficient", "prediction": "insufficient"},
    ]
    result = metrics(rows)
    assert result["supported_recall"] == .5
    assert result["supported_precision"] == .5
    assert result["false_support"] == .5
    assert result["accuracy"] == .5


def test_empty_denominators_are_unavailable_not_perfect():
    result = metrics([])
    assert result["false_support"] is None
    assert result["accuracy"] is None
    assert result["supported_precision"] is None


def test_uniform_expectation_uses_weights_not_expanded_row_counts():
    rows = [
        {"weight": 1., "truth": "supported", "current_decision": "supported",
         "selected_decision": "supported", "current_utility": 0, "selected_utility": 2},
        {"weight": .5, "truth": "contradicted", "current_decision": "insufficient",
         "selected_decision": "supported", "current_utility": 2, "selected_utility": 1},
        {"weight": .5, "truth": "contradicted", "current_decision": "insufficient",
         "selected_decision": "contradicted", "current_utility": 2, "selected_utility": 2},
    ]
    result = score(rows, "selected_decision")
    assert result["trial_mass"] == 2
    assert result["correct_resolution"] == .75
    assert result["false_support"] == .5
    assert result["fully_verifiable_mass"] == 1.5
    assert result["correct_resolution_at_u2"] == 1.


def test_current_u2_subset_uses_current_not_selected_utility():
    rows = [
        {"weight": 1., "truth": "supported", "current_decision": "supported",
         "current_utility": 0, "selected_utility": 2},
        {"weight": 1., "truth": "contradicted", "current_decision": "insufficient",
         "current_utility": 2, "selected_utility": 0},
    ]
    result = score(rows, "current_decision")
    assert result["fully_verifiable_mass"] == 1
    assert result["correct_resolution_at_u2"] == 0.


def test_expected_shortest_and_uniform_use_different_candidate_sets():
    record = {"trial_id": "a", "current_view": "V0", "selected_view": "EXPECTED", "triggered": True}
    assert action_candidates("Shortest-Move", record, {}) == ["V1", "V5"]
    assert action_candidates("Uniform Non-current", record, {}) == ["V1", "V2", "V3", "V4", "V5"]
    record["triggered"] = False
    assert action_candidates("Shortest-Move", record, {}) == ["V0"]


def test_oracle_averages_all_maximizers_without_verifier_peeking():
    record = {"trial_id": "a", "current_view": "V0", "selected_view": "ORACLE"}
    utilities = {("a", f"V{i}"): (2 if i in {1, 4} else 0) for i in range(6)}
    assert action_candidates("Oracle View", record, utilities) == ["V1", "V4"]


class ConstantScorer:
    def score_features(self, features, *, claim_id, step_id=""):
        return {"support_score": .8, "contradiction_score": .1, "visibility_score": .9}


def test_learned_triage_still_calls_specialized_checks_in_bootstrap():
    verifier = InstrumentedVerifier(ConstantScorer(), product_family="A",
                                   prerequisite_confirmation_frames=1)
    triage = SimpleNamespace(support_score=.9, contradiction_score=.05,
        visibility_score=.95, posterior_margin=.8, state="supported",
        insufficient_score=.05, features={})
    verifier.verify([], frame_index=1, proposed_step="S3", proposal_confidence=1,
                    image_shape=(100, 100), learned_triage=triage)
    assert verifier.execution["counterfactual_bootstrap_calls"] == 1
    assert verifier.execution["counterfactual_direct_calls"] == 0


def test_disabling_specialized_checks_really_skips_scorer_calls():
    verifier = InstrumentedVerifier(ConstantScorer(), product_family="A",
                                   specialized_counterfactual_enabled=False)
    verifier.verify([], frame_index=1, proposed_step="S2", proposal_confidence=1,
                    image_shape=(100, 100))
    assert sum(verifier.execution.values()) == 0


def test_readiness_only_ablation_preserves_bootstrap_path():
    verifier = InstrumentedVerifier(ConstantScorer(), product_family="A",
                                   readiness_check_only_disabled=True)
    assert verifier.memory_gate_enabled
    verifier.verify([], frame_index=1, proposed_step="S3", proposal_confidence=1,
                    image_shape=(100, 100))
    assert verifier.execution["counterfactual_bootstrap_calls"] == 1
    assert verifier.execution["counterfactual_direct_calls"] == 1


def test_active_counterfactual_ablation_preserves_prerequisite_checks():
    verifier = InstrumentedVerifier(ConstantScorer(), product_family="A",
                                   active_counterfactual_only_disabled=True)
    verifier.verify([], frame_index=1, proposed_step="S3", proposal_confidence=1,
                    image_shape=(100, 100))
    assert verifier.execution["counterfactual_bootstrap_calls"] == 1
    assert verifier.execution["counterfactual_direct_calls"] == 0
