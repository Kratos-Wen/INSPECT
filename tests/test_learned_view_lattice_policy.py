from __future__ import annotations

import json

from inspect_system.active_view.reveal_model import PriorTableRevealModel
from inspect_system.learned_view_lattice_policy import LearnedViewLatticePolicy
from inspect_system.types import RobotObservation, VerificationResult, ViewCandidate


def _verification(**metadata):
    return VerificationResult(
        observation_id="trial_001_V0",
        predicted_state="step2_small_gear_inserted",
        verified=False,
        confidence=0.3,
        evidence_coverage=0.2,
        observed_evidence=[],
        missing_evidence=["identity unclear", "insertion unclear"],
        contradicted_evidence=[],
        next_step_admissible=False,
        anomaly=False,
        recommended_action="observe",
        metadata=metadata,
    )


def test_learned_policy_uses_current_evidence_and_lattice_only(tmp_path) -> None:
    model_path = tmp_path / "reveal.json"
    PriorTableRevealModel().save(model_path)
    report_path = tmp_path / "requirements.json"
    report_path.write_text(
        json.dumps(
            {
                "trainable_events": [
                    {
                        "claim_id": "gear_inserted",
                        "evidence_role": "identity_disambiguation_view",
                        "transfer_weight": 1.0,
                        "label_confidence": 1.0,
                        "evidence_stability": 1.0,
                        "evidence_importance": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    policy = LearnedViewLatticePolicy.from_paths(
        reveal_model=model_path,
        requirement_report=report_path,
        candidates=[ViewCandidate(view_id=f"V{index}", visible_evidence=[]) for index in range(6)],
        lambda_cost=0.0,
        tau_view=-1.0,
    )
    decision = policy.select_view(
        RobotObservation(
            observation_id="trial_001_V0",
            view_id="V0",
            metadata={"claim_id": "small_gear_inserted"},
        ),
        _verification(),
    )

    assert decision.selected_view in {"V1", "V2", "V3", "V4", "V5"}
    assert decision.metadata["uses_candidate_view_images"] is False
    assert decision.metadata["uses_robot_utility_labels"] is False
    assert decision.metadata["claim_id"] == "gear_inserted"


def test_learned_policy_rejects_forbidden_context(tmp_path) -> None:
    model_path = tmp_path / "reveal.json"
    PriorTableRevealModel().save(model_path)
    policy = LearnedViewLatticePolicy.from_paths(reveal_model=model_path)
    observation = RobotObservation(
        observation_id="trial_001_V0",
        view_id="V0",
        metadata={"candidate_images_used": True},
    )

    try:
        policy.select_view(observation, _verification())
    except RuntimeError as error:
        assert "forbidden" in str(error)
    else:
        raise AssertionError("Forbidden candidate-image context was not rejected.")


def test_claim_structured_policy_preserves_typed_claims(tmp_path) -> None:
    model_path = tmp_path / "reveal.json"
    PriorTableRevealModel().save(model_path)
    report_path = tmp_path / "requirements.json"
    report_path.write_text(
        json.dumps(
            {
                "requirement_events": [
                    {
                        "claim_id": "gear_inserted",
                        "metadata": {"source_claim_id": "small_gear_inserted"},
                        "evidence_role": "identity_disambiguation_view",
                        "requirement_weight": 1.0,
                        "video": "small_gear_session",
                        "after_frame": 12,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    policy = LearnedViewLatticePolicy.from_claim_structured_paths(
        reveal_model=model_path,
        requirement_report=report_path,
        lambda_cost=0.0,
        tau_view=-1.0,
    )
    decision = policy.select_view(
        RobotObservation(
            observation_id="trial_001_V0",
            view_id="V0",
            metadata={"claim_id": "small_gear_inserted"},
        ),
        _verification(),
    )

    assert policy.selector_mode == "claim_structured"
    assert policy.typed_requirement_enabled is True
    assert decision.metadata["claim_id"] == "small_gear_inserted"
    assert decision.metadata["requirement_conditioning"] == "typed_claim"
    assert decision.metadata["uses_candidate_view_images"] is False


def test_claim_structured_policy_retains_best_observed_evidence(tmp_path) -> None:
    model_path = tmp_path / "reveal.json"
    PriorTableRevealModel().save(model_path)
    policy = LearnedViewLatticePolicy.from_claim_structured_paths(
        reveal_model=model_path,
        lambda_cost=0.0,
        tau_view=-1.0,
    )
    session = "trial_checkpoint"
    weak = _verification(
        claim_id="small_gear_inserted",
        counterfactual_family="identity",
        role_scores={"identity_disambiguation_view": 0.1},
        role_threshold=0.65,
    )
    strong = _verification(
        claim_id="small_gear_inserted",
        counterfactual_family="identity",
        role_scores={"identity_disambiguation_view": 0.9},
        role_threshold=0.65,
    )
    policy.select_view(
        RobotObservation(
            observation_id=f"{session}_V0",
            view_id="V0",
            metadata={"session_id": session},
        ),
        weak,
    )
    policy.select_view(
        RobotObservation(
            observation_id=f"{session}_V1",
            view_id="V1",
            metadata={"session_id": session},
        ),
        strong,
    )

    assert policy.best_evidence_view(
        session, "small_gear_inserted", "identity"
    ) == "V1"
