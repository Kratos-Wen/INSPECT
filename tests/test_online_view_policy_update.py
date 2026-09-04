from __future__ import annotations

from inspect_system.active_view.active_selector import ActiveSelector
from inspect_system.active_view.object_centric_evidence_memory import (
    ObjectCentricEvidenceMemory,
)
from inspect_system.active_view.object_centric_reveal import (
    ObjectCentricCalibratedRevealModel,
)
from inspect_system.active_view.requirement_model import RequirementCalibration
from inspect_system.active_view.reveal_model import PriorTableRevealModel
from inspect_system.active_view.view_lattice import SIX_VIEWS
from inspect_system.learned_view_lattice_policy import LearnedViewLatticePolicy
from inspect_system.robot_adapter import RecordingRobotAdapter, RobotProceduralDecisionLoop
from inspect_system.types import (
    ActiveObservationDecision,
    ProceduralEvidenceGraph,
    RobotObservation,
    VerificationResult,
)


ROLE = "identity_disambiguation_view"


def verification(
    observation_id: str,
    *,
    role_score: float,
    missing: bool,
    action: str,
    verified: bool = False,
) -> VerificationResult:
    return VerificationResult(
        observation_id=observation_id,
        predicted_state="small_gear_inserted",
        verified=verified,
        confidence=0.9 if verified else 0.3,
        evidence_coverage=role_score,
        observed_evidence=[] if missing else [ROLE],
        missing_evidence=[ROLE] if missing else [],
        contradicted_evidence=[],
        next_step_admissible=verified,
        anomaly=False,
        recommended_action=action,
        metadata={
            "claim_id": "small_gear_inserted",
            "counterfactual_family": "identity",
            "role_scores": {ROLE: role_score},
            "role_threshold": 0.65,
        },
    )


def test_post_move_update_is_session_local_and_updates_r_and_pi() -> None:
    memory = ObjectCentricEvidenceMemory()
    model = ObjectCentricCalibratedRevealModel(
        base_model=PriorTableRevealModel(),
        memory=memory,
        calibration_strength=1.0,
        support_strength=0.1,
    )
    policy = LearnedViewLatticePolicy(
        selector=ActiveSelector(model=model, views=SIX_VIEWS, lambda_cost=0.0),
        requirement_counts={},
        requirement_calibration=RequirementCalibration(),
    )
    before_observation = RobotObservation(
        observation_id="trial_001_V0",
        view_id="V0",
        metadata={"episode_id": "trial_001"},
    )
    after_observation = RobotObservation(
        observation_id="trial_001_V1",
        view_id="V1",
        metadata={"episode_id": "trial_001"},
    )
    decision = ActiveObservationDecision(
        observation_id="trial_001_V0",
        selected_view="V1",
        utility=0.4,
        expected_observed_evidence=[ROLE],
        unresolved_evidence=[ROLE],
        candidate_scores={"V1": 0.4},
    )

    update = policy.update_after_observation(
        before_observation=before_observation,
        before_verification=verification(
            "trial_001_V0", role_score=0.1, missing=True, action="observe"
        ),
        decision=decision,
        after_observation=after_observation,
        after_verification=verification(
            "trial_001_V1",
            role_score=0.9,
            missing=False,
            action="continue",
            verified=True,
        ),
    )

    assert update["applied"] is True
    assert update["source"] == "post_move_reverification"
    assert policy.session_requirement_counts["trial_001"]["gear_inserted"][ROLE] > 0.0
    assert any(atom.session_id == "trial_001" for atom in memory.atoms)
    assert not any(atom.session_id is None for atom in memory.atoms)

    observed_sessions = []
    original_predict = memory.predict

    def tracked_predict(context, motion, session_id=None):
        observed_sessions.append(session_id)
        return original_predict(context, motion, session_id=session_id)

    memory.predict = tracked_predict  # type: ignore[method-assign]
    model.candidate_affordance_factor(
        "gear_inserted",
        ROLE,
        "V0",
        "V1",
        SIX_VIEWS,
        context={"counterfactual_family": "identity", "session_id": "trial_001"},
    )
    assert observed_sessions == [None, "trial_001"]


def test_online_update_treats_unchanged_missing_evidence_as_unknown() -> None:
    memory = ObjectCentricEvidenceMemory()
    model = ObjectCentricCalibratedRevealModel(
        base_model=PriorTableRevealModel(),
        memory=memory,
        calibration_strength=1.0,
        support_strength=0.1,
    )
    policy = LearnedViewLatticePolicy(
        selector=ActiveSelector(model=model, views=SIX_VIEWS, lambda_cost=0.0),
        requirement_counts={},
        requirement_calibration=RequirementCalibration(),
        online_gain_deadband=0.05,
    )
    before_observation = RobotObservation(
        observation_id="trial_002_V0",
        view_id="V0",
        metadata={"episode_id": "trial_002"},
    )
    after_observation = RobotObservation(
        observation_id="trial_002_V1",
        view_id="V1",
        metadata={"episode_id": "trial_002"},
    )
    decision = ActiveObservationDecision(
        observation_id="trial_002_V0",
        selected_view="V1",
        utility=0.4,
        expected_observed_evidence=[ROLE],
        unresolved_evidence=[ROLE],
        candidate_scores={"V1": 0.4},
    )

    update = policy.update_after_observation(
        before_observation=before_observation,
        before_verification=verification(
            "trial_002_V0", role_score=0.10, missing=True, action="observe"
        ),
        decision=decision,
        after_observation=after_observation,
        after_verification=verification(
            "trial_002_V1", role_score=0.11, missing=True, action="observe"
        ),
    )

    assert update["applied"] is False
    assert update["role_updates"] == []
    assert update["gain_deadband"] == 0.05
    assert not memory.atoms
    assert "trial_002" not in policy.session_requirement_counts

def test_partial_evidence_delta_updates_transport_but_not_requirement() -> None:
    memory = ObjectCentricEvidenceMemory()
    model = ObjectCentricCalibratedRevealModel(
        base_model=PriorTableRevealModel(),
        memory=memory,
        calibration_strength=1.0,
        support_strength=0.1,
    )
    policy = LearnedViewLatticePolicy(
        selector=ActiveSelector(model=model, views=SIX_VIEWS, lambda_cost=0.0),
        requirement_counts={},
        requirement_calibration=RequirementCalibration(),
    )
    before_observation = RobotObservation(
        observation_id="trial_003_V0",
        view_id="V0",
        metadata={"episode_id": "trial_003"},
    )
    after_observation = RobotObservation(
        observation_id="trial_003_V1",
        view_id="V1",
        metadata={"episode_id": "trial_003"},
    )
    decision = ActiveObservationDecision(
        observation_id="trial_003_V0",
        selected_view="V1",
        utility=0.4,
        expected_observed_evidence=[ROLE],
        unresolved_evidence=[ROLE],
        candidate_scores={"V1": 0.4},
    )

    update = policy.update_after_observation(
        before_observation=before_observation,
        before_verification=verification(
            "trial_003_V0", role_score=0.1, missing=True, action="observe"
        ),
        decision=decision,
        after_observation=after_observation,
        after_verification=verification(
            "trial_003_V1", role_score=0.9, missing=False, action="observe"
        ),
    )

    assert update["applied"] is True
    assert update["reason"] == "causal_evidence_delta"
    assert update["update_scope"] == "transport_only_partial"
    assert update["claim_resolved_after_move"] is False
    assert update["role_updates"]
    assert memory.atoms
    assert "trial_003" not in policy.session_requirement_counts


def test_online_update_rejects_workpiece_state_change() -> None:
    memory = ObjectCentricEvidenceMemory()
    model = ObjectCentricCalibratedRevealModel(
        base_model=PriorTableRevealModel(),
        memory=memory,
    )
    policy = LearnedViewLatticePolicy(
        selector=ActiveSelector(model=model, views=SIX_VIEWS, lambda_cost=0.0),
        requirement_counts={},
        requirement_calibration=RequirementCalibration(),
    )
    before_observation = RobotObservation(
        observation_id="trial_004_V0",
        view_id="V0",
        metadata={"episode_id": "trial_004"},
    )
    after_observation = RobotObservation(
        observation_id="trial_004_V1",
        view_id="V1",
        metadata={"episode_id": "trial_004", "state_changed": True},
    )
    decision = ActiveObservationDecision(
        observation_id="trial_004_V0",
        selected_view="V1",
        utility=0.4,
        expected_observed_evidence=[ROLE],
        unresolved_evidence=[ROLE],
        candidate_scores={"V1": 0.4},
    )

    update = policy.update_after_observation(
        before_observation=before_observation,
        before_verification=verification(
            "trial_004_V0", role_score=0.1, missing=True, action="observe"
        ),
        decision=decision,
        after_observation=after_observation,
        after_verification=verification(
            "trial_004_V1", role_score=0.9, missing=False, action="continue"
        ),
    )

    assert update["applied"] is False
    assert update["reason"] == "non_causal_transition:state_changed"
    assert not memory.atoms

def test_robot_loop_updates_only_after_reverification() -> None:
    observations = [
        RobotObservation(observation_id="trial_V0", view_id="V0"),
        RobotObservation(observation_id="trial_V1", view_id="V1"),
    ]
    robot = RecordingRobotAdapter(observations)
    before = verification("trial_V0", role_score=0.1, missing=True, action="observe")
    after = verification(
        "trial_V1",
        role_score=0.9,
        missing=False,
        action="continue",
        verified=True,
    )

    class Verifier:
        def __init__(self) -> None:
            self.results = iter([before, after])

        def verify(self, observation):
            return next(self.results)

    class Policy:
        def __init__(self) -> None:
            self.updates = []

        def select_view(self, observation, result, graph=None):
            return ActiveObservationDecision(
                observation_id=result.observation_id,
                selected_view="V1",
                utility=0.5,
                expected_observed_evidence=[ROLE],
                unresolved_evidence=[ROLE],
                candidate_scores={"V1": 0.5},
            )

        def update_after_observation(self, **kwargs):
            self.updates.append(kwargs)
            return {"applied": True, "source": "post_move_reverification"}

    policy = Policy()
    loop = RobotProceduralDecisionLoop(
        graph=ProceduralEvidenceGraph("g", [], {}, []),
        robot=robot,
        verifier=Verifier(),
        view_lattice_policy=policy,
        max_observation_moves=1,
    )
    result = loop.step(initial_view="V0")

    assert len(policy.updates) == 1
    assert policy.updates[0]["before_observation"].view_id == "V0"
    assert policy.updates[0]["after_observation"].view_id == "V1"
    assert policy.updates[0]["before_verification"].recommended_action == "observe"
    assert policy.updates[0]["after_verification"].recommended_action == "continue"
    assert result.metadata["active_view_online_update"]["applied"] is True
