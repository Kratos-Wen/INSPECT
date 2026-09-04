import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from inspect_runtime.assistant.evidence import select_evidence
from inspect_runtime.assistant.responder import render_answer
from inspect_runtime.assistant.types import AssistantSnapshot, ResolvedAssistantQuery
from inspect_runtime.components.kb import KnowledgeBase
from inspect_runtime.core_types import Detection
from inspect_runtime.inspect_system.live_claim_verifier import LiveClaimVerifier


class FixedScorer:
    metadata = {"training_source": "test", "uses_robot_view_training": False}

    def __init__(self, support: float, contradiction: float) -> None:
        self.support = support
        self.contradiction = contradiction

    def score_features(self, features, *, claim_id, step_id=""):
        return {
            "support_score": self.support,
            "contradiction_score": self.contradiction,
            "visibility_score": max(float(features.get("target_conf", 0.0)), 0.8),
        }


class ClaimAwareScorer(FixedScorer):
    def score_features(self, features, *, claim_id, step_id=""):
        if claim_id == "small_gear_inserted":
            return {"support_score": 0.1, "contradiction_score": 0.9, "visibility_score": 0.8}
        return {"support_score": 0.9, "contradiction_score": 0.1, "visibility_score": 0.8}


class TargetEvidenceScorer(FixedScorer):
    def score_features(self, features, *, claim_id, step_id=""):
        support = float(features.get("target_conf", 0.0))
        return {"support_score": support, "contradiction_score": 0.0, "visibility_score": support}


def detections_a():
    return [
        Detection("type_5_gearbox_housing", (10, 10, 100, 100), 0.9, {"identity_safe": True}),
        Detection("type_3_gear", (35, 35, 60, 60), 0.8, {"identity_safe": True}),
    ]


def test_supported_claim_commits_step_after_causal_family_confirmation() -> None:
    verifier = LiveClaimVerifier(FixedScorer(0.9, 0.1), family_confirmation_frames=2)
    first = verifier.verify(
        detections_a(),
        frame_index=1,
        proposed_step="S2",
        proposal_confidence=0.8,
        image_shape=(120, 160),
    )
    second = verifier.verify(
        detections_a(),
        frame_index=2,
        proposed_step="S2",
        proposal_confidence=0.8,
        image_shape=(120, 160),
    )

    assert first.state == "insufficient"
    assert second.product_family == "A"
    assert second.state == "supported"
    assert second.committed_step == "S2"


def test_explicitly_contradicted_prerequisite_blocks_later_claim() -> None:
    verifier = LiveClaimVerifier(ClaimAwareScorer(0.0, 0.0), family_confirmation_frames=1)
    small = verifier.verify(
        detections_a(),
        frame_index=1,
        proposed_step="S2",
        proposal_confidence=0.8,
        image_shape=(120, 160),
    )
    later = verifier.verify(
        detections_a(),
        frame_index=2,
        proposed_step="S3",
        proposal_confidence=0.8,
        image_shape=(120, 160),
    )

    assert small.state == "contradicted"
    assert later.admissible is False
    assert later.state == "contradicted"
    assert later.committed_step == ""


def test_ambiguous_family_abstains_and_requests_identity_evidence() -> None:
    detections = [
        Detection("type_5_gearbox_housing", (10, 10, 100, 100), 0.5, {"identity_safe": True}),
        Detection("type_6_gearbox_housing", (10, 10, 100, 100), 0.5, {"identity_safe": True}),
    ]
    verifier = LiveClaimVerifier(FixedScorer(0.9, 0.1), family_confirmation_frames=1)
    decision = verifier.verify(
        detections,
        frame_index=1,
        proposed_step="S2",
        proposal_confidence=0.8,
        image_shape=(120, 160),
    )

    assert decision.state == "insufficient"
    assert "identity_disambiguation" in decision.missing_roles


def test_unknown_prerequisite_blocks_support_through_memory_gate() -> None:
    verifier = LiveClaimVerifier(
        FixedScorer(0.9, 0.1),
        product_family="A",
        prerequisite_bootstrap_enabled=False,
        family_confirmation_frames=1,
    )
    decision = verifier.verify(
        detections_a(),
        frame_index=1,
        proposed_step="S3",
        proposal_confidence=0.8,
        image_shape=(120, 160),
    )

    assert decision.admissible is True
    assert decision.memory_ready is False
    assert decision.features["decision_admissible"] == 1.0
    assert decision.features["decision_memory_ready"] == 0.0
    assert decision.features["decision_step_match"] == 1.0
    assert decision.state == "insufficient"
    assert "history_precondition" in decision.missing_roles


def test_visible_prerequisite_bootstraps_after_temporal_confirmation() -> None:
    verifier = LiveClaimVerifier(
        FixedScorer(0.9, 0.1),
        product_family="A",
        prerequisite_confirmation_frames=2,
        family_confirmation_frames=1,
    )
    first = verifier.verify(
        detections_a(),
        frame_index=1,
        proposed_step="S3",
        proposal_confidence=0.8,
        image_shape=(120, 160),
    )
    second = verifier.verify(
        detections_a(),
        frame_index=2,
        proposed_step="S3",
        proposal_confidence=0.8,
        image_shape=(120, 160),
    )

    assert first.memory_ready is False
    assert first.state == "insufficient"
    assert second.memory_ready is True
    assert second.state == "supported"
    assert second.committed_step == "S3"


def test_prerequisite_bootstrap_never_overwrites_contradiction() -> None:
    verifier = LiveClaimVerifier(
        ClaimAwareScorer(0.0, 0.0),
        product_family="A",
        prerequisite_confirmation_frames=1,
        family_confirmation_frames=1,
    )
    verifier.verify(
        detections_a(),
        frame_index=1,
        proposed_step="S2",
        proposal_confidence=0.8,
        image_shape=(120, 160),
    )
    later = verifier.verify(
        detections_a(),
        frame_index=2,
        proposed_step="S3",
        proposal_confidence=0.8,
        image_shape=(120, 160),
    )

    assert later.admissible is False
    assert later.state == "contradicted"


def test_specialized_counterfactual_blocks_ambiguous_hard_pair_support() -> None:
    detections = detections_a() + [
        Detection("type_6_gearbox_housing", (10, 10, 100, 100), 0.9, {"identity_safe": True}),
        Detection("type_7_gear", (35, 35, 60, 60), 0.8, {"identity_safe": True}),
    ]
    full = LiveClaimVerifier(
        TargetEvidenceScorer(0.0, 0.0),
        product_family="A",
        counterfactual_margin=0.15,
        family_confirmation_frames=1,
    )
    without_counterfactual = LiveClaimVerifier(
        TargetEvidenceScorer(0.0, 0.0),
        product_family="A",
        counterfactual_margin=0.15,
        specialized_counterfactual_enabled=False,
        family_confirmation_frames=1,
    )

    full_decision = full.verify(
        detections,
        frame_index=1,
        proposed_step="S2",
        proposal_confidence=0.8,
        image_shape=(120, 160),
    )
    ablated_decision = without_counterfactual.verify(
        detections,
        frame_index=1,
        proposed_step="S2",
        proposal_confidence=0.8,
        image_shape=(120, 160),
    )

    assert full_decision.counterfactual_margin < 0.15
    assert full_decision.state == "insufficient"
    assert ablated_decision.state == "supported"


def test_admissibility_and_memory_gates_are_independently_switchable() -> None:
    verifier = LiveClaimVerifier(
        ClaimAwareScorer(0.0, 0.0),
        product_family="A",
        admissibility_gate_enabled=False,
        memory_gate_enabled=False,
        family_confirmation_frames=1,
    )
    prerequisite = verifier.verify(
        detections_a(),
        frame_index=1,
        proposed_step="S2",
        proposal_confidence=0.8,
        image_shape=(120, 160),
    )
    later = verifier.verify(
        detections_a(),
        frame_index=2,
        proposed_step="S3",
        proposal_confidence=0.8,
        image_shape=(120, 160),
    )

    assert prerequisite.state == "contradicted"
    assert later.admissible is True
    assert later.memory_ready is True
    assert later.state == "supported"


def test_next_step_response_uses_verified_missing_roles() -> None:
    kb = KnowledgeBase(
        {
            "workflow": [
                {"id": "S1", "requires": {}},
                {"id": "S2", "requires": {}},
            ]
        }
    )
    snapshot = AssistantSnapshot(
        frame_index=4,
        step_id="S1",
        step_confidence=0.8,
        runner_up="S2",
        visible_objects=[],
        relevant_objects=[],
        object_counts={},
        scene_relations=[],
        memory_step="S1",
        memory_confidence=0.8,
        proposed_step="S2",
        active_claim="small_gear_inserted",
        claim_state="insufficient",
        missing_evidence_roles=["identity_disambiguation", "slot_relation"],
    )
    query = ResolvedAssistantQuery(text="What is next?", intent="next_step")
    evidence = select_evidence(query, snapshot, kb)
    answer = render_answer(query, evidence, snapshot)

    assert "cannot verify" in answer
    assert "identity evidence" in answer
    assert "slot relation" in answer


def test_next_step_response_requests_new_view_after_internal_compute_stalls() -> None:
    kb = KnowledgeBase(
        {
            "workflow": [
                {"id": "S1", "requires": {}},
                {"id": "S2", "requires": {}},
            ]
        }
    )
    snapshot = AssistantSnapshot(
        frame_index=8,
        step_id="S1",
        step_confidence=0.8,
        runner_up="S2",
        visible_objects=[],
        relevant_objects=[],
        object_counts={},
        scene_relations=[],
        memory_step="S1",
        memory_confidence=0.8,
        proposed_step="S2",
        active_claim="small_gear_inserted",
        claim_state="insufficient",
        missing_evidence_roles=["slot_relation"],
        acquisition_mode="external_observation",
        external_observation_recommended=True,
    )
    query = ResolvedAssistantQuery(text="What is next?", intent="next_step")
    evidence = select_evidence(query, snapshot, kb)
    answer = render_answer(query, evidence, snapshot)

    assert "different viewpoint" in answer
    assert "slot relation" in answer
