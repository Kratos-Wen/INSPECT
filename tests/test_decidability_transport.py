from __future__ import annotations

import json

import pytest

from inspect_system.active_view.decidability_transport import (
    StructuredDecidabilitySelector,
    TypedTraceBackoffRevealModel,
    active_clauses,
    claim_decidability,
    load_typed_requirement_counts,
    mix_typed_requirement_weights,
    parent_claim_id,
    typed_claim_id,
)
from inspect_system.active_view.evidence_state import EvidenceItem, EvidenceState
from inspect_system.active_view.view_lattice import ORBIT_ACTIONS


class PeakedReveal:
    alpha = 2.0
    metadata = {"uses_robot_view_training": False}

    def action_distribution(self, claim_id, evidence_role, context=None):
        del claim_id, evidence_role, context
        return {action: float(action == "orbit_right") for action in ORBIT_ACTIONS}

    def transport_confidence(self, claim_id, evidence_role, context=None):
        del claim_id, evidence_role, context
        return 1.0

    def counterfactual_relevance(self, claim_id, evidence_role, context=None):
        del claim_id, evidence_role, context
        return 1.0

    def candidate_affordance_factor(self, *args, **kwargs):
        del args, kwargs
        return 1.0


class UniformReveal(PeakedReveal):
    def action_distribution(self, claim_id, evidence_role, context=None):
        del claim_id, evidence_role, context
        return {action: 1.0 / len(ORBIT_ACTIONS) for action in ORBIT_ACTIONS}


def identity_state(score: float = 0.0) -> EvidenceState:
    return EvidenceState(
        claim_id="step2_small_gear_inserted",
        items=[
            EvidenceItem(
                name="identity_disambiguation_view",
                evidence_role="identity_disambiguation_view",
                score=score,
                threshold=1.0,
                importance=1.0,
            )
        ],
        counterfactual_family="wrong_identity",
    )


def test_typed_claim_preserves_gear_size_with_generic_parent() -> None:
    assert typed_claim_id("step2_small_gear_inserted") == "small_gear_inserted"
    assert typed_claim_id("step3_big_gear_inserted") == "big_gear_inserted"
    assert parent_claim_id("small_gear_inserted") == "gear_inserted"
    assert parent_claim_id("big_gear_inserted") == "gear_inserted"


def test_counterfactual_selects_only_relevant_clause() -> None:
    state = identity_state(score=0.5)
    clauses = active_clauses(state)
    assert [clause.name for clause in clauses] == ["identity"]
    score, values = claim_decidability(
        state, {"identity_disambiguation_view": 0.5}
    )
    assert score == pytest.approx(0.5)
    assert values == {"identity": pytest.approx(0.5)}


def test_selector_moves_for_directional_reveal_and_stays_when_uninformative() -> None:
    move = StructuredDecidabilitySelector(
        model=PeakedReveal(), lambda_cost=0.0, tau_view=0.0
    ).select(current_view="V1", evidence_state=identity_state())
    assert move.action == "move"
    assert move.selected_view == "V2"
    assert move.reason == "structured_decidability_gain"

    stay = StructuredDecidabilitySelector(
        model=UniformReveal(), lambda_cost=0.0, tau_view=0.0
    ).select(current_view="V1", evidence_state=identity_state())
    assert stay.action == "defer"
    assert stay.selected_view == "V1"
    assert stay.reason == "no_positive_decidability_gain"


def test_typed_trace_posterior_shrinks_to_generic_policy(tmp_path) -> None:
    report = {
        "uses_robot_view_training": False,
        "trainable_events": [
            {
                "claim_id": "gear_inserted",
                "evidence_role": "identity_disambiguation_view",
                "relative_action": "orbit_left",
                "label": 1,
                "transferable": True,
                "transfer_weight": 1.0,
                "label_confidence": 1.0,
                "evidence_stability": 1.0,
                "evidence_importance": 1.0,
                "metadata": {"source_claim_id": "small_gear_inserted"},
                "video": "small.mp4",
            }
        ],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    model = TypedTraceBackoffRevealModel.from_report(PeakedReveal(), path)
    small = model.action_distribution(
        "small_gear_inserted", "identity_disambiguation_view"
    )
    big = model.action_distribution(
        "big_gear_inserted", "identity_disambiguation_view"
    )
    assert small["orbit_left"] > big["orbit_left"]
    assert big["orbit_right"] == pytest.approx(1.0)
    assert model.metadata["uses_robot_utility_labels"] is False


def test_typed_requirement_is_episode_balanced_and_not_missing_mass_matched(
    tmp_path,
) -> None:
    report = {
        "uses_robot_view_training": False,
        "requirement_events": [
            {
                "claim_id": "gear_inserted",
                "evidence_role": "identity_disambiguation_view",
                "requirement_weight": 3.0,
                "video": "small.mp4",
                "after_frame": 10,
                "event_id": "small_gear_event",
            },
            {
                "claim_id": "gear_inserted",
                "evidence_role": "slot_relation_view",
                "requirement_weight": 1.0,
                "video": "small.mp4",
                "after_frame": 10,
                "event_id": "small_gear_event_2",
            },
        ],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    typed = load_typed_requirement_counts(path)
    assert sum(typed["small_gear_inserted"].values()) == pytest.approx(1.0)

    state = EvidenceState(
        claim_id="step2_small_gear_inserted",
        items=[
            EvidenceItem(
                "identity_disambiguation_view",
                score=0.0,
                threshold=1.0,
                importance=1.0,
                evidence_role="identity_disambiguation_view",
            ),
            EvidenceItem(
                "slot_relation_view",
                score=1.0,
                threshold=1.0,
                importance=1.0,
                evidence_role="slot_relation_view",
            ),
        ],
    )
    mixed = mix_typed_requirement_weights(
        state,
        parent_counts={},
        typed_counts=typed,
        beta=0.25,
        temperature=1.0,
    )
    assert mixed.claim_id == "small_gear_inserted"
    assert mixed.items[0].importance > mixed.items[1].importance
