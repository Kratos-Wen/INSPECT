from __future__ import annotations

import pytest

from inspect_system.active_view.evidence_state import EvidenceItem, EvidenceState
from inspect_system.active_view.requirement_model import (
    blend_session_requirement_weights,
    mix_requirement_weights,
)
from inspect_system.active_view.reveal_model import PriorTableRevealModel
from inspect_system.active_view.six_view_projector import score_candidate_view


def test_requirement_weights_preserve_pi_only_missing_mass() -> None:
    state = EvidenceState(
        claim_id="gear_inserted",
        items=[
            EvidenceItem(
                "identity",
                score=0.1,
                threshold=0.65,
                evidence_role="identity_disambiguation_view",
            ),
            EvidenceItem(
                "insertion",
                score=0.2,
                threshold=0.65,
                evidence_role="insertion_verification_view",
            ),
            EvidenceItem(
                "slot", score=0.9, threshold=0.65, evidence_role="slot_relation_view"
            ),
        ],
    )
    learned = {
        "gear_inserted": {
            "identity_disambiguation_view": 10.0,
            "insertion_verification_view": 1.0,
            "slot_relation_view": 1.0,
        }
    }

    pi_only = mix_requirement_weights(
        state,
        learned,
        beta=0.25,
        strength=0.0,
    )
    calibrated = mix_requirement_weights(
        state,
        learned,
        beta=0.25,
        strength=1.0,
        temperature=0.0,
    )

    assert [item.importance for item in pi_only.items] == pytest.approx([1 / 3] * 3)
    assert sum(item.importance for item in calibrated.missing()) == pytest.approx(2 / 3)
    assert sum(item.importance for item in calibrated.observed()) == pytest.approx(
        1 / 3
    )
    assert calibrated.items[0].importance > calibrated.items[1].importance


def test_explicit_counterfactual_excludes_nonseparating_roles() -> None:
    model = PriorTableRevealModel(alpha=1.0, default_probability=0.125)
    identity = EvidenceItem(
        "identity",
        score=0.0,
        threshold=0.65,
        importance=0.5,
        evidence_role="identity_disambiguation_view",
    )
    insertion = EvidenceItem(
        "insertion",
        score=0.0,
        threshold=0.65,
        importance=0.5,
        evidence_role="insertion_verification_view",
    )
    context = {"counterfactual_family": "identity"}
    normalized_identity = EvidenceItem(
        "identity",
        score=0.0,
        threshold=0.65,
        importance=1.0,
        evidence_role="identity_disambiguation_view",
    )

    identity_only = score_candidate_view(
        model=model,
        claim_id="gear_inserted",
        missing_evidence=[normalized_identity],
        current_view="V5",
        candidate_view="V4",
        counterfactual_context=context,
    )
    with_irrelevant_role = score_candidate_view(
        model=model,
        claim_id="gear_inserted",
        missing_evidence=[identity, insertion],
        current_view="V5",
        candidate_view="V4",
        counterfactual_context=context,
    )

    assert with_irrelevant_role.evidence_gain == pytest.approx(
        identity_only.evidence_gain
    )


def test_session_requirement_blend_preserves_missing_mass_and_changes_ranking() -> None:
    state = EvidenceState(
        claim_id="gear_inserted",
        items=[
            EvidenceItem(
                "identity",
                score=0.1,
                threshold=0.65,
                importance=0.30,
                evidence_role="identity_disambiguation_view",
            ),
            EvidenceItem(
                "insertion",
                score=0.2,
                threshold=0.65,
                importance=0.40,
                evidence_role="insertion_verification_view",
            ),
            EvidenceItem(
                "slot",
                score=0.9,
                threshold=0.65,
                importance=0.30,
                evidence_role="slot_relation_view",
            ),
        ],
    )

    blended = blend_session_requirement_weights(
        state,
        {"identity_disambiguation_view": 1.0},
        blend=1.0,
        prior_strength=0.05,
    )

    assert sum(item.importance for item in blended.missing()) == pytest.approx(0.70)
    assert sum(item.importance for item in blended.observed()) == pytest.approx(0.30)
    assert blended.items[0].importance > blended.items[1].importance

def test_counterfactual_conditioned_requirement_overrides_claim_marginal() -> None:
    state = EvidenceState(
        claim_id="gear_inserted",
        counterfactual_family="spatial_relation",
        items=[
            EvidenceItem(
                "identity",
                score=0.0,
                threshold=0.65,
                evidence_role="identity_disambiguation_view",
            ),
            EvidenceItem(
                "insertion",
                score=0.0,
                threshold=0.65,
                evidence_role="insertion_verification_view",
            ),
        ],
    )
    learned = {
        "gear_inserted": {
            "identity_disambiguation_view": 10.0,
            "insertion_verification_view": 1.0,
        },
        "gear_inserted|cf:spatial_relation": {
            "identity_disambiguation_view": 0.0,
            "insertion_verification_view": 5.0,
        },
    }

    calibrated = mix_requirement_weights(
        state,
        learned,
        beta=0.25,
        strength=1.0,
        temperature=0.0,
    )

    assert calibrated.items[1].importance > calibrated.items[0].importance
