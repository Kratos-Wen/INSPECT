from __future__ import annotations

from inspect_system.active_view.active_selector import ActiveSelector
from inspect_system.active_view.evidence_state import EvidenceItem, EvidenceState
from inspect_system.active_view.reveal_model import PriorTableRevealModel


def test_selector_skips_previously_visited_candidate() -> None:
    selector = ActiveSelector(
        model=PriorTableRevealModel(alpha=1.0, default_probability=0.125),
        lambda_cost=0.0,
        tau_view=-1.0,
    )
    state = EvidenceState(
        claim_id="gear_inserted",
        counterfactual_family="identity",
        items=[
            EvidenceItem(
                "identity",
                score=0.0,
                threshold=0.65,
                importance=1.0,
                evidence_role="identity_disambiguation_view",
            )
        ],
    )

    first = selector.select(current_view="V0", evidence_state=state)
    second = selector.select(
        current_view="V0",
        evidence_state=state,
        visited_views=[first.selected_view],
    )

    assert first.action == "move"
    assert second.action == "move"
    assert second.selected_view != first.selected_view


class _RecordingRevealModel(PriorTableRevealModel):
    def __init__(self) -> None:
        super().__init__(alpha=1.0, default_probability=0.125)
        self.families: list[str] = []

    def counterfactual_relevance(self, claim_id, evidence_role, context=None):
        del claim_id, evidence_role
        self.families.append(str((context or {}).get("counterfactual_family", "")))
        return 1.0


def test_selector_does_not_hard_gate_zero_mass_counterfactual_scores() -> None:
    model = _RecordingRevealModel()
    selector = ActiveSelector(model=model, lambda_cost=0.0, tau_view=-1.0)
    state = EvidenceState(
        claim_id="gear_inserted",
        counterfactual_family="",
        counterfactual_scores={"identity": 0.0, "spatial_relation": 0.0},
        items=[
            EvidenceItem(
                "identity",
                score=0.0,
                threshold=0.65,
                importance=1.0,
                evidence_role="identity_disambiguation_view",
            ),
            EvidenceItem(
                "slot",
                score=0.0,
                threshold=0.65,
                importance=1.0,
                evidence_role="slot_relation_view",
            ),
        ],
    )

    result = selector.select(current_view="V0", evidence_state=state)

    assert result.action == "move"
    assert model.families
    assert set(model.families) == {""}
