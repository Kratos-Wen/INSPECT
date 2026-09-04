from __future__ import annotations

import math

from inspect_system.active_view.evidence_state import EvidenceItem
from inspect_system.active_view.reveal_model import (
    FeatureConditionedRevealModel,
    PriorTableRevealModel,
)
from inspect_system.active_view.six_view_projector import score_candidate_view
from inspect_system.active_view.view_lattice import ORBIT_ACTIONS


def test_action_distribution_is_normalized() -> None:
    model = PriorTableRevealModel(
        alpha=2.0,
        default_probability=1.0 / len(ORBIT_ACTIONS),
    )
    model.counts = {
        model._key("gear_inserted", "identity", "orbit_left"): {
            "pos": 3.0,
            "total": 3.0,
        }
    }

    distribution = model.action_distribution("gear_inserted", "identity")

    assert math.isclose(sum(distribution.values()), 1.0)
    assert distribution["orbit_left"] > distribution["orbit_right"]


def test_feature_context_is_used_by_distribution() -> None:
    model = FeatureConditionedRevealModel(
        alpha=1.0,
        default_probability=1.0 / len(ORBIT_ACTIONS),
    )
    model.feature_counts = {
        model._feature_key("gear_inserted", "identity", "orbit_up", "low"): {
            "pos": 4.0,
            "total": 4.0,
        }
    }

    low = model.action_distribution(
        "gear_inserted",
        "identity",
        context={"item_score": 0.1},
    )
    high = model.action_distribution(
        "gear_inserted",
        "identity",
        context={"item_score": 0.9},
    )

    assert low["orbit_up"] > high["orbit_up"]
    assert math.isclose(sum(low.values()), 1.0)
    assert math.isclose(sum(high.values()), 1.0)


def test_projected_gain_is_bounded_by_missing_evidence_weight() -> None:
    model = PriorTableRevealModel(
        alpha=2.0,
        default_probability=1.0 / len(ORBIT_ACTIONS),
    )
    model.counts = {
        model._key("gear_inserted", "identity", "orbit_left"): {
            "pos": 3.0,
            "total": 3.0,
        }
    }
    item = EvidenceItem(
        name="identity",
        evidence_role="identity",
        score=0.0,
        threshold=1.0,
        importance=1.0,
    )

    score = score_candidate_view(
        model=model,
        claim_id="gear_inserted",
        missing_evidence=[item],
        current_view="V1",
        candidate_view="V0",
        lambda_cost=0.0,
    )

    assert 0.0 <= score.evidence_gain <= 1.0
