import csv
import json

import pytest
import numpy as np

from scripts.evaluate_relation_aware_claim_verifier import (
    HierarchicalTriageClassifier,
    OneVsRestTriageClassifier,
    Sample,
    audit_recorded_proposal_provenance,
    causal_filter_probabilities,
    feature_vector,
    support_confirmation_filter,
)


def _summary(tmp_path, feedback_rows):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "feedback.jsonl").open("w", encoding="utf-8") as handle:
        for row in feedback_rows:
            handle.write(json.dumps(row) + chr(10))
    summary = tmp_path / "summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["returncode", "run_dir"])
        writer.writeheader()
        writer.writerow({"returncode": 0, "run_dir": str(run_dir)})
    return summary


def test_recorded_proposals_reject_timeline_feedback(tmp_path) -> None:
    summary = _summary(
        tmp_path,
        [
            {
                "frame_index": 12,
                "feedback": {
                    "source": "gt_timeline",
                    "extras": {"simulated_feedback": True},
                },
            }
        ],
    )
    with pytest.raises(RuntimeError, match="label-contaminated"):
        audit_recorded_proposal_provenance(summary, ["relations"], None)


def test_detection_only_does_not_consume_proposal_feedback(tmp_path) -> None:
    summary = _summary(
        tmp_path,
        [{"feedback": {"source": "gt_timeline"}}],
    )
    report = audit_recorded_proposal_provenance(summary, ["detection"], None)
    assert report["uses_recorded_proposals"] is False


def test_held_out_proposal_cache_bypasses_recorded_fusion(tmp_path) -> None:
    summary = _summary(
        tmp_path,
        [{"feedback": {"source": "gt_timeline"}}],
    )
    report = audit_recorded_proposal_provenance(
        summary,
        ["relations"],
        tmp_path / "held_out_predictions.jsonl",
    )
    assert report["uses_recorded_proposals"] is False


def test_visual_appearance_variant_excludes_proposal_features() -> None:
    sample = Sample(
        video="video.mp4",
        frame=3,
        truth="supported",
        proposal_consistent=False,
        detection=(1.0,),
        relation=(2.0,),
        proposal=(99.0,),
        causal=(),
        appearance=(3.0,),
        counterfactual_detection=(4.0,),
        counterfactual_relation=(5.0,),
        has_specialized_counterfactual=True,
    )
    assert feature_vector(
        sample, "visual_appearance_no_consistency"
    ).tolist() == [1.0, 2.0, 3.0]
    assert feature_vector(
        sample, "causal_visual_appearance_no_consistency"
    ).tolist() == [1.0, 2.0, 3.0]
    assert feature_vector(
        sample, "confirm_visual_appearance_no_consistency"
    ).tolist() == [1.0, 2.0, 3.0]
    assert feature_vector(
        sample, "confirm3_visual_appearance_no_consistency"
    ).tolist() == [1.0, 2.0, 3.0]
    assert feature_vector(
        sample, "hierarchical_visual_appearance_no_consistency"
    ).tolist() == [1.0, 2.0, 3.0]
    assert feature_vector(
        sample, "causal_hierarchical_visual_appearance_no_consistency"
    ).tolist() == [1.0, 2.0, 3.0]


def test_hierarchical_triage_probabilities_form_distribution() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(90, 4))
    labels = np.repeat(np.arange(3), 30)
    model = HierarchicalTriageClassifier().fit(features, labels)

    probabilities = model.predict_proba(features[:8])

    assert probabilities.shape == (8, 3)
    assert np.all(probabilities >= 0.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_one_vs_rest_triage_probabilities_form_distribution() -> None:
    rng = np.random.default_rng(9)
    features = rng.normal(size=(90, 4))
    labels = np.repeat(np.arange(3), 30)
    model = OneVsRestTriageClassifier().fit(features, labels)

    probabilities = model.predict_proba(features[:8])

    assert probabilities.shape == (8, 3)
    assert np.all(probabilities >= 0.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_causal_filter_resets_when_active_claim_changes() -> None:
    def sample(frame: int, claim_id: str) -> Sample:
        return Sample(
            video="session.mp4",
            frame=frame,
            truth="supported",
            proposal_consistent=True,
            detection=(),
            relation=(),
            proposal=(),
            causal=(),
            appearance=(),
            counterfactual_detection=(),
            counterfactual_relation=(),
            has_specialized_counterfactual=False,
            claim_id=claim_id,
        )

    probabilities = np.asarray(
        [[0.90, 0.05, 0.05], [0.05, 0.90, 0.05]], dtype=np.float64
    )
    filtered = causal_filter_probabilities(
        probabilities,
        [sample(1, "small_gear_inserted"), sample(2, "big_gear_inserted")],
        decay=0.90,
    )

    assert np.allclose(filtered, probabilities)


def test_support_confirmation_withholds_first_spike_and_uses_only_past() -> None:
    def sample(frame: int) -> Sample:
        return Sample(
            video="session.mp4",
            frame=frame,
            truth="supported",
            proposal_consistent=True,
            detection=(),
            relation=(),
            proposal=(),
            causal=(),
            appearance=(),
            counterfactual_detection=(),
            counterfactual_relation=(),
            has_specialized_counterfactual=False,
            claim_id="small_gear_inserted",
        )

    probabilities = np.asarray(
        [
            [0.90, 0.05, 0.05],
            [0.80, 0.10, 0.10],
            [0.95, 0.03, 0.02],
        ],
        dtype=np.float64,
    )
    filtered = support_confirmation_filter(
        probabilities,
        [sample(1), sample(2), sample(3)],
    )

    assert filtered[0, 0] == 0.0
    assert filtered[1, 0] == pytest.approx(0.80)
    assert filtered[2, 0] == pytest.approx(0.80)

    confirmed_three = support_confirmation_filter(
        probabilities,
        [sample(1), sample(2), sample(3)],
        confirmation_steps=3,
    )
    assert confirmed_three[0, 0] == 0.0
    assert confirmed_three[1, 0] == 0.0
    assert confirmed_three[2, 0] == pytest.approx(0.80)
