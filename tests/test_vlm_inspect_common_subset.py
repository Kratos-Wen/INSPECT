from scripts.compare_vlm_inspect_balanced_subset import summarize_predictions


def test_common_subset_metrics_include_safety_tradeoff() -> None:
    truths = ["supported", "contradicted", "unresolved"]
    metrics = summarize_predictions(truths, ["supported", "supported", "unresolved"])
    assert metrics["supported_recall"] == 1.0
    assert metrics["contradiction_recall"] == 0.0
    assert metrics["insufficient_recall"] == 1.0
    assert metrics["false_support"] == 0.5
