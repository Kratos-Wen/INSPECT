"""Compare Qwen3-VL and INSPECT on the exact same balanced frame subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping


OUTCOMES = ("supported", "contradicted", "unresolved")
ADOPTED_VARIANT = "causal_visual_appearance_no_consistency"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        PureWindowsPath(str(row["video"])).name,
        str(row["frame"]),
        str(row["truth"]),
    )


def summarize_predictions(truths: Iterable[str], predictions: Iterable[str]) -> dict[str, Any]:
    truth_list = list(truths)
    pred_list = list(predictions)
    if len(truth_list) != len(pred_list):
        raise ValueError("Truth and prediction lengths differ")
    confusion = Counter(zip(truth_list, pred_list))
    truth_counts = Counter(truth_list)
    class_metrics = {}
    f1_values = []
    for label in OUTCOMES:
        tp = confusion[(label, label)]
        fp = sum(confusion[(other, label)] for other in OUTCOMES if other != label)
        fn = truth_counts[label] - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / truth_counts[label] if truth_counts[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_metrics[label] = {"precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
    total = len(truth_list)
    non_supported = truth_counts["contradicted"] + truth_counts["unresolved"]
    false_support = (
        confusion[("contradicted", "supported")]
        + confusion[("unresolved", "supported")]
    ) / non_supported
    return {
        "samples": total,
        "accuracy": sum(confusion[(label, label)] for label in OUTCOMES) / total,
        "macro_f1": sum(f1_values) / len(f1_values),
        "supported_recall": class_metrics["supported"]["recall"],
        "contradiction_recall": class_metrics["contradicted"]["recall"],
        "insufficient_recall": class_metrics["unresolved"]["recall"],
        "false_support": false_support,
        "class_metrics": class_metrics,
        "confusion": {
            f"{truth}->{pred}": confusion[(truth, pred)]
            for truth in OUTCOMES
            for pred in OUTCOMES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-rows", type=Path, required=True)
    parser.add_argument("--qwen-direct-rows", type=Path, required=True)
    parser.add_argument("--qwen-context-rows", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    direct_rows = read_csv(args.qwen_direct_rows)
    context_rows = read_csv(args.qwen_context_rows)
    inspect_rows = [
        row for row in read_csv(args.inspect_rows) if row.get("variant") == ADOPTED_VARIANT
    ]
    if len(direct_rows) != 60 or len(context_rows) != 60:
        raise ValueError("Expected 60 rows for each VLM protocol")
    direct = {key(row): row for row in direct_rows}
    context = {key(row): row for row in context_rows}
    inspect = {key(row): row for row in inspect_rows}
    if set(direct) != set(context):
        raise ValueError("Direct and context VLM subsets differ")
    missing = sorted(set(direct) - set(inspect))
    if missing:
        raise ValueError(f"INSPECT is missing {len(missing)} VLM comparison rows")
    truth_counts = Counter(row["truth"] for row in direct_rows)
    if truth_counts != Counter({label: 20 for label in OUTCOMES}):
        raise ValueError(f"Expected 20 samples per class, got {truth_counts}")

    comparison_rows = []
    for sample_key in sorted(direct):
        direct_row = direct[sample_key]
        context_row = context[sample_key]
        inspect_row = inspect[sample_key]
        comparison_rows.append(
            {
                "video": sample_key[0],
                "frame": sample_key[1],
                "claim_id": direct_row["claim_id"],
                "truth": sample_key[2],
                "qwen_direct": direct_row["pred"],
                "qwen_context": context_row["pred"],
                "inspect": inspect_row["prediction"],
                "inspect_outer_fold": inspect_row["outer_fold"],
            }
        )
    truths = [row["truth"] for row in comparison_rows]
    results = {
        "Qwen3-VL Direct": summarize_predictions(
            truths, [row["qwen_direct"] for row in comparison_rows]
        ),
        "Qwen3-VL + Procedure Context": summarize_predictions(
            truths, [row["qwen_context"] for row in comparison_rows]
        ),
        "INSPECT": summarize_predictions(
            truths, [row["inspect"] for row in comparison_rows]
        ),
    }
    report = {
        "schema": "inspect.vlm-common-subset-comparison.v1",
        "protocol": {
            "samples": 60,
            "per_class": 20,
            "same_video_frame_truth_keys": True,
            "qwen_prompt_version": "triage_v2",
            "inspect_variant": ADOPTED_VARIANT,
            "candidate_or_ground_truth_inputs_to_models": False,
            "ground_truth_used_for_balanced_sampling_and_metrics_only": True,
        },
        "input_sha256": {
            "inspect_rows": sha256(args.inspect_rows),
            "qwen_direct_rows": sha256(args.qwen_direct_rows),
            "qwen_context_rows": sha256(args.qwen_context_rows),
        },
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
