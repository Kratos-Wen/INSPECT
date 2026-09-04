"""Build a support-risk/coverage curve from frozen out-of-fold predictions.

The diagnostic only withholds existing support commitments as insufficient;
it never creates a new support prediction.  Consequently each curve measures
the selective behavior of an already frozen verifier rather than fitting a
new decision rule to evaluation labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def support_score(row: Mapping[str, str]) -> float:
    alternative = max(
        float(row.get("contradiction_probability", 0.0) or 0.0),
        float(row.get("counterfactual_support_probability", 0.0) or 0.0),
    )
    return float(row.get("support_probability", 0.0) or 0.0) - alternative


def summarize(rows: Sequence[Mapping[str, str]], cutoff: float) -> dict[str, Any]:
    confusion: Counter[tuple[str, str]] = Counter()
    support_commits = 0
    support_correct = 0
    for row in rows:
        truth = str(row["truth"])
        prediction = str(row["prediction"])
        if prediction == "supported" and support_score(row) < cutoff:
            prediction = "unresolved"
        confusion[(truth, prediction)] += 1
        if prediction == "supported":
            support_commits += 1
            support_correct += int(truth == "supported")

    supported_total = sum(count for (truth, _), count in confusion.items() if truth == "supported")
    non_supported_total = len(rows) - supported_total
    false_support = support_commits - support_correct
    recalls = []
    for label in ("supported", "contradicted", "unresolved"):
        total = sum(count for (truth, _), count in confusion.items() if truth == label)
        tp = confusion[(label, label)]
        predicted = sum(count for (_, pred), count in confusion.items() if pred == label)
        recall = tp / total if total else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(f1)
    return {
        "cutoff": cutoff,
        "support_commit_coverage": support_commits / len(rows) if rows else 0.0,
        "support_recall": support_correct / supported_total if supported_total else 0.0,
        "support_precision": support_correct / support_commits if support_commits else 1.0,
        "support_risk": false_support / support_commits if support_commits else 0.0,
        "false_support_rate_on_non_supported": false_support / non_supported_total if non_supported_total else 0.0,
        "triage_macro_f1": sum(recalls) / len(recalls),
        "support_commits": support_commits,
        "false_supports": false_support,
    }


def normalized_aurc(points: Sequence[Mapping[str, Any]]) -> float:
    unique: dict[float, float] = {}
    for point in points:
        recall = float(point["support_recall"])
        risk = float(point["support_risk"])
        unique[recall] = min(risk, unique.get(recall, math.inf))
    ordered = sorted(unique.items())
    max_recall = ordered[-1][0] if ordered else 0.0
    if max_recall <= 0.0:
        return 0.0
    area = 0.0
    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        area += 0.5 * (y0 + y1) * (x1 - x0)
    return area / max_recall


def curve(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    support_scores = sorted(
        {support_score(row) for row in rows if str(row["prediction"]) == "supported"},
        reverse=True,
    )
    cutoffs = [math.inf, *support_scores, -math.inf]
    points = [summarize(rows, cutoff) for cutoff in cutoffs]
    deduplicated: dict[tuple[int, int], dict[str, Any]] = {}
    for point in points:
        key = (int(point["support_commits"]), int(point["false_supports"]))
        deduplicated[key] = point
    return sorted(
        deduplicated.values(),
        key=lambda point: (float(point["support_recall"]), float(point["support_risk"])),
    )


def matched_point(points: Sequence[Mapping[str, Any]], target: float) -> Mapping[str, Any] | None:
    eligible = [point for point in points if float(point["support_recall"]) >= target]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda point: (
            float(point["support_risk"]),
            float(point["support_recall"]),
            -float(point["support_precision"]),
        ),
    )


def main() -> None:
    args = parse_args()
    rows = read_rows(args.predictions_csv)
    payload: dict[str, Any] = {
        "protocol": {
            "source": str(args.predictions_csv),
            "predictions": "frozen outer-fold out-of-fold outputs",
            "curve_operation": "withhold low-margin support commitments as insufficient only",
            "evaluation_labels_used_for_model_or_threshold_fitting": False,
        },
        "variants": {},
    }
    flat_rows = []
    for variant in args.variants:
        selected = [row for row in rows if str(row.get("variant")) == variant]
        if not selected:
            raise ValueError(f"No rows found for variant: {variant}")
        points = curve(selected)
        original = summarize(selected, -math.inf)
        matched = {
            f"support_recall_{target:.2f}": matched_point(points, target)
            for target in (0.10, 0.20, 0.30)
        }
        payload["variants"][variant] = {
            "samples": len(selected),
            "original_operating_point": original,
            "normalized_support_aurc": normalized_aurc(points),
            "matched_recall": matched,
        }
        flat_rows.extend({"variant": variant, **point} for point in points)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
