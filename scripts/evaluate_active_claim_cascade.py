"""Evaluate the proposal-to-verification cascade on annotated active claims.

The proposal cache must contain held-out, cross-fitted predictions.  A sample is
counted as an end-to-end success only when the proposal contains the annotated
active step and the verifier predicts the annotated claim state correctly.
No label is imputed for claims belonging to an incorrectly proposed step.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


STATE_ALIASES = {
    "supported": "supported",
    "support": "supported",
    "confirmed": "supported",
    "contradicted": "contradicted",
    "contradiction": "contradicted",
    "rejected": "contradicted",
    "unresolved": "insufficient",
    "insufficient": "insufficient",
    "cannot_tell": "insufficient",
}


def video_key(value: str) -> str:
    return Path(str(value)).name.lower()


def normalize_state(value: Any) -> str:
    return STATE_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())


def normalize_step(value: Any) -> str:
    text = str(value).strip().upper()
    if text.startswith("STEP"):
        text = "S" + text[4:].strip(" _-")
    return text


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def load_timeline(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in read_csv(path):
        if str(row.get("skip", "0")).strip().lower() in {"1", "true", "yes"}:
            continue
        record = dict(row)
        record["start_frame"] = int(float(row.get("start_frame", 0) or 0))
        record["end_frame"] = int(float(row.get("end_frame", 0) or 0))
        grouped.setdefault(video_key(row.get("video", "")), []).append(record)
    for rows in grouped.values():
        rows.sort(key=lambda item: (item["start_frame"], item["end_frame"]))
    return grouped


def timeline_at(
    timeline: Mapping[str, list[dict[str, Any]]],
    video: str,
    frame: int,
) -> dict[str, Any] | None:
    for row in timeline.get(video_key(video), []):
        if int(row["start_frame"]) <= frame <= int(row["end_frame"]):
            return row
    return None


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    by_truth = Counter(str(row["truth"]) for row in rows)
    top1 = sum(bool(row["proposal_top1_correct"]) for row in rows)
    top2 = sum(bool(row["proposal_top2_contains_active"]) for row in rows)
    verifier = sum(bool(row["verifier_correct"]) for row in rows)
    joint1 = sum(bool(row["joint_top1_correct"]) for row in rows)
    joint2 = sum(bool(row["joint_top2_correct"]) for row in rows)

    supported = [row for row in rows if row["truth"] == "supported"]
    non_supported = [row for row in rows if row["truth"] != "supported"]
    supported_joint1 = sum(bool(row["joint_top1_supported"]) for row in supported)
    supported_joint2 = sum(bool(row["joint_top2_supported"]) for row in supported)
    false_support_given_top1 = sum(
        bool(row["proposal_top1_correct"] and row["prediction"] == "supported")
        for row in non_supported
    )
    false_support_given_top2 = sum(
        bool(row["proposal_top2_contains_active"] and row["prediction"] == "supported")
        for row in non_supported
    )
    top1_non_supported = sum(bool(row["proposal_top1_correct"]) for row in non_supported)
    top2_non_supported = sum(bool(row["proposal_top2_contains_active"]) for row in non_supported)

    per_state: dict[str, Any] = {}
    for state in ("supported", "contradicted", "insufficient"):
        subset = [row for row in rows if row["truth"] == state]
        per_state[state] = {
            "samples": len(subset),
            "proposal_top1_recall": safe_ratio(
                sum(bool(row["proposal_top1_correct"]) for row in subset), len(subset)
            ),
            "proposal_top2_recall": safe_ratio(
                sum(bool(row["proposal_top2_contains_active"]) for row in subset), len(subset)
            ),
            "oracle_claim_verifier_recall": safe_ratio(
                sum(bool(row["verifier_correct"]) for row in subset), len(subset)
            ),
            "joint_top1_recall": safe_ratio(
                sum(bool(row["joint_top1_correct"]) for row in subset), len(subset)
            ),
            "joint_top2_recall": safe_ratio(
                sum(bool(row["joint_top2_correct"]) for row in subset), len(subset)
            ),
        }

    return {
        "samples": total,
        "videos": len({row["video"] for row in rows}),
        "truth_distribution": dict(sorted(by_truth.items())),
        "proposal_top1_active_step_recall": safe_ratio(top1, total),
        "proposal_top2_active_step_recall": safe_ratio(top2, total),
        "oracle_active_claim_verifier_accuracy": safe_ratio(verifier, total),
        "joint_top1_triage_accuracy": safe_ratio(joint1, total),
        "joint_top2_triage_accuracy": safe_ratio(joint2, total),
        "joint_top1_supported_recall": safe_ratio(supported_joint1, len(supported)),
        "joint_top2_supported_recall": safe_ratio(supported_joint2, len(supported)),
        "false_support_given_correct_top1_proposal": safe_ratio(
            false_support_given_top1, top1_non_supported
        ),
        "false_support_given_top2_contains_active": safe_ratio(
            false_support_given_top2, top2_non_supported
        ),
        "per_state": per_state,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier-csv", type=Path, required=True)
    parser.add_argument("--proposal-jsonl", type=Path, required=True)
    parser.add_argument("--timeline-csv", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    proposal_lookup = {
        (video_key(row.get("video", "")), int(row.get("frame", -1))): row
        for row in read_jsonl(args.proposal_jsonl)
    }
    timeline = load_timeline(args.timeline_csv)
    joined: list[dict[str, Any]] = []
    missing_proposal = 0
    missing_timeline = 0
    truth_mismatch = 0

    for row in read_csv(args.verifier_csv):
        if str(row.get("variant", "")) != args.variant:
            continue
        video = video_key(row.get("video", ""))
        frame = int(float(row.get("frame", -1) or -1))
        proposal = proposal_lookup.get((video, frame))
        if proposal is None:
            missing_proposal += 1
            continue
        target = timeline_at(timeline, video, frame)
        if target is None:
            missing_timeline += 1
            continue
        active_step = normalize_step(target.get("step_id"))
        truth = normalize_state(row.get("truth"))
        timeline_truth = normalize_state(target.get("outcome"))
        if truth != timeline_truth:
            truth_mismatch += 1
            continue
        prediction = normalize_state(row.get("prediction"))
        scores = {
            normalize_step(step): float(score)
            for step, score in dict(proposal.get("scores") or {}).items()
        }
        ranking = sorted(scores, key=lambda step: (-scores[step], step))
        proposed_step = normalize_step(proposal.get("proposed_step"))
        top1_correct = proposed_step == active_step
        top2_contains = active_step in ranking[:2]
        verifier_correct = prediction == truth
        joined.append(
            {
                "video": video,
                "frame": frame,
                "active_step": active_step,
                "truth": truth,
                "proposed_step": proposed_step,
                "runner_up": normalize_step(proposal.get("runner_up")),
                "proposal_confidence": float(proposal.get("confidence", 0.0) or 0.0),
                "prediction": prediction,
                "support_probability": float(row.get("support_probability", 0.0) or 0.0),
                "contradiction_probability": float(
                    row.get("contradiction_probability", 0.0) or 0.0
                ),
                "insufficient_probability": float(
                    row.get("insufficient_probability", 0.0) or 0.0
                ),
                "proposal_top1_correct": top1_correct,
                "proposal_top2_contains_active": top2_contains,
                "verifier_correct": verifier_correct,
                "joint_top1_correct": top1_correct and verifier_correct,
                "joint_top2_correct": top2_contains and verifier_correct,
                "joint_top1_supported": (
                    top1_correct and truth == "supported" and prediction == "supported"
                ),
                "joint_top2_supported": (
                    top2_contains and truth == "supported" and prediction == "supported"
                ),
            }
        )

    if not joined:
        raise RuntimeError("No verifier rows joined to proposal and timeline records")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)

    payload = {
        "protocol": {
            "definition": (
                "A joint success requires a held-out proposal to contain the annotated "
                "active step and the verifier to predict that active claim's annotated state."
            ),
            "incorrect_proposal_claim_labels_imputed": False,
            "proposal_predictions_cross_fitted": True,
            "candidate_claim_ground_truth_used_by_pipeline": False,
            "verifier_variant": args.variant,
            "missing_proposal_rows": missing_proposal,
            "missing_timeline_rows": missing_timeline,
            "truth_mismatch_rows": truth_mismatch,
        },
        "metrics": summarize(joined),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
