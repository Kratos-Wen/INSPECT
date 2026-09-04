"""Evaluate fixed-lattice selection against human binary decidability labels.

The human labels are evaluation-only. This script never exposes them to a
policy and only rescores already frozen selected-view rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


VIEWS = tuple(f"V{index}" for index in range(6))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_labels(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    connection = sqlite3.connect(path)
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for setup_id, view_id, payload, status in connection.execute(
        "SELECT setup_id, view_id, payload, workflow_status FROM annotations"
    ):
        value = json.loads(payload)
        if status != "complete":
            raise ValueError(f"Incomplete annotation: {setup_id}/{view_id}")
        decidable = value.get("claim_decidable", "")
        occluded = value.get("explicit_occlusion", "")
        if decidable not in {"yes", "no"}:
            raise ValueError(f"Invalid decidability: {setup_id}/{view_id}")
        if decidable == "no" and occluded not in {"yes", "no"}:
            raise ValueError(f"Missing occlusion label: {setup_id}/{view_id}")
        rows[(setup_id, view_id)] = {
            "decidable": int(decidable == "yes"),
            "occluded": int(occluded == "yes"),
            "legacy": str(value.get("legacy_human_utility", "")),
        }
    connection.close()
    return rows


def load_manifest(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    setups = {str(item["setup_id"]): item for item in payload["setups"]}
    return setups, list(payload["setups"])


def row_views(row: Mapping[str, Any]) -> tuple[str, str]:
    current = str(row.get('current_view') or row.get('start_view') or '')
    path = [str(value) for value in row.get('path', [])]
    selected = str(row.get('selected_view') or (path[1] if len(path) > 1 else current))
    if not current or not selected:
        raise ValueError('Policy row does not identify current and selected views.')
    return current, selected


def rows_at_stage(
    rows: Sequence[Mapping[str, Any]],
    stage: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        current, selected_at_1 = row_views(row)
        path = [str(value) for value in row.get("path", [])]
        if stage == "at_1":
            selected = selected_at_1
        elif stage == "final":
            selected = path[-1] if path else selected_at_1
        elif stage == "retained":
            selected = str(row.get("retained_view") or (path[-1] if path else selected_at_1))
        else:
            raise ValueError(stage)
        result.append(
            {
                "trial_id": str(row["trial_id"]),
                "current_view": current,
                "selected_view": selected,
            }
        )
    return result


def observed_path_oracle_rows(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluation-only ceiling over views actually visited by the policy."""
    result: list[dict[str, Any]] = []
    for row in rows:
        setup_id = str(row["trial_id"])
        current, selected_at_1 = row_views(row)
        path = [str(value) for value in row.get("path", [])] or [current, selected_at_1]
        selected = max(
            path,
            key=lambda view: (
                int(labels[(setup_id, view)]["decidable"]),
                view == current,
                -path.index(view),
            ),
        )
        result.append(
            {
                "trial_id": setup_id,
                "current_view": current,
                "selected_view": selected,
            }
        )
    return result


def evaluate_rows(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    transitions: list[tuple[int, int, int, int]] = []
    for row in rows:
        setup_id = str(row["trial_id"])
        current_view, selected_view = row_views(row)
        current = int(labels[(setup_id, current_view)]["decidable"])
        selected = int(labels[(setup_id, selected_view)]["decidable"])
        current_occluded = int(labels[(setup_id, current_view)]["occluded"])
        oracle = max(int(labels[(setup_id, view)]["decidable"]) for view in VIEWS)
        transitions.append((current, selected, oracle, current_occluded))

    count = len(transitions)
    current_count = sum(item[0] for item in transitions)
    selected_count = sum(item[1] for item in transitions)
    oracle_count = sum(item[2] for item in transitions)
    opportunities = sum((not item[0]) and item[2] for item in transitions)
    recovered = sum((not item[0]) and item[1] for item in transitions)
    preserved = sum(item[0] and item[1] for item in transitions)
    lost = sum(item[0] and not item[1] for item in transitions)
    occluded_opportunities = sum((not item[0]) and item[3] for item in transitions)
    occluded_recovered = sum(
        (not item[0]) and item[3] and item[1] for item in transitions
    )
    other_opportunities = sum((not item[0]) and not item[3] for item in transitions)
    other_recovered = sum(
        (not item[0]) and not item[3] and item[1] for item in transitions
    )
    oracle_gap = oracle_count - current_count

    def ratio(numerator: float, denominator: float) -> float:
        return float(numerator) / float(denominator) if denominator else 0.0

    return {
        "trials": count,
        "current_decidable": ratio(current_count, count),
        "decidable_at_1": ratio(selected_count, count),
        "binary_gain": ratio(selected_count - current_count, count),
        "recovery_at_1": ratio(recovered, opportunities),
        "decidable_preservation": ratio(preserved, current_count),
        "decidable_loss": ratio(lost, current_count),
        "oracle_achievable": ratio(oracle_count, count),
        "oracle_gap_closed": ratio(selected_count - current_count, oracle_gap),
        "occlusion_recovery": ratio(occluded_recovered, occluded_opportunities),
        "non_occlusion_recovery": ratio(other_recovered, other_opportunities),
        "counts": {
            "current_decidable": current_count,
            "selected_decidable": selected_count,
            "recoverable": opportunities,
            "recovered": recovered,
            "preserved": preserved,
            "lost": lost,
            "occluded_recoverable": occluded_opportunities,
            "occluded_recovered": occluded_recovered,
            "non_occluded_recoverable": other_opportunities,
            "non_occluded_recovered": other_recovered,
        },
    }


def synthetic_rows(
    base_rows: Sequence[Mapping[str, Any]],
    mode: str,
    labels: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for row in base_rows:
        setup_id = str(row["trial_id"])
        current_view, _ = row_views(row)
        if mode == "current":
            candidates = [current_view]
        elif mode == "uniform_non_current":
            candidates = [view for view in VIEWS if view != current_view]
        elif mode == "oracle":
            candidates = [
                max(
                    VIEWS,
                    key=lambda view: (
                        int(labels[(setup_id, view)]["decidable"]),
                        view == current_view,
                        view,
                    ),
                )
            ]
        else:
            raise ValueError(mode)
        result.extend(
            {
                "trial_id": setup_id,
                "current_view": current_view,
                "selected_view": selected_view,
            }
            for selected_view in candidates
        )
    return result


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-db", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy-rows", type=Path, required=True)
    parser.add_argument("--policy-name", default="INSPECT")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    labels = load_labels(args.annotations_db)
    setups, setup_rows = load_manifest(args.manifest)
    if len(labels) != 360 or len(setups) != 60:
        raise ValueError("Expected the complete 60-setup, 360-view project.")

    by_step: dict[str, Counter[str]] = defaultdict(Counter)
    setup_oracles = 0
    fully_decidable_setups = 0
    for setup in setup_rows:
        setup_id = str(setup["setup_id"])
        values = [int(labels[(setup_id, view)]["decidable"]) for view in VIEWS]
        step = str(setup["target_step"])
        by_step[step]["views"] += len(values)
        by_step[step]["decidable"] += sum(values)
        by_step[step]["not_decidable"] += len(values) - sum(values)
        by_step[step]["occluded"] += sum(
            int(labels[(setup_id, view)]["occluded"]) for view in VIEWS
        )
        setup_oracles += int(any(values))
        fully_decidable_setups += int(all(values))

    policy_payload = json.loads(args.policy_rows.read_text(encoding="utf-8"))
    if args.policy_name not in policy_payload:
        raise ValueError(f"Policy rows do not contain {args.policy_name!r}.")
    base_rows = policy_payload[args.policy_name]
    metrics: dict[str, dict[str, Any]] = {
        "Current View": evaluate_rows(synthetic_rows(base_rows, "current", labels), labels),
        "Uniform Non-current": evaluate_rows(
            synthetic_rows(base_rows, "uniform_non_current", labels), labels
        ),
    }
    for name, rows in policy_payload.items():
        metrics[name] = evaluate_rows(rows, labels)
    metrics["Oracle View"] = evaluate_rows(
        synthetic_rows(base_rows, "oracle", labels), labels
    )

    closed_loop_metrics: dict[str, dict[str, Any]] = {}
    for name, rows in policy_payload.items():
        if not any(len(row.get("path", [])) > 1 for row in rows):
            continue
        closed_loop_metrics[name] = {
            "at_1": evaluate_rows(rows_at_stage(rows, "at_1"), labels),
            "final": evaluate_rows(rows_at_stage(rows, "final"), labels),
            "retained": evaluate_rows(rows_at_stage(rows, "retained"), labels),
            "observed_path_oracle_diagnostic": evaluate_rows(
                observed_path_oracle_rows(rows, labels), labels
            ),
        }

    output = {
        "protocol": {
            "human_labels_used_for_policy": False,
            "candidate_view_images_used": False,
            "policy_rows": str(args.policy_rows),
            "policy_rows_sha256": sha256(args.policy_rows),
            "base_policy_name": args.policy_name,
            "annotations_db": str(args.annotations_db),
            "annotations_db_sha256": sha256(args.annotations_db),
            "full_label_setups": len(setups),
            "full_label_views": len(labels),
            "policy_evaluation_setups": len({row["trial_id"] for row in base_rows}),
            "policy_evaluation_starts": len(base_rows),
            "note": (
                "The frozen policy rows cover Steps 2-4. Step-1 labels are "
                "reported as a dataset diagnostic and are not mixed into policy metrics."
            ),
        },
        "dataset": {
            "decidable_views": sum(value["decidable"] for value in labels.values()),
            "not_decidable_views": sum(
                1 - value["decidable"] for value in labels.values()
            ),
            "explicitly_occluded_views": sum(value["occluded"] for value in labels.values()),
            "setups_with_decidable_view": setup_oracles,
            "setups_with_all_views_decidable": fully_decidable_setups,
            "by_step": {key: dict(value) for key, value in sorted(by_step.items())},
        },
        "policy_metrics": metrics,
        "closed_loop_policy_metrics": closed_loop_metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "robot_binary_decidability_evaluation.json"
    csv_path = args.output_dir / "robot_binary_decidability_policy_metrics.csv"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    columns = (
        "trials",
        "current_decidable",
        "decidable_at_1",
        "binary_gain",
        "recovery_at_1",
        "decidable_preservation",
        "decidable_loss",
        "oracle_achievable",
        "oracle_gap_closed",
        "occlusion_recovery",
        "non_occlusion_recovery",
    )
    write_csv(
        csv_path,
        [
            {"policy": name, **{key: value[key] for key in columns}}
            for name, value in metrics.items()
        ],
    )
    print(json.dumps({"json": str(json_path), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
