"""Cold-start and online validation of outcome-updated evidence atoms."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspect_system.active_view.evidence_affordance_memory import (
    EvidenceAffordanceConfig,
    EvidenceAffordanceContext,
    EvidenceAffordanceMemory,
    OutcomeStatistics,
)
from scripts.evaluate_adaptive_observation_basis import Record, read_records


CONFIGS = (
    EvidenceAffordanceConfig(4.0, 4.0, 4.0, 2.0, 1.0),
    EvidenceAffordanceConfig(4.0, 6.0, 6.0, 3.0, 2.0),
    EvidenceAffordanceConfig(8.0, 8.0, 8.0, 4.0, 2.0),
    EvidenceAffordanceConfig(8.0, 12.0, 12.0, 6.0, 4.0),
)


def context(record: Record) -> EvidenceAffordanceContext:
    return EvidenceAffordanceContext(
        action=record.action,
        role=record.role,
        counterfactual=record.counterfactual,
        claim=record.claim,
    )


def frame_order(record: Record) -> Tuple[int, int, str]:
    fields = record.transition_key.rsplit("|", 3)
    try:
        before = int(fields[-3])
        after = int(fields[-2])
    except (IndexError, ValueError):
        before, after = 0, 0
    return before, after, record.transition_key


def transition_groups(records: Sequence[Record]) -> List[List[Record]]:
    grouped: Dict[str, List[Record]] = defaultdict(list)
    for record in records:
        grouped[record.transition_key].append(record)
    return sorted(grouped.values(), key=lambda rows: frame_order(rows[0]))


def fit_memory(
    records: Sequence[Record], config: EvidenceAffordanceConfig
) -> EvidenceAffordanceMemory:
    memory = EvidenceAffordanceMemory(config=config)
    return memory.fit((context(row), row.target, row.weight) for row in records)


class BackoffOutcomeBaseline:
    def __init__(self, key_type: str, strength: float = 4.0) -> None:
        self.key_type = key_type
        self.strength = float(strength)
        self.global_outcome = OutcomeStatistics()
        self.atoms: Dict[str, OutcomeStatistics] = {}

    def key(self, record: Record) -> str:
        if self.key_type == "action":
            return record.action
        if self.key_type == "gate":
            return "transferable" if record.transferable else "non_transferable"
        raise ValueError(self.key_type)

    def fit(self, records: Iterable[Record]) -> "BackoffOutcomeBaseline":
        for record in records:
            self.global_outcome.update(record.target, record.weight)
            key = self.key(record)
            if key not in self.atoms:
                self.atoms[key] = OutcomeStatistics()
            self.atoms[key].update(record.target, record.weight)
        return self

    def predict(self, record: Record) -> Tuple[float, float, float, str]:
        global_probability = self.global_outcome.probability(0.5, self.strength)
        global_gain = self.global_outcome.expected_gain(0.0, self.strength)
        atom = self.atoms.get(self.key(record), OutcomeStatistics())
        return (
            atom.probability(global_probability, self.strength),
            atom.expected_gain(global_gain, self.strength),
            atom.weight,
            self.key_type,
        )


def row_result(
    record: Record,
    probability: float,
    gain: float,
    support: float,
    scope: str,
    transition_index: int,
    config: EvidenceAffordanceConfig | None = None,
) -> Dict[str, Any]:
    return {
        "event_id": record.event_id,
        "video": record.video,
        "transition_key": record.transition_key,
        "transition_index": transition_index,
        "action": record.action,
        "role": record.role,
        "counterfactual": record.counterfactual,
        "target": record.target,
        "label": int(record.target > 0.0),
        "weight": record.weight,
        "probability": float(probability),
        "predicted_gain": float(gain),
        "support": float(support),
        "scope": scope,
        "config": asdict(config) if config else None,
    }


def evaluate_baseline(
    train: Sequence[Record], test: Sequence[Record], key_type: str
) -> List[Dict[str, Any]]:
    model = BackoffOutcomeBaseline(key_type).fit(train)
    results = []
    for index, group in enumerate(transition_groups(test)):
        for record in group:
            prediction = model.predict(record)
            results.append(row_result(record, *prediction, transition_index=index))
    return results


def evaluate_memory(
    train: Sequence[Record],
    test: Sequence[Record],
    config: EvidenceAffordanceConfig,
    online: bool,
) -> List[Dict[str, Any]]:
    memory = fit_memory(train, config)
    results = []
    session_id = test[0].video if test else None
    for index, group in enumerate(transition_groups(test)):
        for record in group:
            prediction = memory.predict(
                context(record), session_id=session_id if online else None
            )
            results.append(
                row_result(
                    record,
                    prediction.helpful_probability,
                    prediction.expected_gain,
                    prediction.effective_support,
                    prediction.scope,
                    index,
                    config,
                )
            )
        if online:
            for record in group:
                memory.update(
                    context(record),
                    record.target,
                    record.weight,
                    session_id=session_id,
                    update_shared=True,
                )
    return results


def weighted_average(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    weights = np.asarray([float(row["weight"]) for row in rows], dtype=np.float64)
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return float(np.average(values, weights=weights))


def metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    weights = np.asarray([float(row["weight"]) for row in rows], dtype=np.float64)
    labels = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    probabilities = np.asarray(
        [float(row["probability"]) for row in rows], dtype=np.float64
    )
    targets = np.asarray([float(row["target"]) for row in rows], dtype=np.float64)
    gains = np.asarray(
        [float(row["predicted_gain"]) for row in rows], dtype=np.float64
    )
    decisions = probabilities >= 0.5
    positive = labels == 1.0
    negative = ~positive

    def class_accuracy(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.average(decisions[mask] == positive[mask], weights=weights[mask]))

    probability_clip = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    positive_accuracy = class_accuracy(positive)
    negative_accuracy = class_accuracy(negative)
    balanced = float(np.nanmean([positive_accuracy, negative_accuracy]))
    return {
        "events": len(rows),
        "effective_transitions": float(np.sum(weights)),
        "brier": float(np.average(np.square(probabilities - labels), weights=weights)),
        "log_loss": float(
            np.average(
                -(labels * np.log(probability_clip) + (1.0 - labels) * np.log(1.0 - probability_clip)),
                weights=weights,
            )
        ),
        "sign_accuracy": float(np.average(decisions == positive, weights=weights)),
        "balanced_accuracy": balanced,
        "positive_recall": positive_accuracy,
        "negative_recall": negative_accuracy,
        "gain_mae": float(np.average(np.abs(gains - targets), weights=weights)),
    }


def inner_score(
    records: Sequence[Record],
    config: EvidenceAffordanceConfig,
    online: bool,
    objective: str = "brier",
) -> float:
    videos = sorted({record.video for record in records})
    predictions = []
    for video in videos:
        train = [record for record in records if record.video != video]
        test = [record for record in records if record.video == video]
        if not train or not test:
            continue
        predictions.extend(evaluate_memory(train, test, config, online))
    if objective not in {"brier", "gain_mae"}:
        raise ValueError(f"Unknown model-selection objective: {objective}")
    return metrics(predictions)[objective] if predictions else float("inf")


def select_config(
    records: Sequence[Record],
    online: bool,
    objective: str = "brier",
) -> EvidenceAffordanceConfig:
    scored = [
        (inner_score(records, config, online, objective), index, config)
        for index, config in enumerate(CONFIGS)
    ]
    return min(scored, key=lambda item: (item[0], item[1]))[2]


def nested_leave_one_video_out(records: Sequence[Record]) -> Dict[str, List[Dict[str, Any]]]:
    output: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for video in sorted({record.video for record in records}):
        train = [record for record in records if record.video != video]
        test = [record for record in records if record.video == video]
        output["Current Discrete Gate"].extend(evaluate_baseline(train, test, "gate"))
        output["Fixed Action Atoms"].extend(evaluate_baseline(train, test, "action"))
        static_config = select_config(train, online=False)
        online_config = select_config(train, online=True)
        output["Static Hierarchical Atoms"].extend(
            evaluate_memory(train, test, static_config, online=False)
        )
        output["Matched Frozen Atoms"].extend(
            evaluate_memory(train, test, online_config, online=False)
        )
        output["Online Hierarchical Atoms"].extend(
            evaluate_memory(train, test, online_config, online=True)
        )
    return dict(output)


def cluster_bootstrap_reduction(
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    reference: str,
    candidate: str,
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    by_model_video: Dict[str, Dict[str, List[Mapping[str, Any]]]] = {}
    for model, rows in predictions.items():
        grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["video"])].append(row)
        by_model_video[model] = grouped
    videos = sorted(by_model_video[candidate])
    rng = random.Random(seed)
    differences = []
    for _ in range(samples):
        selected = [rng.choice(videos) for _ in videos]
        reference_rows = [row for video in selected for row in by_model_video[reference][video]]
        candidate_rows = [row for video in selected for row in by_model_video[candidate][video]]
        differences.append(metrics(reference_rows)["brier"] - metrics(candidate_rows)["brier"])
    values = np.asarray(differences, dtype=np.float64)
    observed = metrics(predictions[reference])["brier"] - metrics(predictions[candidate])["brier"]
    return {
        "brier_reduction": observed,
        "ci95": np.percentile(values, [2.5, 97.5]).tolist(),
        "positive_probability": float(np.mean(values > 0.0)),
        "clusters": len(videos),
        "samples": samples,
    }


def subset_metrics(
    predictions: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    output: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model, rows in predictions.items():
        output[model] = {
            "cold_start": metrics([row for row in rows if int(row["transition_index"]) == 0]),
            "after_feedback": metrics([row for row in rows if int(row["transition_index"]) > 0]),
            "unknown_action": metrics([row for row in rows if row["action"] == "unknown"]),
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--predictions-json", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_records(args.events)
    predictions = nested_leave_one_video_out(records)
    summary = {
        "protocol": {
            "events": len(records),
            "effective_transitions": float(sum(record.weight for record in records)),
            "videos": len({record.video for record in records}),
            "split": "nested leave-one-video-out with causal prequential updates",
            "target": "sign and magnitude of target-versus-counterfactual margin gain",
            "online_update_timing": "predict transition, then update from its resolved outcome",
            "duplicate_transition_weighting": "inverse multiplicity",
            "candidate_view_images_used": False,
            "robot_view_labels_used": False,
        },
        "results": {model: metrics(rows) for model, rows in predictions.items()},
        "subsets": subset_metrics(predictions),
        "paired_bootstrap": {
            "online_vs_matched_frozen": cluster_bootstrap_reduction(
                predictions,
                "Matched Frozen Atoms",
                "Online Hierarchical Atoms",
                args.bootstrap_samples,
                args.seed,
            ),
            "online_vs_fixed_action": cluster_bootstrap_reduction(
                predictions,
                "Fixed Action Atoms",
                "Online Hierarchical Atoms",
                args.bootstrap_samples,
                args.seed + 1,
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.predictions_json.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
