"""Leave-one-video-out validation of adaptive assistant observation primitives."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspect_system.active_view.adaptive_observation_basis import (
    AdaptiveObservationBasis,
    ObservationTransition,
    RobustFeatureScaler,
)
from inspect_system.active_view.ontology import normalize_key


RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
BASIS_GRID = (
    (6, 0.75, 0.75),
    (8, 1.25, 1.0),
    (12, 1.25, 1.5),
    (12, 2.0, 1.5),
)
CONTEXT_FIELDS = ("claim", "counterfactual", "role", "motion_source")


@dataclass
class Record:
    event_id: str
    video: str
    transition_key: str
    transition: ObservationTransition
    target: float
    weight: float
    claim: str
    counterfactual: str
    role: str
    motion_source: str
    action: str
    transferable: bool

    def context(self, field: str) -> str:
        return str(getattr(self, field))


def read_records(path: Path) -> List[Record]:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matched = []
    multiplicity: Counter[str] = Counter()
    for event in events:
        metadata = event.get("metadata", {})
        gain = metadata.get("signed_counterfactual_margin_gain")
        if gain in (None, ""):
            continue
        video = Path(str(event.get("video", "unknown"))).name
        key = "|".join(
            (
                video,
                str(event.get("before_frame", "")),
                str(event.get("after_frame", "")),
                normalize_key(event.get("claim_id", "generic")),
            )
        )
        multiplicity[key] += 1
        matched.append((event, metadata, video, key, float(gain)))

    records = []
    for event, metadata, video, key, gain in matched:
        transition = ObservationTransition.from_event(event)
        records.append(
            Record(
                event_id=str(event.get("event_id", key)),
                video=video,
                transition_key=key,
                transition=transition,
                target=float(gain),
                weight=1.0 / float(multiplicity[key]),
                claim=normalize_key(event.get("claim_id", "generic")) or "generic",
                counterfactual=normalize_key(
                    metadata.get("counterfactual_family", "generic")
                )
                or "generic",
                role=normalize_key(
                    metadata.get(
                        "normalized_evidence_role",
                        event.get("evidence_view_type", "generic"),
                    )
                )
                or "generic",
                motion_source=normalize_key(transition.motion_source) or "unknown",
                action=normalize_key(transition.relative_action) or "unknown",
                transferable=bool(metadata.get("transferable", False)),
            )
        )
    return records


def vocabulary(records: Sequence[Record], fields: Sequence[str]) -> Dict[str, List[str]]:
    return {
        field: sorted({record.context(field) for record in records}) + ["__other__"]
        for field in fields
    }


def one_hot(value: str, values: Sequence[str]) -> np.ndarray:
    result = np.zeros((len(values),), dtype=np.float64)
    index = values.index(value) if value in values else values.index("__other__")
    result[index] = 1.0
    return result


def context_vector(record: Record, vocab: Mapping[str, Sequence[str]]) -> np.ndarray:
    return np.concatenate(
        [one_hot(record.context(field), vocab[field]) for field in CONTEXT_FIELDS]
    )


@dataclass
class Design:
    kind: str
    vocab: Dict[str, List[str]]
    action_vocab: List[str]
    scaler: RobustFeatureScaler | None = None
    basis: AdaptiveObservationBasis | None = None

    def row(self, record: Record) -> np.ndarray:
        context = context_vector(record, self.vocab)
        counterfactual = one_hot(record.counterfactual, self.vocab["counterfactual"])
        role = one_hot(record.role, self.vocab["role"])
        if self.kind == "context":
            return np.concatenate(([1.0], context))
        if self.kind == "token":
            action = one_hot(record.action, self.action_vocab)
            interactions = np.concatenate(
                [np.outer(action, counterfactual).ravel(), np.outer(action, role).ravel()]
            )
            return np.concatenate(([1.0], context, action, interactions))
        if self.scaler is None:
            raise RuntimeError("continuous design is missing a scaler")
        continuous = np.asarray(
            self.scaler.transform(record.transition.values), dtype=np.float64
        )
        continuous_interactions = np.concatenate(
            [
                np.outer(continuous, counterfactual).ravel(),
                np.outer(continuous, role).ravel(),
            ]
        )
        parts = [[1.0], context, continuous, continuous_interactions]
        if self.kind == "adaptive":
            if self.basis is None:
                raise RuntimeError("adaptive design is missing its basis")
            radial = self.basis.transform(record.transition)
            radial_interactions = np.concatenate(
                [np.outer(radial, counterfactual).ravel(), np.outer(radial, role).ravel()]
            )
            parts.extend((radial, radial_interactions))
        return np.concatenate(parts)


def make_design(
    kind: str,
    records: Sequence[Record],
    basis_config: Tuple[int, float, float] | None = None,
) -> Design:
    vocab = vocabulary(records, CONTEXT_FIELDS)
    action_vocab = sorted({record.action for record in records}) + ["__other__"]
    if kind in {"context", "token"}:
        return Design(kind=kind, vocab=vocab, action_vocab=action_vocab)
    scaler = RobustFeatureScaler().fit([record.transition.values for record in records])
    if kind == "continuous":
        return Design(kind=kind, vocab=vocab, action_vocab=action_vocab, scaler=scaler)
    if kind != "adaptive" or basis_config is None:
        raise ValueError(f"Unsupported design kind: {kind}")
    max_elements, novelty_radius, bandwidth = basis_config
    basis = AdaptiveObservationBasis(
        max_elements=max_elements,
        novelty_radius=novelty_radius,
        bandwidth=bandwidth,
    ).fit(
        [record.transition for record in records],
        [record.weight for record in records],
    )
    return Design(
        kind=kind,
        vocab=vocab,
        action_vocab=action_vocab,
        scaler=basis.scaler,
        basis=basis,
    )


def ridge_fit(design: np.ndarray, target: np.ndarray, weights: np.ndarray, alpha: float) -> np.ndarray:
    features = design[:, 1:]
    total_weight = float(np.sum(weights))
    feature_mean = np.sum(features * weights[:, None], axis=0) / total_weight
    target_mean = float(np.sum(target * weights) / total_weight)
    root_weight = np.sqrt(weights)[:, None]
    x = (features - feature_mean) * root_weight
    y = (target - target_mean) * root_weight[:, 0]

    if x.shape[1] > x.shape[0]:
        system = x @ x.T + np.eye(x.shape[0], dtype=np.float64) * float(alpha)
        slopes = x.T @ np.linalg.solve(system, y)
    else:
        system = x.T @ x + np.eye(x.shape[1], dtype=np.float64) * float(alpha)
        slopes = np.linalg.solve(system, x.T @ y)

    intercept = target_mean - float(feature_mean @ slopes)
    return np.concatenate(([intercept], slopes))


def weighted_mae(target: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.abs(target - prediction), weights=weights))


def fit_predict(
    kind: str,
    train: Sequence[Record],
    test: Sequence[Record],
    alpha: float,
    basis_config: Tuple[int, float, float] | None,
) -> Tuple[np.ndarray, int]:
    design = make_design(kind, train, basis_config)
    train_x = np.stack([design.row(record) for record in train])
    test_x = np.stack([design.row(record) for record in test])
    target = np.asarray([record.target for record in train], dtype=np.float64)
    weights = np.asarray([record.weight for record in train], dtype=np.float64)
    coefficients = ridge_fit(train_x, target, weights, alpha)
    return test_x @ coefficients, len(design.basis.centers) if design.basis else 0


def inner_score(
    kind: str,
    records: Sequence[Record],
    alpha: float,
    basis_config: Tuple[int, float, float] | None,
) -> float:
    groups = sorted({record.video for record in records})
    errors = []
    weights = []
    for group in groups:
        train = [record for record in records if record.video != group]
        test = [record for record in records if record.video == group]
        if not train or not test:
            continue
        prediction, _ = fit_predict(kind, train, test, alpha, basis_config)
        errors.extend(np.abs(np.asarray([record.target for record in test]) - prediction))
        weights.extend(record.weight for record in test)
    return float(np.average(errors, weights=weights)) if errors else float("inf")


def select_config(
    kind: str,
    records: Sequence[Record],
) -> Tuple[float, Tuple[int, float, float] | None]:
    basis_options: Iterable[Tuple[int, float, float] | None] = (
        BASIS_GRID if kind == "adaptive" else (None,)
    )
    candidates = []
    for basis_config in basis_options:
        for alpha in RIDGE_GRID:
            score = inner_score(kind, records, alpha, basis_config)
            candidates.append((score, alpha, basis_config))
    _, alpha, basis_config = min(
        candidates,
        key=lambda item: (
            item[0],
            item[1],
            item[2] if item[2] is not None else (),
        ),
    )
    return float(alpha), basis_config


def predictions_for_model(kind: str, records: Sequence[Record]) -> List[Dict[str, Any]]:
    output = []
    for video in sorted({record.video for record in records}):
        train = [record for record in records if record.video != video]
        test = [record for record in records if record.video == video]
        alpha, basis_config = select_config(kind, train)
        prediction, basis_size = fit_predict(kind, train, test, alpha, basis_config)
        for record, value in zip(test, prediction):
            output.append(
                {
                    "event_id": record.event_id,
                    "video": record.video,
                    "transition_key": record.transition_key,
                    "target": record.target,
                    "prediction": float(value),
                    "weight": record.weight,
                    "action": record.action,
                    "transferable": record.transferable,
                    "counterfactual": record.counterfactual,
                    "role": record.role,
                    "alpha": alpha,
                    "basis_config": basis_config,
                    "basis_size": basis_size,
                }
            )
    return output


def current_gate_predictions(records: Sequence[Record]) -> List[Dict[str, Any]]:
    output = []
    for video in sorted({record.video for record in records}):
        train = [record for record in records if record.video != video]
        test = [record for record in records if record.video == video]
        by_action: Dict[str, List[Record]] = defaultdict(list)
        for record in train:
            by_action[record.action].append(record)
        global_value = np.average(
            [record.target for record in train],
            weights=[record.weight for record in train],
        )
        for record in test:
            candidates = by_action.get(record.action, [])
            value = (
                np.average(
                    [candidate.target for candidate in candidates],
                    weights=[candidate.weight for candidate in candidates],
                )
                if candidates
                else global_value
            )
            if record.action == "unknown" or not record.transferable:
                value = 0.0
            output.append(
                {
                    "event_id": record.event_id,
                    "video": record.video,
                    "transition_key": record.transition_key,
                    "target": record.target,
                    "prediction": float(value),
                    "weight": record.weight,
                    "action": record.action,
                    "transferable": record.transferable,
                    "counterfactual": record.counterfactual,
                    "role": record.role,
                }
            )
    return output


def metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    target = np.asarray([float(row["target"]) for row in rows])
    prediction = np.asarray([float(row["prediction"]) for row in rows])
    weights = np.asarray([float(row["weight"]) for row in rows])
    sign = np.sign(target)
    predicted_sign = np.sign(prediction)
    unknown = np.asarray([str(row["action"]) == "unknown" for row in rows])
    positive_unknown = unknown & (target > 0.0)
    return {
        "events": len(rows),
        "effective_transitions": float(weights.sum()),
        "mae": weighted_mae(target, prediction, weights),
        "rmse": float(np.sqrt(np.average(np.square(target - prediction), weights=weights))),
        "sign_accuracy": float(np.average(sign == predicted_sign, weights=weights)),
        "unknown_events": int(unknown.sum()),
        "unknown_sign_accuracy": float(
            np.average(sign[unknown] == predicted_sign[unknown], weights=weights[unknown])
        )
        if unknown.any()
        else 0.0,
        "discarded_positive_recovery": float(np.mean(prediction[positive_unknown] > 0.0))
        if positive_unknown.any()
        else 0.0,
    }


def clustered_bootstrap_delta(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    baseline_index = {str(row["event_id"]): row for row in baseline}
    pairs = [(row, baseline_index[str(row["event_id"])]) for row in candidate]
    by_video: Dict[str, List[Tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        by_video[str(pair[0]["video"])].append(pair)
    videos = sorted(by_video)
    rng = random.Random(seed)

    def point(sample_pairs):
        weights = np.asarray([float(pair[0]["weight"]) for pair in sample_pairs])
        candidate_error = np.asarray(
            [abs(float(pair[0]["target"]) - float(pair[0]["prediction"])) for pair in sample_pairs]
        )
        baseline_error = np.asarray(
            [abs(float(pair[1]["target"]) - float(pair[1]["prediction"])) for pair in sample_pairs]
        )
        return float(np.average(baseline_error - candidate_error, weights=weights))

    observed = point(pairs)
    draws = []
    for _ in range(samples):
        sampled = [rng.choice(videos) for _ in videos]
        draws.append(point([pair for video in sampled for pair in by_video[video]]))
    ordered = sorted(draws)
    return {
        "mae_reduction": observed,
        "ci95": [
            ordered[int(0.025 * (len(ordered) - 1))],
            ordered[int(0.975 * (len(ordered) - 1))],
        ],
        "positive_probability": float(np.mean(np.asarray(draws) > 0.0)),
        "clusters": len(videos),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--predictions-json", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    records = read_records(args.events)
    if len({record.video for record in records}) < 3:
        raise ValueError("At least three videos are required for nested grouped validation")
    predictions = {
        "Current Discrete Gate": current_gate_predictions(records),
        "Context Only": predictions_for_model("context", records),
        "Fixed Action Tokens": predictions_for_model("token", records),
        "Continuous Motion": predictions_for_model("continuous", records),
        "Adaptive Observation Basis": predictions_for_model("adaptive", records),
    }
    summary = {
        "protocol": {
            "events": len(records),
            "effective_transitions": sum(record.weight for record in records),
            "videos": len({record.video for record in records}),
            "split": "nested leave-one-video-out",
            "target": "signed target-versus-counterfactual margin gain",
            "robot_view_labels_used": False,
            "candidate_view_images_used": False,
            "duplicate_transition_weighting": "inverse multiplicity",
        },
        "results": {name: metrics(rows) for name, rows in predictions.items()},
        "paired_bootstrap": {
            "adaptive_vs_fixed_tokens": clustered_bootstrap_delta(
                predictions["Adaptive Observation Basis"],
                predictions["Fixed Action Tokens"],
                args.bootstrap_samples,
                args.seed,
            ),
            "adaptive_vs_current_gate": clustered_bootstrap_delta(
                predictions["Adaptive Observation Basis"],
                predictions["Current Discrete Gate"],
                args.bootstrap_samples,
                args.seed + 1,
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    args.predictions_json.write_text(
        json.dumps(predictions, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
