"""Evaluate learned causal step proposals from frozen assistant evidence.

Step and outcome annotations are used only as labels. Features contain no
filename, future frame, outcome, target step, or ground-truth object input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspect_assist.__main__ import _load_runtime_package  # noqa: E402

_load_runtime_package()

from evaluate_procedural_claim_triage_calibrated import (  # noqa: E402
    group_timeline,
    read_csv,
    state_id,
    timeline_at,
    video_key,
)
from inspect_runtime.components.kb import KnowledgeBase  # noqa: E402
from inspect_runtime.components.rules import RuleBasedStepExpert  # noqa: E402
from inspect_runtime.core_types import Detection  # noqa: E402


STEPS = ("S1", "S2", "S3", "S4")
CLASSES = (
    "type_2_gear",
    "type_3_gear",
    "type_5_gearbox_cover",
    "type_5_gearbox_housing",
    "type_6_gearbox_cover",
    "type_6_gearbox_housing",
    "type_7_gear",
    "type_8_gear",
)
ROLE_BY_CLASS = {
    "type_2_gear": "big_gear",
    "type_8_gear": "big_gear",
    "type_3_gear": "small_gear",
    "type_7_gear": "small_gear",
    "type_5_gearbox_cover": "cover",
    "type_6_gearbox_cover": "cover",
    "type_5_gearbox_housing": "housing",
    "type_6_gearbox_housing": "housing",
}
ROLES = ("housing", "small_gear", "big_gear", "cover")
PREDICATES = ("inside", "aligned_with", "contacting", "near", "overlapping")
ROLE_PAIRS = (
    ("small_gear", "housing"),
    ("big_gear", "housing"),
    ("cover", "housing"),
    ("small_gear", "big_gear"),
)
STEP_INDEX = {step: index for index, step in enumerate(STEPS)}


@dataclass(frozen=True)
class Sample:
    video: str
    frame: int
    target: str
    outcome: str
    features: tuple[float, ...]
    rule_scores: tuple[float, ...]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def scenario_group(video: str) -> str:
    stem = Path(video).stem.lower()
    stem = re.sub(r"_\d+$", "", stem)
    stem = re.sub(r"_(supported|contradicted|unresolved)$", "", stem)
    return stem.replace("view_fail_", "view_")


def confidence(row: Mapping[str, Any]) -> float:
    return float(row.get("confidence", row.get("conf", 0.0)) or 0.0)


def to_detection(row: Mapping[str, Any]) -> Detection:
    return Detection(
        name=str(row.get("name", "")),
        xyxy=tuple(float(value) for value in row.get("xyxy", (0, 0, 0, 0))),
        confidence=confidence(row),
        meta=dict(row.get("meta") or {}),
    )


def relation_tuple(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    subject = str(row.get("subject_name", row.get("subject", ""))).lower()
    predicate = str(row.get("predicate", "")).lower()
    obj = str(row.get("object_name", row.get("object", ""))).lower()
    return subject, predicate, obj, float(row.get("score", 0.0) or 0.0)


def base_feature_names() -> list[str]:
    names = [f"class_conf:{name}" for name in CLASSES]
    names += [f"class_count:{name}" for name in CLASSES]
    names += [f"role_conf:{role}" for role in ROLES]
    names += [f"role_count:{role}" for role in ROLES]
    names += [f"stable_role_count:{role}" for role in ROLES]
    names += [
        f"pair:{left}:{predicate}:{right}"
        for left, right in ROLE_PAIRS
        for predicate in PREDICATES
    ]
    names += [f"scene_count:{predicate}" for predicate in PREDICATES]
    names += ["det_count", "stable_track_count", "relation_density", "has_visual"]
    names += [f"rule_score:{step}" for step in STEPS]
    return names


EVENT_NAMES = (
    "small_gear_assembly",
    "big_gear_assembly",
    "cover_assembly",
    "small_gear_presence",
    "big_gear_presence",
    "cover_presence",
)


def feature_names() -> list[str]:
    names = base_feature_names()
    names += [f"event:{name}" for name in EVENT_NAMES]
    names += [f"causal_ema:{name}" for name in EVENT_NAMES]
    names += [f"causal_peak:{name}" for name in EVENT_NAMES]
    return names


def load_appearance_embeddings(
    path: Path | None,
    *,
    causal_history: bool = False,
    cache_stride: int = 1,
) -> tuple[dict[tuple[str, int], tuple[float, ...]], list[str]]:
    if path is None:
        return {}, []
    payload = np.load(path, allow_pickle=False)
    embeddings = np.asarray(payload["roi_embedding"], dtype=np.float32)
    projection_dim = min(64, embeddings.shape[1])
    rng = np.random.default_rng(20260828)
    projection = rng.standard_normal(
        (embeddings.shape[1], projection_dim),
        dtype=np.float32,
    ) / np.sqrt(float(projection_dim))
    embeddings = embeddings @ projection
    embeddings /= np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True),
        1e-8,
    )
    roi_valid = np.asarray(payload["roi_valid"], dtype=np.float32)
    detection_count = np.asarray(payload["detection_count"], dtype=np.float32)
    roi_boxes = np.asarray(payload["roi_box"], dtype=np.float32)
    mapping: dict[tuple[str, int], tuple[float, ...]] = {}
    for index, (video, frame) in enumerate(zip(payload["video"], payload["frame"])):
        vector = np.concatenate(
            [
                np.asarray(
                    [roi_valid[index], min(1.0, detection_count[index] / 4.0)],
                    dtype=np.float32,
                ),
                roi_boxes[index],
                embeddings[index],
            ]
        )
        mapping[(video_key(str(video)), int(frame))] = tuple(float(x) for x in vector)
    names = ["appearance:roi_valid", "appearance:detection_count"]
    names += [
        "appearance:roi_x1",
        "appearance:roi_y1",
        "appearance:roi_x2",
        "appearance:roi_y2",
    ]
    names += [
        f"appearance:dinov2_rp_{index:03d}"
        for index in range(embeddings.shape[1])
    ]
    if cache_stride > 1:
        mapping = apply_causal_embedding_cache(
            mapping,
            embedding_dim=int(embeddings.shape[1]),
            stride=cache_stride,
        )
    if causal_history:
        mapping, history_names = add_causal_appearance_history(
            mapping,
            embedding_dim=int(embeddings.shape[1]),
        )
        names += history_names
    return mapping, names


def apply_causal_embedding_cache(
    mapping: Mapping[tuple[str, int], Sequence[float]],
    *,
    embedding_dim: int,
    stride: int,
) -> dict[tuple[str, int], tuple[float, ...]]:
    """Reuse only the most recent past appearance embedding.

    Detector-derived validity, count, and ROI geometry remain current. The
    expensive embedding block is refreshed at the requested observation
    stride and is never copied backward or across videos.
    """

    resolved_stride = max(1, int(stride))
    if resolved_stride == 1:
        return {key: tuple(float(value) for value in values) for key, values in mapping.items()}

    output: dict[tuple[str, int], tuple[float, ...]] = {}
    by_video: dict[str, list[tuple[int, Sequence[float]]]] = defaultdict(list)
    for (video, frame), values in mapping.items():
        by_video[str(video)].append((int(frame), values))

    for video, rows in by_video.items():
        cached: np.ndarray | None = None
        for observation_index, (frame, raw) in enumerate(sorted(rows)):
            values = np.asarray(raw, dtype=np.float32).copy()
            current = values[6 : 6 + embedding_dim]
            current_valid = bool(values[0] > 0.5) and bool(np.linalg.norm(current) > 1e-8)
            refresh_due = observation_index % resolved_stride == 0
            if current_valid and (cached is None or refresh_due):
                cached = current.copy()
            elif cached is not None:
                values[6 : 6 + embedding_dim] = cached
            output[(video, frame)] = tuple(float(value) for value in values)
    return output


def add_causal_appearance_history(
    mapping: Mapping[tuple[str, int], Sequence[float]],
    *,
    embedding_dim: int,
) -> tuple[dict[tuple[str, int], tuple[float, ...]], list[str]]:
    """Append past-only short/long appearance state and innovation.

    The first six values are ROI validity/count/geometry and the remaining
    values are the frozen appearance embedding. Invalid observations preserve
    the previous state but emit zero innovation. No labels or future frames
    enter this representation.
    """

    output: dict[tuple[str, int], tuple[float, ...]] = {}
    by_video: dict[str, list[tuple[int, Sequence[float]]]] = defaultdict(list)
    for (video, frame), values in mapping.items():
        by_video[str(video)].append((int(frame), values))
    for video, rows in by_video.items():
        short = np.zeros(embedding_dim, dtype=np.float32)
        long = np.zeros(embedding_dim, dtype=np.float32)
        initialized = False
        for frame, raw in sorted(rows):
            values = np.asarray(raw, dtype=np.float32)
            current = values[6 : 6 + embedding_dim]
            valid = bool(values[0] > 0.5) and bool(np.linalg.norm(current) > 1e-8)
            innovation = np.zeros_like(current)
            if valid:
                if not initialized:
                    short = current.copy()
                    long = current.copy()
                    initialized = True
                else:
                    innovation = current - long
                    short = 0.50 * current + 0.50 * short
                    long = 0.15 * current + 0.85 * long
            output[(video, frame)] = tuple(
                float(value)
                for value in np.concatenate((values, short, long, innovation))
            )
    names = []
    for prefix in ("short_ema", "long_ema", "innovation"):
        names.extend(
            f"appearance:{prefix}_{index:03d}"
            for index in range(embedding_dim)
        )
    return output, names


def event_signals(features: Sequence[float]) -> np.ndarray:
    index = {name: offset for offset, name in enumerate(base_feature_names())}

    def value(name: str) -> float:
        return float(features[index[name]])

    def assembly(role: str) -> float:
        relation = max(
            value(f"pair:{role}:inside:housing"),
            value(f"pair:{role}:aligned_with:housing"),
            value(f"pair:{role}:contacting:housing"),
            value(f"pair:{role}:overlapping:housing"),
            0.25 * value(f"pair:{role}:near:housing"),
        )
        presence = min(value(f"role_conf:{role}"), value("role_conf:housing"))
        stable = min(
            value(f"stable_role_count:{role}"),
            value("stable_role_count:housing"),
        )
        return max(relation, 0.45 * presence, 0.35 * stable)

    return np.asarray(
        [
            assembly("small_gear"),
            assembly("big_gear"),
            assembly("cover"),
            min(value("role_conf:small_gear"), value("role_conf:housing")),
            min(value("role_conf:big_gear"), value("role_conf:housing")),
            min(value("role_conf:cover"), value("role_conf:housing")),
        ],
        dtype=np.float64,
    )


def add_causal_event_memory(samples: Sequence[Sample]) -> list[Sample]:
    """Append online event state computed only from current and past evidence."""

    output = list(samples)
    state: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    order = sorted(
        range(len(samples)),
        key=lambda index: (samples[index].video, samples[index].frame),
    )
    for index in order:
        sample = samples[index]
        current = event_signals(sample.features)
        previous_ema, previous_peak = state.get(
            sample.video,
            (np.zeros_like(current), np.zeros_like(current)),
        )
        ema = 0.55 * current + 0.45 * previous_ema
        peak = np.maximum(current, 0.995 * previous_peak)
        output[index] = replace(
            sample,
            features=tuple(sample.features) + tuple(current) + tuple(ema) + tuple(peak),
        )
        state[sample.video] = (ema, peak)
    return output


def frame_features(item: Mapping[str, Any], rule: RuleBasedStepExpert) -> tuple[tuple[float, ...], tuple[float, ...]]:
    raw = list(item.get("fused_detections") or item.get("raw_detections") or [])
    relations = list(item.get("scene_graph_relations") or [])
    detections = [to_detection(row) for row in raw]
    prediction = rule.predict({"detections": detections, "relations": relations})

    class_conf = {name: 0.0 for name in CLASSES}
    class_count = Counter()
    role_conf = {role: 0.0 for role in ROLES}
    role_count = Counter()
    for row in raw:
        name = str(row.get("name", "")).lower()
        score = confidence(row)
        if name in class_conf:
            class_conf[name] = max(class_conf[name], score)
            class_count[name] += 1
        role = ROLE_BY_CLASS.get(name)
        if role:
            role_conf[role] = max(role_conf[role], score)
            role_count[role] += 1

    stable_roles = Counter()
    stable = ((item.get("evidence_token") or {}).get("stable_track_counts") or {})
    for name, count in stable.items():
        role = ROLE_BY_CLASS.get(str(name).lower())
        if role:
            stable_roles[role] += int(count)

    pair_scores = {
        (left, predicate, right): 0.0
        for left, right in ROLE_PAIRS
        for predicate in PREDICATES
    }
    predicate_counts = Counter()
    for row in relations:
        subject, predicate, obj, score = relation_tuple(row)
        if predicate not in PREDICATES:
            continue
        predicate_counts[predicate] += 1
        subject_role = ROLE_BY_CLASS.get(subject, subject)
        object_role = ROLE_BY_CLASS.get(obj, obj)
        for left, right in ROLE_PAIRS:
            if {subject_role, object_role} == {left, right}:
                pair_scores[(left, predicate, right)] = max(
                    pair_scores[(left, predicate, right)], score
                )

    values = [class_conf[name] for name in CLASSES]
    values += [min(3, class_count[name]) / 3.0 for name in CLASSES]
    values += [role_conf[role] for role in ROLES]
    values += [min(3, role_count[role]) / 3.0 for role in ROLES]
    values += [min(3, stable_roles[role]) / 3.0 for role in ROLES]
    values += [
        pair_scores[(left, predicate, right)]
        for left, right in ROLE_PAIRS
        for predicate in PREDICATES
    ]
    values += [min(4, predicate_counts[predicate]) / 4.0 for predicate in PREDICATES]
    values += [
        min(8, len(raw)) / 8.0,
        min(8, sum(stable_roles.values())) / 8.0,
        min(20, len(relations)) / 20.0,
        float(bool(item.get("has_visual_evidence", raw or relations))),
    ]
    rule_scores = tuple(float(prediction.scores.get(step, 0.0)) for step in STEPS)
    values += list(rule_scores)
    return tuple(values), rule_scores


def build_samples(
    summary_csv: Path,
    timeline_csv: Path,
    kb_path: Path,
    embedding_npz: Path | None = None,
    causal_appearance_history: bool = False,
    appearance_cache_stride: int = 1,
) -> tuple[list[Sample], list[str]]:
    timelines = group_timeline(timeline_csv)
    rule = RuleBasedStepExpert(KnowledgeBase.from_path(str(kb_path)), list(STEPS))
    appearance_lookup, appearance_names = load_appearance_embeddings(
        embedding_npz,
        causal_history=causal_appearance_history,
        cache_stride=appearance_cache_stride,
    )
    samples: list[Sample] = []
    for run in read_csv(summary_csv):
        if str(run.get("returncode", "")).strip() not in {"", "0"}:
            continue
        video = str(run.get("video", ""))
        timeline = timelines.get(video_key(video), [])
        run_dir = Path(str(run.get("run_dir", "")))
        if not timeline or not (run_dir / "iterations.jsonl").is_file():
            continue
        for item in iter_jsonl(run_dir / "iterations.jsonl"):
            frame = int(item.get("frame_index", -1))
            target_row = timeline_at(timeline, frame)
            if not target_row:
                continue
            target = state_id(target_row.get("step_id"))
            if target not in STEP_INDEX:
                continue
            features, rule_scores = frame_features(item, rule)
            features = tuple(features) + appearance_lookup.get(
                (video_key(video), frame),
                tuple(0.0 for _ in appearance_names),
            )
            samples.append(
                Sample(
                    video=video,
                    frame=frame,
                    target=target,
                    outcome=str(target_row.get("outcome_norm", "")),
                    features=features,
                    rule_scores=rule_scores,
                )
            )
    return add_causal_event_memory(samples), appearance_names


def balanced_weights(labels: np.ndarray) -> np.ndarray:
    counts = Counter(int(value) for value in labels)
    return np.asarray([len(labels) / (len(counts) * counts[int(value)]) for value in labels])


def model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=3.0,
        random_state=20260828,
    )


def structured_feature_vector(sample: Sample, appearance_dim: int) -> np.ndarray:
    """Remove the frozen appearance block while retaining online event memory."""

    values = np.asarray(sample.features, dtype=np.float64)
    if appearance_dim <= 0:
        return values
    boundary = len(base_feature_names())
    return np.concatenate((values[:boundary], values[boundary + appearance_dim :]))


def appearance_event_feature_vector(sample: Sample, appearance_dim: int) -> np.ndarray:
    """Keep frozen appearance and causal event state, excluding raw structured cues."""

    values = np.asarray(sample.features, dtype=np.float64)
    if appearance_dim <= 0:
        return values
    boundary = len(base_feature_names())
    return np.concatenate(
        (
            values[boundary : boundary + appearance_dim],
            values[boundary + appearance_dim :],
        )
    )


def classwise_blend(
    appearance: np.ndarray,
    structured: np.ndarray,
    weights: Sequence[float],
) -> np.ndarray:
    """Fuse frozen appearance only for stages selected on training groups."""

    alpha = np.asarray(weights, dtype=np.float64).reshape(1, -1)
    mixed = alpha * appearance + (1.0 - alpha) * structured
    return mixed / np.maximum(mixed.sum(axis=1, keepdims=True), 1e-9)


def appearance_route_mask(
    structured: np.ndarray,
    *,
    confidence_threshold: float,
    force_insertion_competition: bool,
) -> np.ndarray:
    """Select expensive appearance inference from cheap proposal uncertainty."""

    confidence = np.max(structured, axis=1)
    predicted = np.argmax(structured, axis=1)
    route = confidence < float(confidence_threshold)
    if force_insertion_competition:
        route |= np.isin(predicted, [STEP_INDEX["S2"], STEP_INDEX["S3"]])
    return route


def aligned_probabilities(estimator: HistGradientBoostingClassifier, values: np.ndarray) -> np.ndarray:
    raw = estimator.predict_proba(values)
    result = np.zeros((len(values), len(STEPS)), dtype=np.float64)
    for column, class_index in enumerate(estimator.classes_):
        result[:, int(class_index)] = raw[:, column]
    return result


def rule_probabilities(samples: Sequence[Sample]) -> np.ndarray:
    scores = np.asarray([sample.rule_scores for sample in samples], dtype=np.float64)
    scores = scores / 0.20
    scores -= scores.max(axis=1, keepdims=True)
    scores = np.exp(scores)
    return scores / np.maximum(scores.sum(axis=1, keepdims=True), 1e-9)


def causal_filter(probabilities: np.ndarray, samples: Sequence[Sample], decay: float) -> np.ndarray:
    result = probabilities.copy()
    state: dict[str, np.ndarray] = {}
    order = sorted(range(len(samples)), key=lambda index: (samples[index].video, samples[index].frame))
    for index in order:
        previous = state.get(samples[index].video)
        posterior = probabilities[index]
        if previous is not None and decay > 0.0:
            posterior = (1.0 - decay) * posterior + decay * previous
            posterior = posterior / max(float(posterior.sum()), 1e-9)
        result[index] = posterior
        state[samples[index].video] = posterior
    return result


def stage_shared_blend(
    learned: np.ndarray,
    rule: np.ndarray,
    *,
    boundary_weight: float,
    insertion_weight: float,
) -> np.ndarray:
    """Use one expert gate for insertion stages and one for boundary stages."""

    weights = np.asarray(
        [boundary_weight, insertion_weight, insertion_weight, boundary_weight],
        dtype=np.float64,
    ).reshape(1, -1)
    mixed = weights * learned + (1.0 - weights) * rule
    return mixed / np.maximum(mixed.sum(axis=1, keepdims=True), 1e-9)


def proposal_score(probabilities: np.ndarray, samples: Sequence[Sample]) -> float:
    labels = np.asarray([STEP_INDEX[sample.target] for sample in samples])
    order = np.argsort(-probabilities, axis=1)
    top1 = float(np.mean(order[:, 0] == labels))
    supported = np.asarray([sample.outcome == "supported" for sample in samples])
    class_top1: list[float] = []
    class_top2: list[float] = []
    for step_index in range(len(STEPS)):
        mask = supported & (labels == step_index)
        if not mask.any():
            continue
        class_top1.append(float(np.mean(order[mask, 0] == labels[mask])))
        class_top2.append(
            float(np.mean([label in row[:2] for label, row in zip(labels[mask], order[mask])]))
        )
    macro_top1 = float(np.mean(class_top1)) if class_top1 else 0.0
    macro_top2 = float(np.mean(class_top2)) if class_top2 else 0.0
    return macro_top1 + 0.30 * macro_top2 + 0.10 * top1


def calibration_groups(groups: Sequence[str]) -> set[str]:
    ranked = sorted(set(groups), key=lambda value: hashlib.sha1(value.encode()).hexdigest())
    return set(ranked[: max(2, int(round(0.2 * len(ranked))))])


def file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "size_bytes": int(path.stat().st_size),
    }


def summarize(probabilities: np.ndarray, samples: Sequence[Sample]) -> dict[str, Any]:
    labels = np.asarray([STEP_INDEX[sample.target] for sample in samples])
    order = np.argsort(-probabilities, axis=1)
    result: dict[str, Any] = {"frames": len(samples)}
    for outcome in ("all", "supported", "contradicted", "unresolved"):
        mask = np.ones(len(samples), dtype=bool) if outcome == "all" else np.asarray(
            [sample.outcome == outcome for sample in samples]
        )
        result[outcome] = {
            "frames": int(mask.sum()),
            "top1": float(np.mean(order[mask, 0] == labels[mask])) if mask.any() else None,
            "top2": float(np.mean([label in row[:2] for label, row in zip(labels[mask], order[mask])])) if mask.any() else None,
        }
    predictions = order[:, 0]
    result["by_step"] = {}
    for step, step_index in STEP_INDEX.items():
        mask = labels == step_index
        supported_mask = mask & np.asarray(
            [sample.outcome == "supported" for sample in samples]
        )
        result["by_step"][step] = {
            "frames": int(mask.sum()),
            "top1": float(np.mean(predictions[mask] == labels[mask])) if mask.any() else None,
            "supported_frames": int(supported_mask.sum()),
            "supported_top1": (
                float(np.mean(predictions[supported_mask] == labels[supported_mask]))
                if supported_mask.any()
                else None
            ),
            "predicted_as": {
                candidate: int(np.sum(predictions[mask] == candidate_index))
                for candidate, candidate_index in STEP_INDEX.items()
            },
        }
    supported_step_rows = [
        row["supported_top1"]
        for row in result["by_step"].values()
        if row["supported_top1"] is not None
    ]
    result["supported_macro_top1"] = (
        float(np.mean(supported_step_rows)) if supported_step_rows else None
    )
    return result


def evaluate(
    samples: Sequence[Sample],
    folds: int,
    appearance_dim: int = 0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not samples:
        raise ValueError(
            "No replay frames matched the timeline. Check video keys, run directories, "
            "frame indices, and timeline step labels before evaluation."
        )
    labels = np.asarray([STEP_INDEX[sample.target] for sample in samples], dtype=np.int64)
    groups = np.asarray([scenario_group(sample.video) for sample in samples])
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Grouped evaluation requires at least two independent scenario groups.")
    folds = min(int(folds), len(unique_groups))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=20260828)
    learned = np.zeros((len(samples), len(STEPS)), dtype=np.float64)
    structured = np.zeros_like(learned)
    appearance_event = np.zeros_like(learned)
    fused = np.zeros_like(learned)
    stage_shared = np.zeros_like(learned)
    selective_appearance = np.zeros_like(learned)
    epistemic_cascade = np.zeros_like(learned)
    cascade_invoked = np.zeros(len(samples), dtype=bool)
    fold_ids = np.zeros(len(samples), dtype=np.int32)
    selected: dict[str, dict[str, float]] = {}
    for fold, (train_index, test_index) in enumerate(splitter.split(np.zeros(len(samples)), labels, groups), 1):
        fold_ids[test_index] = int(fold)
        train_groups = groups[train_index]
        held_calibration = calibration_groups(train_groups)
        fit_index = np.asarray([index for index in train_index if groups[index] not in held_calibration])
        cal_index = np.asarray([index for index in train_index if groups[index] in held_calibration])
        estimator = model()
        estimator.fit(
            np.asarray([samples[index].features for index in fit_index]),
            labels[fit_index],
            sample_weight=balanced_weights(labels[fit_index]),
        )
        cal_samples = [samples[index] for index in cal_index]
        cal_learned = aligned_probabilities(
            estimator, np.asarray([sample.features for sample in cal_samples])
        )
        structured_estimator = model()
        structured_estimator.fit(
            np.asarray(
                [structured_feature_vector(samples[index], appearance_dim) for index in fit_index]
            ),
            labels[fit_index],
            sample_weight=balanced_weights(labels[fit_index]),
        )
        cal_structured = aligned_probabilities(
            structured_estimator,
            np.asarray(
                [structured_feature_vector(sample, appearance_dim) for sample in cal_samples]
            ),
        )
        cal_rule = rule_probabilities(cal_samples)
        best = (-1.0, 1.0, 0.0)
        for alpha in (0.50, 0.75, 1.0):
            for decay in (0.0, 0.25, 0.50, 0.75):
                probabilities = causal_filter(
                    alpha * cal_learned + (1.0 - alpha) * cal_rule,
                    cal_samples,
                    decay,
                )
                candidate = (proposal_score(probabilities, cal_samples), alpha, -decay)
                if candidate > best:
                    best = candidate
        alpha, decay = best[1], -best[2]
        cascade_threshold = 1.01
        cascade_force_insertion = False
        cascade_calibration_score = 0.0
        cascade_full_score = 0.0
        if appearance_dim > 0:
            full_calibration = causal_filter(
                alpha * cal_learned + (1.0 - alpha) * cal_rule,
                cal_samples,
                decay,
            )
            cascade_full_score = proposal_score(full_calibration, cal_samples)
            cascade_candidates: list[dict[str, float | bool]] = []
            for force_insertion in (False, True):
                for threshold in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.01):
                    route = appearance_route_mask(
                        cal_structured,
                        confidence_threshold=threshold,
                        force_insertion_competition=force_insertion,
                    )
                    routed = np.where(route[:, None], cal_learned, cal_structured)
                    probabilities = causal_filter(
                        alpha * routed + (1.0 - alpha) * cal_rule,
                        cal_samples,
                        decay,
                    )
                    cascade_candidates.append(
                        {
                            "score": proposal_score(probabilities, cal_samples),
                            "invocation_rate": float(np.mean(route)),
                            "threshold": float(threshold),
                            "force_insertion": bool(force_insertion),
                        }
                    )
            tolerance = 0.01
            feasible = [
                candidate
                for candidate in cascade_candidates
                if float(candidate["score"]) >= cascade_full_score - tolerance
            ]
            if feasible:
                chosen = max(
                    feasible,
                    key=lambda candidate: (
                        -float(candidate["invocation_rate"]),
                        float(candidate["score"]),
                    ),
                )
            else:
                chosen = max(
                    cascade_candidates,
                    key=lambda candidate: (
                        float(candidate["score"]),
                        -float(candidate["invocation_rate"]),
                    ),
                )
            cascade_threshold = float(chosen["threshold"])
            cascade_force_insertion = bool(chosen["force_insertion"])
            cascade_calibration_score = float(chosen["score"])
        shared_best = (-float("inf"), 0.5, 0.5, 0.0)
        for boundary_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            for insertion_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
                mixed = stage_shared_blend(
                    cal_learned,
                    cal_rule,
                    boundary_weight=boundary_weight,
                    insertion_weight=insertion_weight,
                )
                for candidate_decay in (0.0, 0.25, 0.50, 0.75):
                    probabilities = causal_filter(mixed, cal_samples, candidate_decay)
                    score = proposal_score(probabilities, cal_samples)
                    candidate = (
                        score,
                        -abs(boundary_weight - insertion_weight),
                        -candidate_decay,
                    )
                    if candidate > (
                        shared_best[0],
                        -abs(shared_best[1] - shared_best[2]),
                        -shared_best[3],
                    ):
                        shared_best = (
                            score,
                            float(boundary_weight),
                            float(insertion_weight),
                            float(candidate_decay),
                        )
        _, boundary_weight, insertion_weight, shared_decay = shared_best
        appearance_weights = (0.0, 0.0, 0.0, 0.0)
        selective_boundary_weight = boundary_weight
        selective_insertion_weight = insertion_weight
        selective_decay = shared_decay
        if appearance_dim > 0:
            appearance_best = (-float("inf"), 0.0, 0.0, 0.0)
            for small_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
                for big_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
                    for cover_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
                        candidate_weights = (0.0, small_weight, big_weight, cover_weight)
                        probabilities = classwise_blend(
                            cal_learned,
                            cal_structured,
                            candidate_weights,
                        )
                        score = proposal_score(probabilities, cal_samples)
                        candidate = (
                            score,
                            -sum(candidate_weights),
                            small_weight,
                            big_weight,
                            cover_weight,
                        )
                        if candidate > (
                            appearance_best[0],
                            -sum(appearance_best[1:]),
                            *appearance_best[1:],
                        ):
                            appearance_best = (
                                score,
                                small_weight,
                                big_weight,
                                cover_weight,
                            )
            appearance_weights = (0.0, *appearance_best[1:])
            cal_selective = classwise_blend(
                cal_learned,
                cal_structured,
                appearance_weights,
            )
            selective_best = (-float("inf"), 0.5, 0.5, 0.0)
            for candidate_boundary in (0.0, 0.25, 0.5, 0.75, 1.0):
                for candidate_insertion in (0.0, 0.25, 0.5, 0.75, 1.0):
                    mixed = stage_shared_blend(
                        cal_selective,
                        cal_rule,
                        boundary_weight=candidate_boundary,
                        insertion_weight=candidate_insertion,
                    )
                    for candidate_decay in (0.0, 0.25, 0.50, 0.75):
                        probabilities = causal_filter(mixed, cal_samples, candidate_decay)
                        score = proposal_score(probabilities, cal_samples)
                        candidate = (
                            score,
                            -abs(candidate_boundary - candidate_insertion),
                            -candidate_decay,
                        )
                        if candidate > (
                            selective_best[0],
                            -abs(selective_best[1] - selective_best[2]),
                            -selective_best[3],
                        ):
                            selective_best = (
                                score,
                                float(candidate_boundary),
                                float(candidate_insertion),
                                float(candidate_decay),
                            )
            (
                _,
                selective_boundary_weight,
                selective_insertion_weight,
                selective_decay,
            ) = selective_best
        selected[f"fold_{fold}"] = {
            "learned_weight": alpha,
            "causal_decay": decay,
            "stage_shared_boundary_weight": boundary_weight,
            "stage_shared_insertion_weight": insertion_weight,
            "stage_shared_causal_decay": shared_decay,
            "appearance_weights_by_step": {
                step: appearance_weights[index] for index, step in enumerate(STEPS)
            },
            "selective_boundary_weight": selective_boundary_weight,
            "selective_insertion_weight": selective_insertion_weight,
            "selective_causal_decay": selective_decay,
            "cascade_confidence_threshold": cascade_threshold,
            "cascade_force_insertion_competition": cascade_force_insertion,
            "cascade_calibration_score": cascade_calibration_score,
            "cascade_full_appearance_score": cascade_full_score,
        }

        estimator = model()
        estimator.fit(
            np.asarray([samples[index].features for index in train_index]),
            labels[train_index],
            sample_weight=balanced_weights(labels[train_index]),
        )
        test_samples = [samples[index] for index in test_index]
        test_learned = aligned_probabilities(
            estimator, np.asarray([sample.features for sample in test_samples])
        )
        structured_estimator = model()
        structured_estimator.fit(
            np.asarray(
                [structured_feature_vector(samples[index], appearance_dim) for index in train_index]
            ),
            labels[train_index],
            sample_weight=balanced_weights(labels[train_index]),
        )
        test_structured = aligned_probabilities(
            structured_estimator,
            np.asarray(
                [structured_feature_vector(sample, appearance_dim) for sample in test_samples]
            ),
        )
        if appearance_dim > 0:
            appearance_estimator = model()
            appearance_estimator.fit(
                np.asarray(
                    [
                        appearance_event_feature_vector(samples[index], appearance_dim)
                        for index in train_index
                    ]
                ),
                labels[train_index],
                sample_weight=balanced_weights(labels[train_index]),
            )
            appearance_event[test_index] = aligned_probabilities(
                appearance_estimator,
                np.asarray(
                    [
                        appearance_event_feature_vector(sample, appearance_dim)
                        for sample in test_samples
                    ]
                ),
            )
        test_rule = rule_probabilities(test_samples)
        learned[test_index] = test_learned
        structured[test_index] = test_structured
        fused[test_index] = causal_filter(
            alpha * test_learned + (1.0 - alpha) * test_rule,
            test_samples,
            decay,
        )
        stage_shared[test_index] = causal_filter(
            stage_shared_blend(
                test_learned,
                test_rule,
                boundary_weight=boundary_weight,
                insertion_weight=insertion_weight,
            ),
            test_samples,
            shared_decay,
        )
        selective_learned = classwise_blend(
            test_learned,
            test_structured,
            appearance_weights,
        )
        selective_appearance[test_index] = causal_filter(
            stage_shared_blend(
                selective_learned,
                test_rule,
                boundary_weight=selective_boundary_weight,
                insertion_weight=selective_insertion_weight,
            ),
            test_samples,
            selective_decay,
        )
        if appearance_dim > 0:
            test_route = appearance_route_mask(
                test_structured,
                confidence_threshold=cascade_threshold,
                force_insertion_competition=cascade_force_insertion,
            )
            test_cascade = np.where(test_route[:, None], test_learned, test_structured)
            epistemic_cascade[test_index] = causal_filter(
                alpha * test_cascade + (1.0 - alpha) * test_rule,
                test_samples,
                decay,
            )
            cascade_invoked[test_index] = test_route
    result = {
        "Rule Evidence Prior": summarize(rule_probabilities(samples), samples),
        "Learned Evidence Proposal": summarize(learned, samples),
        "Causal Context Fusion": summarize(fused, samples),
        "Stage-Shared Causal Fusion": summarize(stage_shared, samples),
        "selected_by_fold": selected,
    }
    if appearance_dim > 0:
        result["Structured Evidence Proposal"] = summarize(structured, samples)
        result["Appearance + Causal Event Proposal"] = summarize(
            appearance_event,
            samples,
        )
        result["Stage-Selective Appearance Fusion"] = summarize(
            selective_appearance,
            samples,
        )
        result["Epistemic Appearance Cascade"] = summarize(
            epistemic_cascade,
            samples,
        )
        result["Epistemic Appearance Cascade"]["appearance_invocations"] = int(
            cascade_invoked.sum()
        )
        result["Epistemic Appearance Cascade"]["appearance_invocation_rate"] = float(
            np.mean(cascade_invoked)
        )
    return result, {
        "causal_context_fusion": fused,
        "fold_id": fold_ids,
    }


def evaluate_causal_only(
    samples: Sequence[Sample],
    folds: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate only the proposal used by downstream claim verification."""

    if not samples:
        raise ValueError("No replay frames matched the timeline.")
    labels = np.asarray(
        [STEP_INDEX[sample.target] for sample in samples],
        dtype=np.int64,
    )
    groups = np.asarray([scenario_group(sample.video) for sample in samples])
    folds = min(int(folds), len(np.unique(groups)))
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=20260828,
    )
    learned = np.zeros((len(samples), len(STEPS)), dtype=np.float64)
    fused = np.zeros((len(samples), len(STEPS)), dtype=np.float64)
    fold_ids = np.zeros(len(samples), dtype=np.int32)
    selected: dict[str, dict[str, float]] = {}
    for fold, (train_index, test_index) in enumerate(
        splitter.split(np.zeros(len(samples)), labels, groups),
        start=1,
    ):
        fold_ids[test_index] = fold
        held_calibration = calibration_groups(groups[train_index])
        fit_index = np.asarray(
            [index for index in train_index if groups[index] not in held_calibration]
        )
        calibration_index = np.asarray(
            [index for index in train_index if groups[index] in held_calibration]
        )
        estimator = model()
        estimator.fit(
            np.asarray([samples[index].features for index in fit_index]),
            labels[fit_index],
            sample_weight=balanced_weights(labels[fit_index]),
        )
        calibration_samples = [samples[index] for index in calibration_index]
        calibration_learned = aligned_probabilities(
            estimator,
            np.asarray([sample.features for sample in calibration_samples]),
        )
        calibration_rule = rule_probabilities(calibration_samples)
        best = (-float("inf"), 1.0, 0.0)
        for alpha in (0.50, 0.75, 1.0):
            for decay in (0.0, 0.25, 0.50, 0.75):
                probabilities = causal_filter(
                    alpha * calibration_learned
                    + (1.0 - alpha) * calibration_rule,
                    calibration_samples,
                    decay,
                )
                candidate = (
                    proposal_score(probabilities, calibration_samples),
                    alpha,
                    -decay,
                )
                if candidate > best:
                    best = candidate
        alpha, decay = best[1], -best[2]
        selected[f"fold_{fold}"] = {
            "learned_weight": float(alpha),
            "causal_decay": float(decay),
        }

        estimator = model()
        estimator.fit(
            np.asarray([samples[index].features for index in train_index]),
            labels[train_index],
            sample_weight=balanced_weights(labels[train_index]),
        )
        test_samples = [samples[index] for index in test_index]
        test_learned = aligned_probabilities(
            estimator,
            np.asarray([sample.features for sample in test_samples]),
        )
        test_rule = rule_probabilities(test_samples)
        learned[test_index] = test_learned
        fused[test_index] = causal_filter(
            alpha * test_learned + (1.0 - alpha) * test_rule,
            test_samples,
            decay,
        )
    return {
        "Learned Evidence Proposal": summarize(learned, samples),
        "Causal Context Fusion": summarize(fused, samples),
        "selected_by_fold": selected,
    }, {
        "learned_evidence_proposal": learned,
        "causal_context_fusion": fused,
        "fold_id": fold_ids,
    }


def write_cross_fitted_predictions(
    path: Path,
    samples: Sequence[Sample],
    probabilities: np.ndarray,
    fold_ids: np.ndarray,
) -> None:
    """Write label-free held-out proposal probabilities for downstream replay."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample, row, fold in zip(samples, probabilities, fold_ids):
            order = np.argsort(-row)
            record = {
                "video": video_key(sample.video),
                "frame": int(sample.frame),
                "fold": int(fold),
                "proposed_step": STEPS[int(order[0])],
                "runner_up": STEPS[int(order[1])],
                "confidence": float(row[int(order[0])]),
                "scores": {
                    step: float(row[index]) for index, step in enumerate(STEPS)
                },
                "source": "cross_fitted_causal_context_fusion",
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--timeline-csv", type=Path, required=True)
    parser.add_argument("--kb", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--predictions-jsonl",
        type=Path,
        help="Optional label-free held-out proposal cache for end-to-end replay.",
    )
    parser.add_argument("--embedding-npz", type=Path)
    parser.add_argument(
        "--causal-appearance-history",
        action="store_true",
        help="Append short/long past-only frozen appearance state and innovation.",
    )
    parser.add_argument(
        "--appearance-cache-stride",
        type=int,
        default=1,
        help="Past-only DINO embedding refresh stride; detector/ROI metadata remains current.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--causal-only",
        action="store_true",
        help="Evaluate only the causal proposal consumed by the verifier.",
    )
    parser.add_argument(
        "--prediction-source",
        choices=("causal", "learned"),
        default="causal",
        help="Cross-fitted posterior written to --predictions-jsonl.",
    )
    args = parser.parse_args()
    samples, appearance_names = build_samples(
        args.summary_csv,
        args.timeline_csv,
        args.kb,
        args.embedding_npz,
        causal_appearance_history=bool(args.causal_appearance_history),
        appearance_cache_stride=max(1, int(args.appearance_cache_stride)),
    )
    if args.causal_only:
        results, prediction_arrays = evaluate_causal_only(samples, args.folds)
    else:
        results, prediction_arrays = evaluate(
            samples,
            args.folds,
            len(appearance_names),
        )
    payload = {
        "protocol": {
            "split": f"{args.folds}-fold stratified scenario-grouped cross-validation",
            "frames": len(samples),
            "videos": len({sample.video for sample in samples}),
            "scenario_groups": len({scenario_group(sample.video) for sample in samples}),
            "future_frames_used": False,
            "test_labels_used_for_fitting_or_calibration": False,
            "ground_truth_boxes_used": False,
            "filename_or_outcome_features_used": False,
            "causal_appearance_history": bool(args.causal_appearance_history),
            "appearance_cache_stride": max(1, int(args.appearance_cache_stride)),
            "nominal_appearance_invocation_rate": 1.0
            / max(1, int(args.appearance_cache_stride)),
            "input_fingerprints": {
                "summary_csv": file_fingerprint(args.summary_csv),
                "timeline_csv": file_fingerprint(args.timeline_csv),
                "knowledge_base": file_fingerprint(args.kb),
                "appearance_embeddings": (
                    file_fingerprint(args.embedding_npz)
                    if args.embedding_npz is not None
                    else None
                ),
            },
            "feature_names": (
                base_feature_names()
                + appearance_names
                + feature_names()[len(base_feature_names()) :]
            ),
            "frozen_appearance_encoder": (
                "DINOv2 ViT-B/14 predicted-box ROI, fixed 64D projection"
                if args.embedding_npz is not None
                else None
            ),
        },
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.predictions_jsonl is not None:
        prediction_key = (
            "learned_evidence_proposal"
            if args.prediction_source == "learned"
            else "causal_context_fusion"
        )
        write_cross_fitted_predictions(
            args.predictions_jsonl,
            samples,
            prediction_arrays[prediction_key],
            prediction_arrays["fold_id"],
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
