"""Evaluate a relation-aware claim verifier with video-grouped predictions.

The evaluator consumes only replay-time evidence available to the assistant.
For every held-out video group, model fitting and operating-point calibration
use other videos. No filename, outcome metadata, or future frame enters
features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_procedural_claim_triage_calibrated import (  # noqa: E402
    OUTCOMES,
    group_timeline,
    read_csv,
    summarize_predictions,
    timeline_at,
    video_key,
)
from inspect_system.evidence_scorer import (  # noqa: E402
    FEATURE_NAMES as BOX_FEATURE_NAMES,
    EXPECTED_BY_PRODUCT_ROLE,
    extract_relation_features,
    expected_class,
    normalize_claim,
    target_role_for,
)


CLASSES = tuple(sorted(set(EXPECTED_BY_PRODUCT_ROLE.values())))
PREDICATES = (
    "inside",
    "aligned_with",
    "contacting",
    "near",
    "overlapping",
    "supporting",
    "supported_by",
    "in_front_of",
    "behind",
    "left_of",
    "right_of",
    "above",
    "below",
)
CLAIMS = ("small_gear_inserted", "big_gear_inserted", "cover_seated")
PRODUCTS = ("A", "B", "UNKNOWN")
LABEL_TO_INDEX = {label: index for index, label in enumerate(OUTCOMES)}
FORBIDDEN_PROPOSAL_CACHE_KEYS = {
    "target",
    "truth",
    "label",
    "outcome",
    "outcome_norm",
    "step_id",
}
PROPOSAL_FEATURE_VARIANTS = {
    "proposal_no_consistency",
    "proposal_no_margin",
    "proposal_context",
    "relations",
    "calibrated_fusion",
    "calibrated_fusion_no_consistency",
    "causal",
    "appearance",
    "appearance_no_consistency",
    "relations_appearance",
    "causal_appearance",
    "causal_appearance_detection_only_no_consistency",
    "causal_appearance_no_consistency",
    "causal_appearance_no_margin",
    "causal_appearance_counterfactual",
    "causal_appearance_counterfactual_no_consistency",
}
SUPPORT_CONFIRM_VARIANTS = {
    "confirm_visual_appearance_no_consistency",
    "confirm3_visual_appearance_no_consistency",
}
LABEL_DERIVED_FEEDBACK_SOURCES = {
    "simulated",
    "timeline_gt",
    "gt_timeline",
    "offline_replay",
}


@dataclass(frozen=True)
class Sample:
    video: str
    frame: int
    truth: str
    proposal_consistent: bool
    detection: tuple[float, ...]
    relation: tuple[float, ...]
    proposal: tuple[float, ...]
    causal: tuple[float, ...]
    appearance: tuple[float, ...]
    counterfactual_detection: tuple[float, ...]
    counterfactual_relation: tuple[float, ...]
    has_specialized_counterfactual: bool
    claim_id: str = ""
    step_id: str = ""
    product: str = ""


class HierarchicalTriageClassifier:
    """Factor triage into evidence sufficiency and resolved-claim polarity."""

    classes_ = np.arange(len(OUTCOMES), dtype=np.int64)

    @staticmethod
    def _model() -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=160,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=2.0,
            random_state=20260828,
        )

    @staticmethod
    def _balanced_weights(labels: np.ndarray) -> np.ndarray:
        counts = Counter(int(value) for value in labels)
        return np.asarray(
            [len(labels) / (len(counts) * counts[int(value)]) for value in labels],
            dtype=np.float64,
        )

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "HierarchicalTriageClassifier":
        del sample_weight
        unresolved = LABEL_TO_INDEX["unresolved"]
        supported = LABEL_TO_INDEX["supported"]
        resolved_labels = (labels != unresolved).astype(np.int64)
        self.resolution_model = self._model()
        self.resolution_model.fit(
            features,
            resolved_labels,
            sample_weight=self._balanced_weights(resolved_labels),
        )
        resolved_mask = resolved_labels.astype(bool)
        polarity_labels = (labels[resolved_mask] == supported).astype(np.int64)
        self.polarity_model = self._model()
        self.polarity_model.fit(
            features[resolved_mask],
            polarity_labels,
            sample_weight=self._balanced_weights(polarity_labels),
        )
        return self

    @staticmethod
    def _positive_probability(
        model: HistGradientBoostingClassifier,
        features: np.ndarray,
    ) -> np.ndarray:
        probabilities = model.predict_proba(features)
        positive = np.where(model.classes_ == 1)[0]
        if len(positive) != 1:
            return np.zeros((len(features),), dtype=np.float64)
        return probabilities[:, int(positive[0])]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        resolved = self._positive_probability(self.resolution_model, features)
        supported_given_resolved = self._positive_probability(
            self.polarity_model, features
        )
        result = np.zeros((len(features), len(OUTCOMES)), dtype=np.float64)
        result[:, LABEL_TO_INDEX["supported"]] = (
            resolved * supported_given_resolved
        )
        result[:, LABEL_TO_INDEX["contradicted"]] = (
            resolved * (1.0 - supported_given_resolved)
        )
        result[:, LABEL_TO_INDEX["unresolved"]] = 1.0 - resolved
        return result


class OneVsRestTriageClassifier:
    """Estimate each epistemic state with an independently balanced head."""

    classes_ = np.arange(len(OUTCOMES), dtype=np.int64)

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "OneVsRestTriageClassifier":
        del sample_weight
        self.models = []
        for class_index in self.classes_:
            binary = (labels == int(class_index)).astype(np.int64)
            model = HierarchicalTriageClassifier._model()
            model.fit(
                features,
                binary,
                sample_weight=HierarchicalTriageClassifier._balanced_weights(binary),
            )
            self.models.append(model)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        scores = np.column_stack(
            [
                HierarchicalTriageClassifier._positive_probability(model, features)
                for model in self.models
            ]
        )
        return scores / np.maximum(scores.sum(axis=1, keepdims=True), 1e-8)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def audit_recorded_proposal_provenance(
    summary_csv: Path,
    variants: Sequence[str],
    proposal_predictions_jsonl: Path | None,
) -> dict[str, Any]:
    """Reject recorded proposal features that were updated from test labels."""

    uses_recorded_proposals = (
        proposal_predictions_jsonl is None
        and any(variant in PROPOSAL_FEATURE_VARIANTS for variant in variants)
    )
    report: dict[str, Any] = {
        "uses_recorded_proposals": uses_recorded_proposals,
        "runs_checked": 0,
        "feedback_events_checked": 0,
        "label_derived_feedback_events": 0,
    }
    if not uses_recorded_proposals:
        return report

    contaminated: list[str] = []
    for run in read_csv(summary_csv):
        if str(run.get("returncode", "")).strip() not in {"", "0"}:
            continue
        run_dir = Path(str(run.get("run_dir", "")))
        report["runs_checked"] += 1
        for record in iter_jsonl(run_dir / "feedback.jsonl"):
            report["feedback_events_checked"] += 1
            feedback = record.get("feedback") or record
            source = str(feedback.get("source", "")).strip().lower().replace("-", "_")
            extras = feedback.get("extras") or {}
            if source in LABEL_DERIVED_FEEDBACK_SOURCES or bool(
                extras.get("simulated_feedback", False)
            ):
                report["label_derived_feedback_events"] += 1
                if len(contaminated) < 5:
                    contaminated.append(
                        f"{run_dir.name}:frame={record.get('frame_index', -1)}:source={source}"
                    )
    if contaminated:
        preview = ", ".join(contaminated)
        raise RuntimeError(
            "Recorded proposal features are label-contaminated: replay feedback "
            "contains timeline-derived or simulated labels. Re-run the assistant "
            "with --feedback-mode none, or provide a label-free held-out proposal "
            f"cache via --proposal-predictions-jsonl. First events: {preview}"
        )
    return report


def _confidence(detection: Mapping[str, Any]) -> float:
    return float(detection.get("confidence", detection.get("conf", 0.0)) or 0.0)


def _class_summary(detections: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    features: dict[str, float] = {}
    for class_name in CLASSES:
        rows = [item for item in detections if str(item.get("name", "")) == class_name]
        features[f"class_conf:{class_name}"] = max((_confidence(item) for item in rows), default=0.0)
        features[f"class_count:{class_name}"] = min(3.0, float(len(rows))) / 3.0
    return features


def _product_key(value: object) -> str:
    product = str(value or "").strip().upper()
    return product if product in {"A", "B"} else "UNKNOWN"


def _context_features(claim: str, product: str) -> dict[str, float]:
    result = {f"claim:{name}": float(claim == name) for name in CLAIMS}
    result.update({f"product:{name}": float(product == name) for name in PRODUCTS})
    return result


def _relation_features(
    item: Mapping[str, Any],
    *,
    claim: str,
    step: str,
    product: str,
) -> dict[str, float]:
    role = target_role_for(claim, step)
    target_names = {
        expected_class(product, role)
    } if product in {"A", "B"} else {
        expected_class(candidate, role) for candidate in ("A", "B")
    }
    target_names.discard("")
    housing_names = {
        expected_class(product, "housing")
    } if product in {"A", "B"} else {
        expected_class(candidate, "housing") for candidate in ("A", "B")
    }
    housing_names.discard("")
    result = {f"target_housing:{predicate}": 0.0 for predicate in PREDICATES}
    result.update({f"role_pair:{predicate}": 0.0 for predicate in PREDICATES})
    result.update({f"scene_count:{predicate}": 0.0 for predicate in PREDICATES})
    if role == "small_gear":
        role_targets = {"type_3_gear", "type_7_gear"}
        role_context = {
            "type_5_gearbox_housing",
            "type_6_gearbox_housing",
        }
    elif role == "big_gear":
        role_targets = {"type_2_gear", "type_8_gear"}
        role_context = {
            "type_5_gearbox_housing",
            "type_6_gearbox_housing",
        }
    elif role == "cover":
        role_targets = {
            "type_5_gearbox_cover",
            "type_6_gearbox_cover",
        }
        role_context = {
            "type_5_gearbox_housing",
            "type_6_gearbox_housing",
        }
    else:
        role_targets = set(target_names)
        role_context = set(housing_names)
    relations = item.get("scene_graph_relations") or []
    for relation in relations:
        if not isinstance(relation, Mapping):
            continue
        predicate = str(relation.get("predicate", ""))
        if predicate not in PREDICATES:
            continue
        score = max(0.0, min(1.0, float(relation.get("score", 0.0) or 0.0)))
        result[f"scene_count:{predicate}"] = min(
            1.0, result[f"scene_count:{predicate}"] + 0.25
        )
        subject = str(relation.get("subject_name", relation.get("subject", "")))
        obj = str(relation.get("object_name", relation.get("object", "")))
        target_pair = (
            subject in target_names and obj in housing_names
        ) or (
            obj in target_names and subject in housing_names
        )
        if target_pair:
            result[f"target_housing:{predicate}"] = max(
                result[f"target_housing:{predicate}"], score
            )
        robust_pair = (
            subject in role_targets and obj in role_context
        ) or (
            obj in role_targets and subject in role_context
        )
        if robust_pair:
            result[f"role_pair:{predicate}"] = max(
                result[f"role_pair:{predicate}"], score
            )
    stats = item.get("scene_graph_stats") or {}
    result["scene_relation_density"] = min(
        1.0, float(stats.get("num_relations", 0.0) or 0.0) / 20.0
    )
    result["scene_relation_confidence"] = max(
        0.0, min(1.0, float(stats.get("avg_relation_score", 0.0) or 0.0))
    )
    result["scene_focus_density"] = min(
        1.0, float(stats.get("focus_relations", 0.0) or 0.0) / 12.0
    )
    return result


def load_proposal_predictions(
    path: Path | None,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Load held-out proposal scores and reject caches containing labels."""

    if path is None:
        return {}
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for record in iter_jsonl(path):
        leaked = FORBIDDEN_PROPOSAL_CACHE_KEYS.intersection(record)
        if leaked:
            names = ", ".join(sorted(leaked))
            raise ValueError(f"Proposal cache contains forbidden label fields: {names}")
        video = video_key(record.get("video", ""))
        frame = int(record.get("frame", -1))
        scores = dict(record.get("scores") or {})
        if not video or frame < 0 or not scores:
            raise ValueError("Proposal cache rows require video, frame, and scores")
        key = (video, frame)
        if key in output:
            raise ValueError(f"Duplicate proposal cache row: {video}:{frame}")
        output[key] = dict(record)
    if not output:
        raise ValueError(f"Proposal cache is empty: {path}")
    return output


def _proposal_features(
    item: Mapping[str, Any],
    step: str,
    proposal_record: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    scores = (proposal_record or {}).get("scores") or item.get("fusion_scores") or {}
    target_score = float(scores.get(step, 0.0) or 0.0)
    ranked_scores = sorted((float(value) for value in scores.values()), reverse=True)
    confidence = float(
        (proposal_record or {}).get(
            "confidence",
            item.get("fused_conf", ranked_scores[0] if ranked_scores else 0.0),
        )
        or 0.0
    )
    margin = ranked_scores[0] - ranked_scores[1] if len(ranked_scores) >= 2 else confidence
    return {
        "proposal_target_score": target_score,
        "proposal_confidence": confidence,
        "proposal_margin": float(margin),
        "proposal_disagreement": (
            0.0
            if proposal_record is not None
            else float(item.get("expert_disagreement", 0.0) or 0.0)
        ),
        "visual_evidence_present": float(bool(item.get("has_visual_evidence", False))),
        "stable_tracks": min(
            1.0,
            sum(float(value) for value in ((item.get("evidence_token") or {}).get("stable_track_counts") or {}).values()) / 4.0,
        ),
    }


def _vector(mapping: Mapping[str, float], names: Sequence[str]) -> tuple[float, ...]:
    return tuple(float(mapping.get(name, 0.0)) for name in names)


def load_appearance_embeddings(
    path: Path | None,
) -> tuple[dict[tuple[str, int], tuple[float, ...]], list[str]]:
    if path is None:
        return {}, []
    payload = np.load(path, allow_pickle=False)
    roi_embeddings = np.asarray(payload["roi_embedding"], dtype=np.float32)
    projection_dim = min(64, roi_embeddings.shape[1])
    rng = np.random.default_rng(20260828)
    projection = rng.standard_normal(
        (roi_embeddings.shape[1], projection_dim),
        dtype=np.float32,
    ) / np.sqrt(float(projection_dim))
    roi_embeddings = roi_embeddings @ projection
    roi_embeddings /= np.maximum(
        np.linalg.norm(roi_embeddings, axis=1, keepdims=True),
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
                roi_embeddings[index],
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
        for index in range(roi_embeddings.shape[1])
    ]
    return mapping, names


def _online_proposal_steps(
    item: Mapping[str, Any],
    top_k: int,
    proposal_record: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return online proposal candidates without consulting timeline labels."""
    raw_scores = (proposal_record or {}).get("scores") or item.get("fusion_scores") or {}
    scores = {
        str(key).strip().upper(): float(value)
        for key, value in dict(raw_scores).items()
        if str(key).strip().upper() in {"S1", "S2", "S3", "S4"}
    }
    if scores:
        ranked = [
            key
            for key, _ in sorted(
                scores.items(),
                key=lambda pair: (-pair[1], pair[0]),
            )
        ]
    else:
        ranked = [
            str(item.get("fused_step", item.get("decision_step", "")))
            .strip()
            .upper()
        ]
        runner_up = str(item.get("fused_runner_up", "")).strip().upper()
        if runner_up in {"S1", "S2", "S3", "S4"} and runner_up not in ranked:
            ranked.append(runner_up)
    return [step for step in ranked if step in {"S1", "S2", "S3", "S4"}][
        : max(1, int(top_k))
    ]


def build_samples(
    summary_csv: Path,
    timeline_csv: Path,
    ema_decay: float,
    proposal_top_k: int,
    claim_scope: str,
    embedding_npz: Path | None = None,
    proposal_predictions_jsonl: Path | None = None,
) -> tuple[list[Sample], dict[str, list[str]]]:
    timelines = group_timeline(timeline_csv)
    appearance_lookup, appearance_names = load_appearance_embeddings(embedding_npz)
    proposal_lookup = load_proposal_predictions(proposal_predictions_jsonl)
    missing_proposal_rows: list[tuple[str, int]] = []
    raw_rows: dict[
        str,
        list[
            tuple[
                int,
                str,
                bool,
                dict[str, float],
                dict[str, float],
                tuple[float, ...],
                dict[str, float],
                dict[str, float],
                bool,
                str,
                str,
                str,
            ]
        ],
    ] = defaultdict(list)
    detection_names = list(BOX_FEATURE_NAMES)
    detection_names += [f"class_conf:{name}" for name in CLASSES]
    detection_names += [f"class_count:{name}" for name in CLASSES]
    detection_names += [f"claim:{name}" for name in CLAIMS]
    detection_names += [f"product:{name}" for name in PRODUCTS]
    relation_names = [f"target_housing:{name}" for name in PREDICATES]
    relation_names += [f"role_pair:{name}" for name in PREDICATES]
    relation_names += [f"scene_count:{name}" for name in PREDICATES]
    relation_names += ["scene_relation_density", "scene_relation_confidence", "scene_focus_density"]
    proposal_names = [
        "proposal_target_score",
        "proposal_confidence",
        "proposal_margin",
        "proposal_disagreement",
        "visual_evidence_present",
        "stable_tracks",
    ]

    for run in read_csv(summary_csv):
        if str(run.get("returncode", "")).strip() not in {"", "0"}:
            continue
        video = str(run.get("video", ""))
        timeline = timelines.get(video_key(video), [])
        if not timeline:
            continue
        for item in iter_jsonl(Path(str(run.get("run_dir", ""))) / "iterations.jsonl"):
            frame = int(item.get("frame_index", item.get("frame", -1)))
            trow = timeline_at(timeline, frame)
            if not trow:
                continue
            step = str(trow.get("step_id", "")).strip().upper()
            claim = normalize_claim(trow.get("claim_id", ""), step)
            if claim_scope == "assembly" and claim not in CLAIMS:
                continue
            proposal_record = proposal_lookup.get((video_key(video), frame))
            if proposal_lookup and proposal_record is None:
                missing_proposal_rows.append((video_key(video), frame))
                continue
            product = _product_key(trow.get("assembly_set", ""))
            detections = list(item.get("fused_detections") or item.get("raw_detections") or [])
            image_shape = tuple(item.get("frame_shape") or (1080, 1920))
            current = extract_relation_features(
                detections,
                claim_id=claim,
                step_id=step,
                product=product,
                image_shape=(int(image_shape[0]), int(image_shape[1])),
            )
            current.update(_class_summary(detections))
            current.update(_context_features(claim, product))
            relations = _relation_features(item, claim=claim, step=step, product=product)
            relations.update(_proposal_features(item, step, proposal_record))
            counterfactual_product = {"A": "B", "B": "A"}.get(product)
            if counterfactual_product is not None:
                counterfactual = extract_relation_features(
                    detections,
                    claim_id=claim,
                    step_id=step,
                    product=counterfactual_product,
                    image_shape=(int(image_shape[0]), int(image_shape[1])),
                )
                counterfactual.update(_class_summary(detections))
                counterfactual.update(
                    _context_features(claim, counterfactual_product)
                )
                counterfactual_relations = _relation_features(
                    item,
                    claim=claim,
                    step=step,
                    product=counterfactual_product,
                )
                counterfactual_relations.update(
                    _proposal_features(item, step, proposal_record)
                )
            else:
                counterfactual = dict(current)
                counterfactual_relations = dict(relations)
            proposal_steps = _online_proposal_steps(
                item,
                proposal_top_k,
                proposal_record,
            )
            raw_rows[video].append(
                (
                    frame,
                    str(trow.get("outcome_norm", "")),
                    step in proposal_steps,
                    current,
                    relations,
                    appearance_lookup.get(
                        (video_key(video), frame),
                        tuple(0.0 for _ in appearance_names),
                    ),
                    counterfactual,
                    counterfactual_relations,
                    counterfactual_product is not None,
                    claim,
                    step,
                    product,
                )
            )

    if missing_proposal_rows:
        preview = ", ".join(
            f"{video}:{frame}" for video, frame in missing_proposal_rows[:5]
        )
        raise RuntimeError(
            f"Cross-fitted proposal cache is missing {len(missing_proposal_rows)} "
            f"evaluated frames; first rows: {preview}"
        )

    samples: list[Sample] = []
    causal_source_names = (
        "target_conf",
        "wrong_same_role_conf",
        "identity_margin",
        "containment_score",
        "intersection_over_target",
        "center_distance_norm",
        "target_housing:inside",
        "target_housing:aligned_with",
        "target_housing:contacting",
        "target_housing:overlapping",
        "scene_relation_confidence",
        "proposal_target_score",
    )
    causal_names = [f"ema:{name}" for name in causal_source_names]
    causal_names += [f"delta:{name}" for name in causal_source_names]
    for video, rows in raw_rows.items():
        ema: dict[str, float] = {}
        previous: dict[str, float] = {}
        for (
            frame,
            truth,
            proposal_consistent,
            detection,
            relation,
            appearance,
            counterfactual_detection,
            counterfactual_relation,
            has_specialized_counterfactual,
            claim,
            step,
            product,
        ) in sorted(rows, key=lambda row: row[0]):
            merged = {**detection, **relation}
            causal: dict[str, float] = {}
            for name in causal_source_names:
                value = float(merged.get(name, 0.0))
                old_ema = ema.get(name, value)
                causal[f"ema:{name}"] = old_ema
                causal[f"delta:{name}"] = value - previous.get(name, value)
                ema[name] = float(ema_decay) * old_ema + (1.0 - float(ema_decay)) * value
                previous[name] = value
            samples.append(
                Sample(
                    video=video,
                    frame=frame,
                    truth=truth,
                    proposal_consistent=proposal_consistent,
                    detection=_vector(detection, detection_names),
                    relation=_vector(relation, relation_names),
                    proposal=_vector(relation, proposal_names),
                    causal=_vector(causal, causal_names),
                    appearance=appearance,
                    counterfactual_detection=_vector(
                        counterfactual_detection, detection_names
                    ),
                    counterfactual_relation=_vector(
                        counterfactual_relation, relation_names
                    ),
                    has_specialized_counterfactual=has_specialized_counterfactual,
                    claim_id=claim,
                    step_id=step,
                    product=product,
                )
            )
    return samples, {
        "detection": detection_names,
        "scene_relation": relation_names,
        "proposal": proposal_names,
        "causal": ["causal posterior filter; decay selected on calibration groups"],
        "appearance": appearance_names,
    }


def feature_vector(sample: Sample, variant: str) -> np.ndarray:
    if variant in {
        "confirm_visual_appearance_no_consistency",
        "confirm3_visual_appearance_no_consistency",
        "causal_visual_appearance_no_consistency",
        "causal_visual_appearance_counterfactual_no_consistency",
        "hierarchical_visual_appearance_no_consistency",
        "causal_hierarchical_visual_appearance_no_consistency",
        "ovr_visual_appearance_no_consistency",
        "causal_ovr_visual_appearance_no_consistency",
    }:
        variant = "visual_appearance_no_consistency"
    if variant in {"proposal_no_consistency", "proposal_no_margin"}:
        variant = "proposal_context"
    if variant in {
        "causal_appearance_no_consistency",
        "causal_appearance_no_margin",
        "causal_appearance_counterfactual",
        "causal_appearance_counterfactual_no_consistency",
    }:
        variant = "causal_appearance"
    if variant in {
        "appearance_no_consistency",
        "causal_appearance_detection_only_no_consistency",
    }:
        variant = "appearance"
    values = list(sample.detection)
    if variant in {
        "scene_relations",
        "relations",
        "causal",
        "relations_appearance",
        "causal_appearance",
        "visual_appearance_no_consistency",
    }:
        values.extend(sample.relation)
    if variant in {
        "proposal_context",
        "relations",
        "causal",
        "appearance",
        "relations_appearance",
        "causal_appearance",
    }:
        values.extend(sample.proposal)
    if variant in {
        "appearance",
        "relations_appearance",
        "causal_appearance",
        "visual_appearance_no_consistency",
        "visual_appearance_no_relations_no_consistency",
    }:
        values.extend(sample.appearance)
    return np.asarray(values, dtype=np.float64)


def counterfactual_feature_vector(sample: Sample, variant: str) -> np.ndarray:
    """Re-evaluate the observation under the confusable product-family claim."""
    if variant in {
        "causal_visual_appearance_no_consistency",
        "causal_visual_appearance_counterfactual_no_consistency",
        "hierarchical_visual_appearance_no_consistency",
        "causal_hierarchical_visual_appearance_no_consistency",
        "ovr_visual_appearance_no_consistency",
        "causal_ovr_visual_appearance_no_consistency",
    }:
        variant = "visual_appearance_no_consistency"
    if variant in {
        "causal_appearance_counterfactual",
        "causal_appearance_counterfactual_no_consistency",
    }:
        variant = "causal_appearance"
    values = list(sample.counterfactual_detection)
    if variant in {
        "scene_relations",
        "relations",
        "causal",
        "relations_appearance",
        "causal_appearance",
        "visual_appearance_no_consistency",
    }:
        values.extend(sample.counterfactual_relation)
    if variant in {
        "proposal_context",
        "relations",
        "causal",
        "appearance",
        "relations_appearance",
        "causal_appearance",
    }:
        values.extend(sample.proposal)
    if variant in {
        "appearance",
        "relations_appearance",
        "causal_appearance",
        "visual_appearance_no_consistency",
    }:
        values.extend(sample.appearance)
    return np.asarray(values, dtype=np.float64)


def make_model(y: np.ndarray, variant: str) -> Any:
    if "hierarchical" in variant:
        return HierarchicalTriageClassifier()
    if "ovr_" in variant:
        return OneVsRestTriageClassifier()
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=160,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=2.0,
        random_state=20260828,
    )


def sample_weights(y: np.ndarray) -> np.ndarray:
    counts = Counter(int(value) for value in y)
    return np.asarray([len(y) / (len(counts) * counts[int(value)]) for value in y])


def align_probabilities(model: Any, probabilities: np.ndarray) -> np.ndarray:
    aligned = np.zeros((probabilities.shape[0], len(OUTCOMES)), dtype=np.float64)
    for column, class_index in enumerate(model.classes_):
        aligned[:, int(class_index)] = probabilities[:, column]
    return aligned


def grouped_oof_probabilities(
    samples: Sequence[Sample],
    variant: str,
    group_for_video: Mapping[str, str],
    inner_folds: int,
    random_state: int,
    *,
    counterfactual: bool = False,
) -> np.ndarray:
    """Generate calibration posteriors without fitting on the scored group."""
    labels = np.asarray(
        [LABEL_TO_INDEX[sample.truth] for sample in samples],
        dtype=np.int64,
    )
    groups = np.asarray([group_for_video[sample.video] for sample in samples])
    unique_groups = set(groups.tolist())
    folds = min(int(inner_folds), len(unique_groups))
    if folds < 2:
        raise ValueError("grouped OOF calibration requires at least two groups")
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    probabilities = np.zeros((len(samples), len(OUTCOMES)), dtype=np.float64)
    covered = np.zeros(len(samples), dtype=bool)
    dummy = np.zeros(len(samples), dtype=np.float32)
    for fit_indices, validation_indices in splitter.split(dummy, labels, groups=groups):
        fit_rows = [samples[index] for index in fit_indices]
        validation_rows = [samples[index] for index in validation_indices]
        y_fit = labels[fit_indices]
        model = make_model(y_fit, variant)
        model.fit(
            np.stack([feature_vector(sample, variant) for sample in fit_rows]),
            y_fit,
            sample_weight=sample_weights(y_fit),
        )
        validation_probabilities = align_probabilities(
            model,
            model.predict_proba(
                np.stack(
                    [
                        (
                            counterfactual_feature_vector(sample, variant)
                            if counterfactual
                            else feature_vector(sample, variant)
                        )
                        for sample in validation_rows
                    ]
                )
            ),
        )
        probabilities[validation_indices] = validation_probabilities
        covered[validation_indices] = True
    if not bool(np.all(covered)):
        raise RuntimeError("grouped OOF calibration did not cover every training sample")
    return probabilities


def decide(
    probabilities: np.ndarray,
    support_threshold: float,
    contradiction_threshold: float,
    margin: float,
    proposal_consistent: bool,
    *,
    require_proposal_consistency: bool = True,
) -> str:
    support = float(probabilities[LABEL_TO_INDEX["supported"]])
    contradiction = float(probabilities[LABEL_TO_INDEX["contradicted"]])
    insufficient = float(probabilities[LABEL_TO_INDEX["unresolved"]])
    if contradiction >= contradiction_threshold and contradiction >= max(support, insufficient) + margin:
        return "contradicted"
    if (
        support >= support_threshold
        and support >= max(contradiction, insufficient) + margin
        and (proposal_consistent or not require_proposal_consistency)
    ):
        return "supported"
    return "unresolved"


def decide_counterfactual(
    probabilities: np.ndarray,
    counterfactual_probabilities: np.ndarray,
    support_threshold: float,
    contradiction_threshold: float,
    posterior_margin: float,
    counterfactual_margin: float,
    sample: Sample,
    *,
    require_proposal_consistency: bool = False,
) -> str:
    base = decide(
        probabilities,
        support_threshold,
        contradiction_threshold,
        posterior_margin,
        sample.proposal_consistent,
        require_proposal_consistency=require_proposal_consistency,
    )
    if base != "supported" or not sample.has_specialized_counterfactual:
        return base
    active_support = float(probabilities[LABEL_TO_INDEX["supported"]])
    alternative_support = float(
        counterfactual_probabilities[LABEL_TO_INDEX["supported"]]
    )
    if active_support - alternative_support < float(counterfactual_margin):
        return "unresolved"
    return base


def choose_thresholds(
    probabilities: np.ndarray,
    samples: Sequence[Sample],
    *,
    require_proposal_consistency: bool = True,
    allow_margin: bool = True,
    false_support_cap: float = 0.03,
) -> tuple[float, float, float]:
    best = (-1e9, 0.5, 0.5, 0.0)
    for support_threshold in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
        for contradiction_threshold in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
            for margin in (
                (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
                if allow_margin
                else (0.0,)
            ):
                predictions = [
                    decide(
                        row,
                        support_threshold,
                        contradiction_threshold,
                        margin,
                        sample.proposal_consistent,
                        require_proposal_consistency=require_proposal_consistency,
                    )
                    for row, sample in zip(probabilities, samples)
                ]
                metrics = summarize_predictions(samples, predictions)
                macro = float(metrics.get("triage_macro_f1") or 0.0)
                false_support = float(metrics.get("false_accept_rate_on_non_supported") or 0.0)
                supported = float(metrics.get("support_accuracy") or 0.0)
                contradiction = float(metrics.get("contradiction_recall") or 0.0)
                score = macro + 0.12 * supported + 0.08 * contradiction
                if false_support > float(false_support_cap):
                    score -= 2.0 + 10.0 * (
                        false_support - float(false_support_cap)
                    )
                candidate = (score, support_threshold, contradiction_threshold, margin)
                if candidate > best:
                    best = candidate
    return best[1], best[2], best[3]


def choose_relation_blend(
    proposal_probabilities: np.ndarray,
    relation_probabilities: np.ndarray,
    samples: Sequence[Sample],
    *,
    require_proposal_consistency: bool = True,
    false_support_cap: float = 0.03,
) -> tuple[float, float, float, float]:
    """Select relation residual weight on calibration groups only."""
    best = (-1e9, 0.0, 0.5, 0.5, 0.0)
    for relation_weight in (0.0, 0.10, 0.25, 0.50, 0.75, 1.0):
        probabilities = (
            (1.0 - relation_weight) * proposal_probabilities
            + relation_weight * relation_probabilities
        )
        selected = choose_thresholds(
            probabilities,
            samples,
            require_proposal_consistency=require_proposal_consistency,
            false_support_cap=false_support_cap,
        )
        predictions = [
            decide(
                row,
                *selected,
                sample.proposal_consistent,
                require_proposal_consistency=require_proposal_consistency,
            )
            for row, sample in zip(probabilities, samples)
        ]
        metrics = summarize_predictions(samples, predictions)
        macro = float(metrics.get("triage_macro_f1") or 0.0)
        false_support = float(
            metrics.get("false_accept_rate_on_non_supported") or 0.0
        )
        supported = float(metrics.get("support_accuracy") or 0.0)
        contradiction = float(metrics.get("contradiction_recall") or 0.0)
        score = macro + 0.12 * supported + 0.08 * contradiction
        if false_support > float(false_support_cap):
            score -= 2.0 + 10.0 * (
                false_support - float(false_support_cap)
            )
        candidate = (
            score,
            -relation_weight,
            selected[0],
            selected[1],
            selected[2],
        )
        if candidate > best:
            best = candidate
    return -best[1], best[2], best[3], best[4]


def causal_filter_probabilities(
    probabilities: np.ndarray,
    samples: Sequence[Sample],
    decay: float,
) -> np.ndarray:
    """Filter posteriors using only past evidence for the same active claim."""
    filtered = np.zeros_like(probabilities)
    state: dict[tuple[str, str], np.ndarray] = {}
    order = sorted(
        range(len(samples)),
        key=lambda index: (video_key(samples[index].video), samples[index].frame),
    )
    for index in order:
        current = probabilities[index]
        state_key = (samples[index].video, samples[index].claim_id)
        previous = state.get(state_key)
        posterior = (
            current
            if previous is None or decay <= 0.0
            else float(decay) * previous + (1.0 - float(decay)) * current
        )
        posterior = posterior / max(float(posterior.sum()), 1e-8)
        filtered[index] = posterior
        state[state_key] = posterior
    return filtered


def support_confirmation_filter(
    probabilities: np.ndarray,
    samples: Sequence[Sample],
    confirmation_steps: int = 2,
) -> np.ndarray:
    """Require consecutive observations before increasing support belief."""
    if int(confirmation_steps) < 2:
        raise ValueError("confirmation_steps must be at least two")
    filtered = np.asarray(probabilities, dtype=np.float64).copy()
    raw_history: dict[tuple[str, str], list[np.ndarray]] = {}
    support_index = LABEL_TO_INDEX["supported"]
    insufficient_index = LABEL_TO_INDEX["unresolved"]
    order = sorted(
        range(len(samples)),
        key=lambda index: (video_key(samples[index].video), samples[index].frame),
    )
    for index in order:
        raw = np.asarray(probabilities[index], dtype=np.float64)
        state_key = (samples[index].video, samples[index].claim_id)
        history = raw_history.setdefault(state_key, [])
        confirmed_support = (
            0.0
            if len(history) < int(confirmation_steps) - 1
            else min(
                [float(raw[support_index])]
                + [
                    float(item[support_index])
                    for item in history[-(int(confirmation_steps) - 1) :]
                ]
            )
        )
        row = raw.copy()
        withheld = max(0.0, float(row[support_index]) - confirmed_support)
        row[support_index] = confirmed_support
        row[insufficient_index] += withheld
        row /= max(float(row.sum()), 1e-8)
        filtered[index] = row
        history.append(raw)
    return filtered


def choose_causal_operating_point(
    probabilities: np.ndarray,
    samples: Sequence[Sample],
    false_support_cap: float,
    *,
    require_proposal_consistency: bool = True,
    allow_margin: bool = True,
) -> tuple[float, float, float, float]:
    best = (-1e9, 0.0, 0.5, 0.5, 0.0)
    for decay in (0.0, 0.20, 0.40, 0.60, 0.75, 0.85):
        filtered = causal_filter_probabilities(probabilities, samples, decay)
        support_threshold, contradiction_threshold, margin = choose_thresholds(
            filtered,
            samples,
            require_proposal_consistency=require_proposal_consistency,
            allow_margin=allow_margin,
            false_support_cap=false_support_cap,
        )
        predictions = [
            decide(
                row,
                support_threshold,
                contradiction_threshold,
                margin,
                sample.proposal_consistent,
                require_proposal_consistency=require_proposal_consistency,
            )
            for row, sample in zip(filtered, samples)
        ]
        metrics = summarize_predictions(samples, predictions)
        macro = float(metrics.get("triage_macro_f1") or 0.0)
        false_support = float(metrics.get("false_accept_rate_on_non_supported") or 0.0)
        supported = float(metrics.get("support_accuracy") or 0.0)
        contradiction = float(metrics.get("contradiction_recall") or 0.0)
        score = macro + 0.12 * supported + 0.08 * contradiction
        if false_support > float(false_support_cap):
            score -= 2.0 + 10.0 * (
                false_support - float(false_support_cap)
            )
        candidate = (
            score,
            -float(decay),
            support_threshold,
            contradiction_threshold,
            margin,
        )
        if candidate > best:
            best = candidate
    return -best[1], best[2], best[3], best[4]


def choose_counterfactual_causal_operating_point(
    probabilities: np.ndarray,
    counterfactual_probabilities: np.ndarray,
    samples: Sequence[Sample],
    false_support_cap: float,
    *,
    require_proposal_consistency: bool = False,
) -> tuple[float, float, float, float, float]:
    """Select a true claim-vs-alternative margin on calibration groups only."""
    best = (-1e9, 0.0, 0.5, 0.5, 0.0, 0.0)
    for decay in (0.0, 0.20, 0.40, 0.60, 0.75, 0.85):
        filtered = causal_filter_probabilities(probabilities, samples, decay)
        filtered_counterfactual = causal_filter_probabilities(
            counterfactual_probabilities, samples, decay
        )
        support_threshold, contradiction_threshold, posterior_margin = choose_thresholds(
            filtered,
            samples,
            require_proposal_consistency=require_proposal_consistency,
            allow_margin=False,
            false_support_cap=false_support_cap,
        )
        for counterfactual_margin in (0.0, 0.025, 0.05, 0.10, 0.15, 0.20):
            predictions = [
                decide_counterfactual(
                    row,
                    alternative,
                    support_threshold,
                    contradiction_threshold,
                    posterior_margin,
                    counterfactual_margin,
                    sample,
                    require_proposal_consistency=require_proposal_consistency,
                )
                for row, alternative, sample in zip(
                    filtered, filtered_counterfactual, samples
                )
            ]
            metrics = summarize_predictions(samples, predictions)
            macro = float(metrics.get("triage_macro_f1") or 0.0)
            false_support = float(
                metrics.get("false_accept_rate_on_non_supported") or 0.0
            )
            supported = float(metrics.get("support_accuracy") or 0.0)
            contradiction = float(metrics.get("contradiction_recall") or 0.0)
            score = macro + 0.12 * supported + 0.08 * contradiction
            if false_support > float(false_support_cap):
                score -= 2.0 + 10.0 * (
                    false_support - float(false_support_cap)
                )
            candidate = (
                score,
                -float(decay),
                support_threshold,
                contradiction_threshold,
                posterior_margin,
                counterfactual_margin,
            )
            if candidate > best:
                best = candidate
    return -best[1], best[2], best[3], best[4], best[5]


def calibration_videos(train_videos: Sequence[str]) -> set[str]:
    ranked = sorted(
        train_videos,
        key=lambda value: hashlib.sha1(video_key(value).encode("utf-8")).hexdigest(),
    )
    count = max(3, int(round(0.2 * len(ranked))))
    return set(ranked[:count])


def scenario_group(video: str) -> str:
    """Group clips that share a physical claim/scenario without exposing it."""
    stem = Path(video).stem.lower()
    stem = re.sub(r"_\d+$", "", stem)
    stem = re.sub(r"_(supported|contradicted|unresolved)$", "", stem)
    stem = stem.replace("view_fail_", "view_")
    return stem


def evaluate_variant(
    samples: Sequence[Sample],
    variant: str,
    outer_folds: int,
    group_level: str,
    calibration_mode: str,
    inner_folds: int,
    false_support_cap: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fusion_variants = {
        "calibrated_fusion",
        "calibrated_fusion_no_consistency",
    }
    counterfactual_variants = {
        "causal_appearance_counterfactual",
        "causal_appearance_counterfactual_no_consistency",
        "causal_visual_appearance_counterfactual_no_consistency",
    }
    require_proposal_consistency = variant not in {
        "proposal_no_consistency",
        "calibrated_fusion_no_consistency",
        "appearance_no_consistency",
        "causal_appearance_detection_only_no_consistency",
        "causal_appearance_no_consistency",
        "causal_appearance_counterfactual_no_consistency",
        "visual_appearance_no_consistency",
        "visual_appearance_no_relations_no_consistency",
        "confirm_visual_appearance_no_consistency",
        "causal_visual_appearance_no_consistency",
        "causal_visual_appearance_counterfactual_no_consistency",
        "hierarchical_visual_appearance_no_consistency",
        "causal_hierarchical_visual_appearance_no_consistency",
        "ovr_visual_appearance_no_consistency",
        "causal_ovr_visual_appearance_no_consistency",
    }
    allow_margin = variant not in {
        "proposal_no_margin",
        "causal_appearance_no_margin",
    }
    group_for_video = {
        video: video if group_level == "video" else scenario_group(video)
        for video in {sample.video for sample in samples}
    }
    unique_groups = sorted(set(group_for_video.values()), key=video_key)
    if outer_folds < 2 or outer_folds > len(unique_groups):
        raise ValueError(f"outer_folds must be in [2, {len(unique_groups)}]")
    predictions: dict[tuple[str, int], str] = {}
    detail: list[dict[str, Any]] = []
    thresholds: dict[str, dict[str, Any]] = {}
    groups = np.asarray([group_for_video[sample.video] for sample in samples])
    splitter = StratifiedGroupKFold(
        n_splits=outer_folds,
        shuffle=True,
        random_state=20260828,
    )
    dummy = np.zeros(len(samples), dtype=np.float32)
    labels = np.asarray([LABEL_TO_INDEX[sample.truth] for sample in samples], dtype=np.int64)
    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(dummy, labels, groups=groups),
        start=1,
    ):
        train_groups = sorted(set(groups[train_indices]), key=video_key)
        test_groups = sorted(set(groups[test_indices]), key=video_key)
        train_rows = [samples[index] for index in train_indices]
        test_rows = [samples[index] for index in test_indices]
        if calibration_mode == "group_oof":
            calibration = set(train_groups)
            calibration_rows = train_rows
            fit_rows: list[Sample] = []
        else:
            calibration = calibration_videos(train_groups)
            fit_rows = [
                samples[index] for index in train_indices
                if groups[index] not in calibration
            ]
            calibration_rows = [
                samples[index] for index in train_indices
                if groups[index] in calibration
            ]
        relation_blend = 0.0
        cal_counterfactual_prob: np.ndarray | None = None
        counterfactual_margin = 0.0
        if variant in fusion_variants:
            calibration_probabilities = {}
            for branch in ("proposal_context", "relations"):
                if calibration_mode == "group_oof":
                    calibration_probabilities[branch] = grouped_oof_probabilities(
                        train_rows,
                        branch,
                        group_for_video,
                        inner_folds,
                        20260828 + fold_index,
                    )
                else:
                    y_fit = np.asarray(
                        [LABEL_TO_INDEX[sample.truth] for sample in fit_rows],
                        dtype=np.int64,
                    )
                    branch_model = make_model(y_fit, branch)
                    branch_model.fit(
                        np.stack(
                            [feature_vector(sample, branch) for sample in fit_rows]
                        ),
                        y_fit,
                        sample_weight=sample_weights(y_fit),
                    )
                    calibration_probabilities[branch] = align_probabilities(
                        branch_model,
                        branch_model.predict_proba(
                            np.stack(
                                [
                                    feature_vector(sample, branch)
                                    for sample in calibration_rows
                                ]
                            )
                        ),
                    )
            (
                relation_blend,
                support_threshold,
                contradiction_threshold,
                margin,
            ) = choose_relation_blend(
                calibration_probabilities["proposal_context"],
                calibration_probabilities["relations"],
                calibration_rows,
                require_proposal_consistency=require_proposal_consistency,
                false_support_cap=false_support_cap,
            )
            selected = (
                support_threshold,
                contradiction_threshold,
                margin,
            )
            causal_decay = 0.0
        else:
            if calibration_mode == "group_oof":
                cal_prob = grouped_oof_probabilities(
                    train_rows,
                    variant,
                    group_for_video,
                    inner_folds,
                    20260828 + fold_index,
                )
                if variant in counterfactual_variants:
                    cal_counterfactual_prob = grouped_oof_probabilities(
                        train_rows,
                        variant,
                        group_for_video,
                        inner_folds,
                        20260828 + fold_index,
                        counterfactual=True,
                    )
            else:
                y_fit = np.asarray(
                    [LABEL_TO_INDEX[sample.truth] for sample in fit_rows],
                    dtype=np.int64,
                )
                calibration_model = make_model(y_fit, variant)
                calibration_model.fit(
                    np.stack(
                        [feature_vector(sample, variant) for sample in fit_rows]
                    ),
                    y_fit,
                    sample_weight=sample_weights(y_fit),
                )
                cal_prob = align_probabilities(
                    calibration_model,
                    calibration_model.predict_proba(
                        np.stack(
                            [
                                feature_vector(sample, variant)
                                for sample in calibration_rows
                            ]
                        )
                    ),
                )
                if variant in counterfactual_variants:
                    cal_counterfactual_prob = align_probabilities(
                        calibration_model,
                        calibration_model.predict_proba(
                            np.stack(
                                [
                                    counterfactual_feature_vector(
                                        sample, variant
                                    )
                                    for sample in calibration_rows
                                ]
                            )
                        ),
                    )
        if variant in SUPPORT_CONFIRM_VARIANTS:
            cal_prob = support_confirmation_filter(
                cal_prob,
                calibration_rows,
                confirmation_steps=3 if variant.startswith("confirm3_") else 2,
            )
        if variant in counterfactual_variants:
            if cal_counterfactual_prob is None:
                raise RuntimeError("Counterfactual calibration scores are missing")
            (
                causal_decay,
                support_threshold,
                contradiction_threshold,
                margin,
                counterfactual_margin,
            ) = choose_counterfactual_causal_operating_point(
                cal_prob,
                cal_counterfactual_prob,
                calibration_rows,
                false_support_cap,
                require_proposal_consistency=require_proposal_consistency,
            )
            selected = (support_threshold, contradiction_threshold, margin)
        elif variant in {
            "causal",
            "causal_appearance",
            "causal_visual_appearance_no_consistency",
            "causal_hierarchical_visual_appearance_no_consistency",
            "causal_ovr_visual_appearance_no_consistency",
            "causal_appearance_detection_only_no_consistency",
            "causal_appearance_no_consistency",
            "causal_appearance_no_margin",
        }:
            causal_decay, support_threshold, contradiction_threshold, margin = (
                choose_causal_operating_point(
                    cal_prob,
                    calibration_rows,
                    false_support_cap,
                    require_proposal_consistency=require_proposal_consistency,
                    allow_margin=allow_margin,
                )
            )
            selected = (support_threshold, contradiction_threshold, margin)
        elif variant not in fusion_variants:
            causal_decay = 0.0
            selected = choose_thresholds(
                cal_prob,
                calibration_rows,
                require_proposal_consistency=require_proposal_consistency,
                allow_margin=allow_margin,
                false_support_cap=false_support_cap,
            )
        thresholds[f"fold_{fold_index}"] = {
            "test_groups": list(test_groups),
            "calibration_groups": sorted(calibration, key=video_key),
            "calibration_mode": calibration_mode,
            "support": selected[0],
            "contradiction": selected[1],
            "margin": selected[2],
            "counterfactual_margin": counterfactual_margin,
            "causal_decay": causal_decay,
            "relation_blend": relation_blend,
        }

        y_train = np.asarray([LABEL_TO_INDEX[sample.truth] for sample in train_rows], dtype=np.int64)
        if variant in fusion_variants:
            test_probabilities = {}
            for branch in ("proposal_context", "relations"):
                x_train = np.stack(
                    [feature_vector(sample, branch) for sample in train_rows]
                )
                model = make_model(y_train, branch)
                model.fit(
                    x_train,
                    y_train,
                    sample_weight=sample_weights(y_train),
                )
                x_test = np.stack(
                    [feature_vector(sample, branch) for sample in test_rows]
                )
                test_probabilities[branch] = align_probabilities(
                    model,
                    model.predict_proba(x_test),
                )
            test_prob = (
                (1.0 - relation_blend)
                * test_probabilities["proposal_context"]
                + relation_blend * test_probabilities["relations"]
            )
        else:
            x_train = np.stack(
                [feature_vector(sample, variant) for sample in train_rows]
            )
            model = make_model(y_train, variant)
            model.fit(
                x_train,
                y_train,
                sample_weight=sample_weights(y_train),
            )
            x_test = np.stack(
                [feature_vector(sample, variant) for sample in test_rows]
            )
            test_prob = align_probabilities(
                model,
                model.predict_proba(x_test),
            )
            test_counterfactual_prob = (
                align_probabilities(
                    model,
                    model.predict_proba(
                        np.stack(
                            [
                                counterfactual_feature_vector(sample, variant)
                                for sample in test_rows
                            ]
                        )
                    ),
                )
                if variant in counterfactual_variants
                else None
            )
        if variant in counterfactual_variants:
            test_prob = causal_filter_probabilities(
                test_prob, test_rows, causal_decay
            )
            if test_counterfactual_prob is None:
                raise RuntimeError("Counterfactual test scores are missing")
            test_counterfactual_prob = causal_filter_probabilities(
                test_counterfactual_prob, test_rows, causal_decay
            )
        elif variant in {
            "causal",
            "causal_appearance",
            "causal_visual_appearance_no_consistency",
            "causal_hierarchical_visual_appearance_no_consistency",
            "causal_ovr_visual_appearance_no_consistency",
            "causal_appearance_detection_only_no_consistency",
            "causal_appearance_no_consistency",
            "causal_appearance_no_margin",
        }:
            test_prob = causal_filter_probabilities(test_prob, test_rows, causal_decay)
        elif variant in SUPPORT_CONFIRM_VARIANTS:
            test_prob = support_confirmation_filter(
                test_prob,
                test_rows,
                confirmation_steps=3 if variant.startswith("confirm3_") else 2,
            )
        for sample_index, (sample, probability) in enumerate(
            zip(test_rows, test_prob)
        ):
            if variant in counterfactual_variants:
                alternative = test_counterfactual_prob[sample_index]
                prediction = decide_counterfactual(
                    probability,
                    alternative,
                    *selected,
                    counterfactual_margin,
                    sample,
                    require_proposal_consistency=require_proposal_consistency,
                )
                alternative_support_probability = float(
                    alternative[LABEL_TO_INDEX["supported"]]
                )
            else:
                prediction = decide(
                    probability,
                    *selected,
                    sample.proposal_consistent,
                    require_proposal_consistency=require_proposal_consistency,
                )
                alternative_support_probability = 0.0
            predictions[(sample.video, sample.frame)] = prediction
            detail.append(
                {
                    "video": sample.video,
                    "frame": sample.frame,
                    "outer_fold": fold_index,
                    "truth": sample.truth,
                    "prediction": prediction,
                    "support_probability": float(probability[LABEL_TO_INDEX["supported"]]),
                    "contradiction_probability": float(probability[LABEL_TO_INDEX["contradicted"]]),
                    "insufficient_probability": float(probability[LABEL_TO_INDEX["unresolved"]]),
                    "proposal_consistent": int(sample.proposal_consistent),
                    "support_threshold": selected[0],
                    "contradiction_threshold": selected[1],
                    "margin": selected[2],
                    "counterfactual_margin": counterfactual_margin,
                    "counterfactual_support_probability": alternative_support_probability,
                    "has_specialized_counterfactual": int(
                        sample.has_specialized_counterfactual
                    ),
                    "causal_decay": causal_decay,
                    "relation_blend": relation_blend,
                    "require_proposal_consistency": require_proposal_consistency,
                    "allow_margin": allow_margin,
                }
            )
    ordered_predictions = [predictions[(sample.video, sample.frame)] for sample in samples]
    result = summarize_predictions(samples, ordered_predictions)
    result["thresholds_by_fold"] = thresholds
    return result, detail


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    available_variants = (
        "detection",
        "scene_relations",
        "proposal_no_consistency",
        "proposal_no_margin",
        "proposal_context",
        "relations",
        "calibrated_fusion",
        "calibrated_fusion_no_consistency",
        "causal",
        "appearance",
        "appearance_no_consistency",
        "relations_appearance",
        "causal_appearance",
        "causal_appearance_detection_only_no_consistency",
        "causal_appearance_no_consistency",
        "causal_appearance_no_margin",
        "causal_appearance_counterfactual",
        "causal_appearance_counterfactual_no_consistency",
        "visual_appearance_no_consistency",
        "visual_appearance_no_relations_no_consistency",
        "confirm_visual_appearance_no_consistency",
        "confirm3_visual_appearance_no_consistency",
        "causal_visual_appearance_no_consistency",
        "causal_visual_appearance_counterfactual_no_consistency",
        "hierarchical_visual_appearance_no_consistency",
        "causal_hierarchical_visual_appearance_no_consistency",
        "ovr_visual_appearance_no_consistency",
        "causal_ovr_visual_appearance_no_consistency",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--timeline-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--embedding-npz",
        type=Path,
        help="Frozen predicted-box assembly embeddings; required by appearance variants.",
    )
    parser.add_argument(
        "--proposal-predictions-jsonl",
        type=Path,
        help=(
            "Label-free held-out proposal probabilities. Every evaluated frame "
            "must be present; no recorded-fusion fallback is allowed."
        ),
    )
    parser.add_argument("--ema-decay", type=float, default=0.60)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument(
        "--calibration-false-support-cap",
        type=float,
        default=0.03,
        help="Maximum false-support rate allowed when selecting an operating point on training-fold predictions.",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=("group_oof", "heldout"),
        default="group_oof",
        help=(
            "Use nested group out-of-fold predictions or a deterministic "
            "held-out subset of the outer training groups for calibration."
        ),
    )
    parser.add_argument(
        "--proposal-top-k",
        type=int,
        choices=(1, 2),
        default=1,
        help="Number of online proposal candidates that may instantiate claims.",
    )
    parser.add_argument(
        "--claim-scope",
        choices=("assembly", "all"),
        default="assembly",
        help=(
            "Evaluate atomic S2-S4 assembly claims, or include auxiliary "
            "state-validity intervals."
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=available_variants,
        default=list(available_variants),
        help="Verifier variants to evaluate; defaults to the complete suite.",
    )
    parser.add_argument(
        "--group-level",
        choices=("video", "scenario"),
        default="scenario",
        help="Hold out individual clips or all clips from the same scenario.",
    )
    args = parser.parse_args()
    if any("appearance" in variant for variant in args.variants) and args.embedding_npz is None:
        parser.error("--embedding-npz is required for appearance variants")

    proposal_provenance = audit_recorded_proposal_provenance(
        args.summary_csv,
        args.variants,
        args.proposal_predictions_jsonl,
    )
    samples, feature_names = build_samples(
        args.summary_csv,
        args.timeline_csv,
        args.ema_decay,
        args.proposal_top_k,
        args.claim_scope,
        args.embedding_npz,
        args.proposal_predictions_jsonl,
    )
    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for variant in args.variants:
        metrics, detail = evaluate_variant(
            samples,
            variant,
            args.outer_folds,
            args.group_level,
            args.calibration_mode,
            args.inner_folds,
            args.calibration_false_support_cap,
        )
        results[variant] = metrics
        rows.extend({"variant": variant, **row} for row in detail)
        print(
            f"{variant}: macro_f1={metrics['triage_macro_f1']:.4f}, "
            f"false_support={metrics['false_accept_rate_on_non_supported']:.4f}, "
            f"support={metrics['support_accuracy']:.4f}, "
            f"contradiction={metrics['contradiction_recall']:.4f}, "
            f"insufficient={metrics['unresolved_coverage']:.4f}",
            flush=True,
        )
    payload = {
        "protocol": {
            "outer_split": (
                f"{args.outer_folds}-fold stratified {args.group_level}-grouped "
                "cross-validation"
            ),
            "group_count": len({
                sample.video if args.group_level == "video" else scenario_group(sample.video)
                for sample in samples
            }),
            "operating_point": (
                f"{args.inner_folds}-fold nested group-OOF calibration"
                if args.calibration_mode == "group_oof"
                else "deterministic 20% outer-training-group calibration split"
            ),
            "future_frames_used": False,
            "test_outcomes_used_for_training_or_calibration": False,
            "active_claim_protocol": (
                "conditional verification of the annotated active claim; "
                "step proposal is evaluated separately"
            ),
            "proposal_top_k": args.proposal_top_k,
            "proposal_candidates": (
                "cross-fitted held-out causal-context probabilities"
                if args.proposal_predictions_jsonl is not None
                else (
                    "recorded online fusion scores"
                    if any(
                        variant in PROPOSAL_FEATURE_VARIANTS
                        for variant in args.variants
                    )
                    else "unused by the selected verifier variants"
                )
            ),
            "proposal_cache": (
                str(args.proposal_predictions_jsonl)
                if args.proposal_predictions_jsonl is not None
                else None
            ),
            "proposal_provenance_audit": proposal_provenance,
            "claim_scope": args.claim_scope,
            "calibration_false_support_cap": args.calibration_false_support_cap,
            "variant_inputs": {
                variant: {
                    "proposal_features": variant in PROPOSAL_FEATURE_VARIANTS,
                    "frozen_predicted_box_appearance": "appearance" in variant,
                    "past_only_posterior_filter": (
                        variant.startswith("causal")
                        or variant in SUPPORT_CONFIRM_VARIANTS
                    ),
                }
                for variant in args.variants
            },
            "features": feature_names,
            "samples": len(samples),
            "videos": len({sample.video for sample in samples}),
        },
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_rows(args.output_csv, rows)


if __name__ == "__main__":
    main()
