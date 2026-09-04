"""Sparse causal evidence shared by step proposal and claim verification."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import joblib
import numpy as np
import torch
import torch.nn.functional as F

from core_types import (
    Detection,
    EvidenceToken,
    FusionResult,
    SceneGraphFrame,
    StepPrediction,
)
from .evidence_scorer import (
    EXPECTED_BY_PRODUCT_ROLE,
    FEATURE_NAMES as BOX_FEATURE_NAMES,
    det_box,
    det_conf,
    det_name,
    expected_class,
    extract_relation_features,
    target_role_for,
)


STEPS = ("S1", "S2", "S3", "S4")
CLASSES = tuple(sorted(set(EXPECTED_BY_PRODUCT_ROLE.values())))
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
PROPOSAL_PREDICATES = ("inside", "aligned_with", "contacting", "near", "overlapping")
ROLE_PAIRS = (
    ("small_gear", "housing"),
    ("big_gear", "housing"),
    ("cover", "housing"),
    ("small_gear", "big_gear"),
)
TRIAGE_PREDICATES = (
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
EVENT_NAMES = (
    "small_gear_assembly",
    "big_gear_assembly",
    "cover_assembly",
    "small_gear_presence",
    "big_gear_presence",
    "cover_presence",
)
MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


@dataclass(frozen=True)
class TriageOutput:
    state: str
    support_score: float
    contradiction_score: float
    insufficient_score: float
    visibility_score: float
    posterior_margin: float
    features: Mapping[str, float]


class CausalEvidenceBank:
    """Maintain current cheap evidence and sparse frozen appearance evidence."""

    def __init__(
        self,
        proposal_model_path: Path,
        triage_model_path: Path,
        *,
        dinov2_repo: Path,
        device: str = "cuda:0",
        load_encoder_on_start: bool = False,
    ) -> None:
        self.proposal_artifact = joblib.load(proposal_model_path)
        self.triage_artifact = joblib.load(triage_model_path)
        self._validate_artifacts()
        self.proposal_model = self.proposal_artifact["estimator"]
        self.triage_model = self.triage_artifact["estimator"]
        self.proposal_feature_names = tuple(self.proposal_artifact["feature_names"])
        self.triage_feature_names = tuple(self.triage_artifact["feature_names"])
        self.projection = np.asarray(
            self.proposal_artifact["appearance_projection"],
            dtype=np.float32,
        )
        triage_projection = np.asarray(
            self.triage_artifact["appearance_projection"],
            dtype=np.float32,
        )
        if self.projection.shape != triage_projection.shape or not np.allclose(
            self.projection,
            triage_projection,
        ):
            raise ValueError("Proposal and triage appearance projections differ")
        appearance = dict(self.proposal_artifact["appearance"])
        self.model_name = str(appearance.get("encoder", "dinov2_vitb14"))
        self.image_size = int(appearance.get("image_size", 224))
        self.roi_padding = float(appearance.get("roi_padding", 0.15))
        self.refresh_stride = max(1, int(appearance.get("refresh_stride", 4)))
        self.dinov2_repo = Path(dinov2_repo)
        self.device = torch.device(
            device if str(device).startswith("cpu") or torch.cuda.is_available() else "cpu"
        )
        self.encoder: torch.nn.Module | None = None
        if load_encoder_on_start:
            self._load_encoder()
        self.reset()

    def _validate_artifacts(self) -> None:
        for name, artifact in (
            ("proposal", self.proposal_artifact),
            ("triage", self.triage_artifact),
        ):
            if int(artifact.get("format_version", 0)) != 1:
                raise ValueError(f"Unsupported {name} artifact format")
            training = dict(artifact.get("training") or {})
            if training.get("robot_view_labels_used") is not False:
                raise ValueError(f"{name} artifact uses robot-view labels")
            if training.get("candidate_robot_images_used") is not False:
                raise ValueError(f"{name} artifact uses candidate robot images")
            if training.get("ground_truth_boxes_used") is not False:
                raise ValueError(f"{name} artifact uses ground-truth boxes")
            if training.get("future_frames_used") is not False:
                raise ValueError(f"{name} artifact uses future frames")
            estimator = artifact.get("estimator")
            feature_names = tuple(artifact.get("feature_names") or ())
            expected_features = int(getattr(estimator, "n_features_in_", -1))
            if expected_features != len(feature_names):
                raise ValueError(
                    f"{name} feature contract mismatch: "
                    f"artifact has {len(feature_names)}, model expects {expected_features}"
                )

    def reset(self) -> None:
        self._observation_index = 0
        self._last_refresh_index = -1
        self._appearance_embedding: np.ndarray | None = None
        self._appearance_refreshes = 0
        self._proposal_scores: dict[str, float] = {}
        self._proposal_step = ""
        self._proposal_runner_up = ""
        self._event_ema = np.zeros(len(EVENT_NAMES), dtype=np.float64)
        self._event_peak = np.zeros(len(EVENT_NAMES), dtype=np.float64)
        self._triage_posterior: np.ndarray | None = None
        self._last_appearance_vector = np.zeros(
            6 + int(self.projection.shape[1]),
            dtype=np.float64,
        )
        self._last_diagnostics: dict[str, float] = {}

    @property
    def diagnostics(self) -> Mapping[str, float]:
        observations = max(1, self._observation_index)
        return {
            **self._last_diagnostics,
            "appearance_refreshes": float(self._appearance_refreshes),
            "appearance_refresh_rate": self._appearance_refreshes / observations,
            "appearance_age": float(
                max(0, self._observation_index - 1 - self._last_refresh_index)
            ),
        }

    def _load_encoder(self) -> torch.nn.Module:
        if self.encoder is None:
            self.encoder = torch.hub.load(
                str(self.dinov2_repo),
                self.model_name,
                source="local",
                pretrained=True,
            ).eval().to(self.device)
        return self.encoder

    @staticmethod
    def _square_pad(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        side = max(height, width)
        canvas = np.full((side, side, 3), 114, dtype=np.uint8)
        top = (side - height) // 2
        left = (side - width) // 2
        canvas[top : top + height, left : left + width] = image
        return canvas

    def _image_tensor(self, image: np.ndarray) -> torch.Tensor:
        image = self._square_pad(image)
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
        return (tensor - MEAN) / STD

    def _roi_metadata(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = frame_bgr.shape[:2]
        boxes = [
            det_box(item)
            for item in detections
            if det_name(item) in CLASSES and det_box(item)[2] > det_box(item)[0]
            and det_box(item)[3] > det_box(item)[1]
        ]
        if boxes:
            x1 = min(box[0] for box in boxes)
            y1 = min(box[1] for box in boxes)
            x2 = max(box[2] for box in boxes)
            y2 = max(box[3] for box in boxes)
            span = max(32.0, max(x2 - x1, y2 - y1) * (1.0 + 2.0 * self.roi_padding))
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            left = max(0, int(round(cx - 0.5 * span)))
            top = max(0, int(round(cy - 0.5 * span)))
            right = min(width, int(round(cx + 0.5 * span)))
            bottom = min(height, int(round(cy + 0.5 * span)))
        else:
            left, top, right, bottom = 0, 0, width, height
        roi = frame_bgr[top : max(top + 1, bottom), left : max(left + 1, right)]
        metadata = np.asarray(
            [
                float(len(boxes) >= 2),
                min(1.0, len(boxes) / 4.0),
                left / max(1, width),
                top / max(1, height),
                right / max(1, width),
                bottom / max(1, height),
            ],
            dtype=np.float64,
        )
        return roi, metadata

    @torch.inference_mode()
    def _encode_roi(self, roi: np.ndarray) -> tuple[np.ndarray, float]:
        encoder = self._load_encoder()
        started = time.perf_counter()
        batch = self._image_tensor(roi).unsqueeze(0).to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            raw = encoder(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        raw = F.normalize(raw.float(), dim=1).cpu().numpy()[0]
        projected = raw.astype(np.float32) @ self.projection
        projected /= max(float(np.linalg.norm(projected)), 1e-8)
        return projected.astype(np.float64), (time.perf_counter() - started) * 1000.0

    def _appearance(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[Any],
        *,
        force_refresh: bool = False,
    ) -> tuple[np.ndarray, bool, float]:
        roi, metadata = self._roi_metadata(frame_bgr, detections)
        refresh = (
            force_refresh
            or self._appearance_embedding is None
            or self._observation_index % self.refresh_stride == 0
        )
        latency_ms = 0.0
        if refresh:
            self._appearance_embedding, latency_ms = self._encode_roi(roi)
            self._appearance_refreshes += 1
            self._last_refresh_index = self._observation_index
        vector = np.concatenate((metadata, self._appearance_embedding))
        self._last_appearance_vector = vector
        return vector, refresh, latency_ms

    @staticmethod
    def _relation_rows(scene_graph: SceneGraphFrame) -> list[dict[str, Any]]:
        return [
            {
                "subject": relation.subject_name,
                "predicate": relation.predicate,
                "object": relation.object_name,
                "score": float(relation.score),
            }
            for relation in scene_graph.relations
        ]

    def _proposal_base_features(
        self,
        detections: Sequence[Any],
        scene_graph: SceneGraphFrame,
        evidence_token: EvidenceToken,
        rule_prediction: StepPrediction,
    ) -> dict[str, float]:
        features: dict[str, float] = {}
        class_count: Counter[str] = Counter()
        role_count: Counter[str] = Counter()
        for class_name in CLASSES:
            rows = [item for item in detections if det_name(item) == class_name]
            features[f"class_conf:{class_name}"] = max(
                (det_conf(item) for item in rows),
                default=0.0,
            )
            class_count[class_name] = len(rows)
            features[f"class_count:{class_name}"] = min(3, len(rows)) / 3.0
        for role in ROLES:
            names = {name for name, value in ROLE_BY_CLASS.items() if value == role}
            rows = [item for item in detections if det_name(item) in names]
            features[f"role_conf:{role}"] = max(
                (det_conf(item) for item in rows),
                default=0.0,
            )
            role_count[role] = len(rows)
            features[f"role_count:{role}"] = min(3, len(rows)) / 3.0
        stable_roles: Counter[str] = Counter()
        for name, count in evidence_token.stable_track_counts.items():
            role = ROLE_BY_CLASS.get(str(name).lower())
            if role:
                stable_roles[role] += int(count)
        for role in ROLES:
            features[f"stable_role_count:{role}"] = min(
                3,
                stable_roles[role],
            ) / 3.0
        pair_scores = {
            (left, predicate, right): 0.0
            for left, right in ROLE_PAIRS
            for predicate in PROPOSAL_PREDICATES
        }
        predicate_counts: Counter[str] = Counter()
        for relation in scene_graph.relations:
            predicate = str(relation.predicate).lower()
            if predicate not in PROPOSAL_PREDICATES:
                continue
            predicate_counts[predicate] += 1
            subject = ROLE_BY_CLASS.get(
                str(relation.subject_name).lower(),
                str(relation.subject_name).lower(),
            )
            obj = ROLE_BY_CLASS.get(
                str(relation.object_name).lower(),
                str(relation.object_name).lower(),
            )
            for left, right in ROLE_PAIRS:
                if {subject, obj} == {left, right}:
                    key = (left, predicate, right)
                    pair_scores[key] = max(pair_scores[key], float(relation.score))
        for key, value in pair_scores.items():
            features[f"pair:{key[0]}:{key[1]}:{key[2]}"] = value
        for predicate in PROPOSAL_PREDICATES:
            features[f"scene_count:{predicate}"] = min(
                4,
                predicate_counts[predicate],
            ) / 4.0
        features.update(
            {
                "det_count": min(8, len(detections)) / 8.0,
                "stable_track_count": min(8, sum(stable_roles.values())) / 8.0,
                "relation_density": min(20, len(scene_graph.relations)) / 20.0,
                "has_visual": float(bool(detections or scene_graph.relations)),
            }
        )
        for step in STEPS:
            features[f"rule_score:{step}"] = float(
                rule_prediction.scores.get(step, 0.0)
            )
        return features

    @staticmethod
    def _event_signals(features: Mapping[str, float]) -> np.ndarray:
        def value(name: str) -> float:
            return float(features.get(name, 0.0))

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

    def _update_events(self, features: dict[str, float]) -> None:
        current = self._event_signals(features)
        self._event_ema = 0.55 * current + 0.45 * self._event_ema
        self._event_peak = np.maximum(current, 0.995 * self._event_peak)
        for index, name in enumerate(EVENT_NAMES):
            features[f"event:{name}"] = float(current[index])
            features[f"causal_ema:{name}"] = float(self._event_ema[index])
            features[f"causal_peak:{name}"] = float(self._event_peak[index])

    @staticmethod
    def _appearance_features(vector: np.ndarray) -> dict[str, float]:
        result = {
            "appearance:roi_valid": float(vector[0]),
            "appearance:detection_count": float(vector[1]),
            "appearance:roi_x1": float(vector[2]),
            "appearance:roi_y1": float(vector[3]),
            "appearance:roi_x2": float(vector[4]),
            "appearance:roi_y2": float(vector[5]),
        }
        for index, value in enumerate(vector[6:]):
            result[f"appearance:dinov2_rp_{index:03d}"] = float(value)
        return result

    @staticmethod
    def _aligned_probabilities(model: Any, values: np.ndarray, size: int) -> np.ndarray:
        raw = model.predict_proba(values.reshape(1, -1))[0]
        output = np.zeros(size, dtype=np.float64)
        for column, class_index in enumerate(model.classes_):
            output[int(class_index)] = raw[column]
        return output

    def propose(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[Any],
        scene_graph: SceneGraphFrame,
        evidence_token: EvidenceToken,
        rule_prediction: StepPrediction,
        *,
        force_appearance_refresh: bool = False,
    ) -> FusionResult:
        appearance, refreshed, encoder_ms = self._appearance(
            frame_bgr,
            detections,
            force_refresh=force_appearance_refresh,
        )
        features = self._proposal_base_features(
            detections,
            scene_graph,
            evidence_token,
            rule_prediction,
        )
        self._update_events(features)
        features.update(self._appearance_features(appearance))
        proposal_started = time.perf_counter()
        if refreshed or not self._proposal_scores:
            values = np.asarray(
                [features.get(name, 0.0) for name in self.proposal_feature_names],
                dtype=np.float64,
            )
            probabilities = self._aligned_probabilities(
                self.proposal_model,
                values,
                len(STEPS),
            )
            self._proposal_scores = {
                step: float(probabilities[index])
                for index, step in enumerate(STEPS)
            }
            ranked = sorted(
                STEPS,
                key=lambda step: (-self._proposal_scores[step], step),
            )
            self._proposal_step, self._proposal_runner_up = ranked[:2]
        proposal_ms = (time.perf_counter() - proposal_started) * 1000.0
        self._observation_index += 1
        self._last_diagnostics = {
            "appearance_refreshed": float(refreshed),
            "appearance_encoder_ms": float(encoder_ms),
            "proposal_model_ms": float(proposal_ms),
            "proposal_held": float(not refreshed),
        }
        return FusionResult(
            step_id=self._proposal_step,
            confidence=float(self._proposal_scores[self._proposal_step]),
            scores=dict(self._proposal_scores),
            runner_up=self._proposal_runner_up,
            gates={"causal_evidence_bank": 1.0},
            extras={
                "source": "sparse_causal_evidence_bank",
                "appearance_refreshed": bool(refreshed),
                "appearance_encoder_ms": float(encoder_ms),
                "refresh_stride": int(self.refresh_stride),
            },
        )

    @staticmethod
    def _triage_relation_features(
        scene_graph: SceneGraphFrame,
        *,
        claim: str,
        step: str,
        product: str,
    ) -> dict[str, float]:
        role = target_role_for(claim, step)
        product_key = product if product in {"A", "B"} else "UNKNOWN"
        products = (product_key,) if product_key in {"A", "B"} else ("A", "B")
        targets = {expected_class(value, role) for value in products}
        housings = {expected_class(value, "housing") for value in products}
        targets.discard("")
        housings.discard("")
        role_targets = {
            "small_gear": {"type_3_gear", "type_7_gear"},
            "big_gear": {"type_2_gear", "type_8_gear"},
            "cover": {"type_5_gearbox_cover", "type_6_gearbox_cover"},
        }.get(role, set(targets))
        role_context = {
            "type_5_gearbox_housing",
            "type_6_gearbox_housing",
        }
        output = {
            f"{prefix}:{predicate}": 0.0
            for prefix in ("target_housing", "role_pair", "scene_count")
            for predicate in TRIAGE_PREDICATES
        }
        for relation in scene_graph.relations:
            predicate = str(relation.predicate)
            if predicate not in TRIAGE_PREDICATES:
                continue
            score = max(0.0, min(1.0, float(relation.score)))
            output[f"scene_count:{predicate}"] = min(
                1.0,
                output[f"scene_count:{predicate}"] + 0.25,
            )
            subject = str(relation.subject_name)
            obj = str(relation.object_name)
            if (subject in targets and obj in housings) or (
                obj in targets and subject in housings
            ):
                key = f"target_housing:{predicate}"
                output[key] = max(output[key], score)
            if (subject in role_targets and obj in role_context) or (
                obj in role_targets and subject in role_context
            ):
                key = f"role_pair:{predicate}"
                output[key] = max(output[key], score)
        output["scene_relation_density"] = min(
            1.0,
            float(scene_graph.stats.get("num_relations", len(scene_graph.relations)))
            / 20.0,
        )
        output["scene_relation_confidence"] = max(
            0.0,
            min(1.0, float(scene_graph.stats.get("avg_relation_score", 0.0))),
        )
        output["scene_focus_density"] = min(
            1.0,
            float(scene_graph.stats.get("focus_relations", 0.0)) / 12.0,
        )
        return output

    def triage(
        self,
        detections: Sequence[Any],
        scene_graph: SceneGraphFrame,
        evidence_token: EvidenceToken,
        fusion_result: FusionResult,
        *,
        claim: str,
        step: str,
        product: str,
        image_shape: tuple[int, int],
        role_detections: Sequence[Any] | None = None,
    ) -> TriageOutput:
        product_key = product if product in {"A", "B"} else "UNKNOWN"
        features = extract_relation_features(
            detections,
            claim_id=claim,
            step_id=step,
            product=product_key,
            image_shape=image_shape,
            role_detections=role_detections,
        )
        for class_name in CLASSES:
            rows = [item for item in detections if det_name(item) == class_name]
            features[f"class_conf:{class_name}"] = max(
                (det_conf(item) for item in rows),
                default=0.0,
            )
            features[f"class_count:{class_name}"] = min(3, len(rows)) / 3.0
        features.update(
            {f"claim:{name}": float(claim == name) for name in CLAIMS}
        )
        features.update(
            {f"product:{name}": float(product_key == name) for name in PRODUCTS}
        )
        features.update(
            self._triage_relation_features(
                scene_graph,
                claim=claim,
                step=step,
                product=product_key,
            )
        )
        ranked = sorted(fusion_result.scores.values(), reverse=True)
        features.update(
            {
                "proposal_target_score": float(fusion_result.scores.get(step, 0.0)),
                "proposal_confidence": float(fusion_result.confidence),
                "proposal_margin": (
                    float(ranked[0] - ranked[1])
                    if len(ranked) >= 2
                    else float(fusion_result.confidence)
                ),
                "proposal_disagreement": 0.0,
                "visual_evidence_present": float(
                    bool(detections or scene_graph.relations)
                ),
                "stable_tracks": min(
                    1.0,
                    sum(float(value) for value in evidence_token.stable_track_counts.values())
                    / 4.0,
                ),
            }
        )
        features.update(self._appearance_features(self._last_appearance_vector))
        values = np.asarray(
            [features.get(name, 0.0) for name in self.triage_feature_names],
            dtype=np.float64,
        )
        triage_started = time.perf_counter()
        raw = self._aligned_probabilities(self.triage_model, values, 3)
        operating = dict(self.triage_artifact["operating_point"])
        decay = float(operating.get("causal_decay", 0.0))
        posterior = (
            raw
            if self._triage_posterior is None
            else decay * self._triage_posterior + (1.0 - decay) * raw
        )
        posterior /= max(float(posterior.sum()), 1e-8)
        self._triage_posterior = posterior
        outcomes = tuple(self.triage_artifact["outcomes"])
        scores = {name: float(posterior[index]) for index, name in enumerate(outcomes)}
        support = scores["supported"]
        contradiction = scores["contradicted"]
        insufficient = scores["unresolved"]
        margin = float(operating.get("posterior_margin", 0.0))
        if (
            contradiction >= float(operating["contradiction_threshold"])
            and contradiction >= max(support, insufficient) + margin
        ):
            state = "contradicted"
        elif (
            support >= float(operating["support_threshold"])
            and support >= max(contradiction, insufficient) + margin
        ):
            state = "supported"
        else:
            state = "insufficient"
        self._last_diagnostics["triage_model_ms"] = (
            time.perf_counter() - triage_started
        ) * 1000.0
        return TriageOutput(
            state=state,
            support_score=support,
            contradiction_score=contradiction,
            insufficient_score=insufficient,
            visibility_score=1.0 - insufficient,
            posterior_margin=support - max(contradiction, insufficient),
            features=features,
        )
