"""Assistance-calibrated evidence scoring for procedural claims.

This module stays inside the verifier/evidence layer. It learns lightweight
support and contradiction prototypes from online assistance feedback windows
and scores new RGB observations from detector-derived geometry features.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


HOUSING_CLASSES = ("type_5_gearbox_housing", "type_6_gearbox_housing")
ROLE_BY_CLASS = {
    "type_3_gear": "small_gear",
    "type_7_gear": "small_gear",
    "type_2_gear": "big_gear",
    "type_8_gear": "big_gear",
    "type_5_gearbox_cover": "cover",
    "type_6_gearbox_cover": "cover",
}
EXPECTED_BY_PRODUCT_ROLE = {
    ("A", "small_gear"): "type_3_gear",
    ("A", "big_gear"): "type_8_gear",
    ("A", "cover"): "type_5_gearbox_cover",
    ("A", "housing"): "type_5_gearbox_housing",
    ("B", "small_gear"): "type_7_gear",
    ("B", "big_gear"): "type_2_gear",
    ("B", "cover"): "type_6_gearbox_cover",
    ("B", "housing"): "type_6_gearbox_housing",
}
STEP_TARGET_ROLE = {
    "S2": "small_gear",
    "S3": "big_gear",
    "S4": "cover",
    "STEP2": "small_gear",
    "STEP3": "big_gear",
    "STEP4": "cover",
    "step2": "small_gear",
    "step3": "big_gear",
    "step4": "cover",
}
CLAIM_TARGET_ROLE = {
    "small_gear_inserted": "small_gear",
    "step2_small_gear_inserted": "small_gear",
    "big_gear_inserted": "big_gear",
    "step3_big_gear_inserted": "big_gear",
    "gear_inserted": "gear",
    "cover_fully_seated": "cover",
    "step4_cover_seated": "cover",
    "cover_seated": "cover",
}
FEATURE_NAMES = [
    "target_conf",
    "target_proposal_conf",
    "target_role_conf",
    "wrong_same_role_conf",
    "identity_margin",
    "housing_role_conf",
    "expected_housing_conf",
    "target_area_ratio",
    "housing_area_ratio",
    "target_to_housing_area",
    "containment_score",
    "intersection_over_target",
    "center_inside",
    "center_distance_norm",
    "edge_gap_norm",
    "vertical_gap_norm",
    "det_count",
    "top_conf",
]


def normalize_key(value: object) -> str:
    text = str(value or "").strip().replace("-", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text).strip("_")


def normalize_claim(value: object, step_id: object = "") -> str:
    claim = normalize_key(value).lower()
    if claim in {"step2_small_gear_inserted", "small_gear_inserted"}:
        return "small_gear_inserted"
    if claim in {"step3_big_gear_inserted", "big_gear_inserted"}:
        return "big_gear_inserted"
    if claim in {"step4_cover_seated", "cover_fully_seated", "cover_seated"}:
        return "cover_seated"
    step = normalize_key(step_id).upper()
    if step == "S2" or step == "STEP2":
        return "small_gear_inserted"
    if step == "S3" or step == "STEP3":
        return "big_gear_inserted"
    if step == "S4" or step == "STEP4":
        return "cover_seated"
    return claim or "state_validity"


def target_role_for(claim_id: object, step_id: object = "") -> str:
    claim = normalize_key(claim_id).lower()
    if claim in CLAIM_TARGET_ROLE:
        return CLAIM_TARGET_ROLE[claim]
    step = str(step_id or "").strip()
    return STEP_TARGET_ROLE.get(step, STEP_TARGET_ROLE.get(step.upper(), ""))


def expected_class(product: object, role: str) -> str:
    return EXPECTED_BY_PRODUCT_ROLE.get((str(product or "").strip().upper(), role), "")


def counterpart_class(product: object, role: str) -> str:
    product_key = str(product or "").strip().upper()
    other = "B" if product_key == "A" else "A" if product_key == "B" else ""
    return EXPECTED_BY_PRODUCT_ROLE.get((other, role), "")


def box_area(box: Tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(first: Tuple[float, float, float, float], second: Tuple[float, float, float, float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def center_inside(inner: Tuple[float, float, float, float], outer: Tuple[float, float, float, float]) -> float:
    cx = (inner[0] + inner[2]) * 0.5
    cy = (inner[1] + inner[3]) * 0.5
    return 1.0 if outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3] else 0.0


def edge_gap(first: Tuple[float, float, float, float], second: Tuple[float, float, float, float]) -> float:
    dx = max(first[0] - second[2], second[0] - first[2], 0.0)
    dy = max(first[1] - second[3], second[1] - first[3], 0.0)
    return float((dx * dx + dy * dy) ** 0.5)


def max_conf(detections: Iterable[Any], class_name: str) -> float:
    return max((det_conf(item) for item in detections if det_name(item) == class_name), default=0.0)


def identity_safe(item: Any) -> bool:
    if isinstance(item, Mapping):
        meta = item.get("meta", {})
    else:
        meta = getattr(item, "meta", {})
    if not isinstance(meta, Mapping) or "identity_safe" not in meta:
        return True
    return bool(meta.get("identity_safe", False))


def identity_conf(detections: Iterable[Any], class_name: str) -> float:
    return max(
        (det_conf(item) for item in detections if det_name(item) == class_name and identity_safe(item)),
        default=0.0,
    )


def det_name(item: Any) -> str:
    if isinstance(item, Mapping):
        return str(item.get("name", ""))
    return str(getattr(item, "name", ""))


def det_conf(item: Any) -> float:
    if isinstance(item, Mapping):
        return float(item.get("confidence", item.get("conf", 0.0)) or 0.0)
    return float(getattr(item, "confidence", 0.0) or 0.0)


def det_box(item: Any) -> Tuple[float, float, float, float]:
    if isinstance(item, Mapping):
        values = item.get("xyxy", [0.0, 0.0, 0.0, 0.0])
    else:
        values = getattr(item, "xyxy", [0.0, 0.0, 0.0, 0.0])
    return tuple(float(v) for v in values[:4])  # type: ignore[return-value]


def best_box(detections: Iterable[Any], class_names: Iterable[str]) -> Optional[Tuple[float, float, float, float]]:
    allowed = set(class_names)
    best: Optional[Tuple[float, float, float, float]] = None
    best_conf = -1.0
    for item in detections:
        if det_name(item) not in allowed:
            continue
        conf = det_conf(item)
        if conf > best_conf:
            best_conf = conf
            best = det_box(item)
    return best


def role_conf(detections: Iterable[Any], class_names: Iterable[str]) -> float:
    allowed = set(class_names)
    return max((det_conf(item) for item in detections if det_name(item) in allowed), default=0.0)


def role_classes(role: str) -> Tuple[str, ...]:
    if role == "small_gear":
        return ("type_3_gear", "type_7_gear")
    if role == "big_gear":
        return ("type_2_gear", "type_8_gear")
    if role == "gear":
        return ("type_2_gear", "type_3_gear", "type_7_gear", "type_8_gear")
    if role == "cover":
        return ("type_5_gearbox_cover", "type_6_gearbox_cover")
    if role == "housing":
        return HOUSING_CLASSES
    return ()


def extract_relation_features(
    detections: Sequence[Any],
    *,
    claim_id: object,
    step_id: object = "",
    product: object = "",
    image_shape: Optional[Tuple[int, int]] = None,
    role_detections: Optional[Sequence[Any]] = None,
) -> Dict[str, float]:
    """Extract identity and geometry evidence from factorized detections.

    Fine-grained detections remain the only source of product identity. When
    supplied, role detections provide category-level localization for geometry
    and relation features without being promoted to identity evidence.
    """

    claim = normalize_claim(claim_id, step_id)
    role = target_role_for(claim, step_id)
    if role == "gear":
        step = str(step_id or "").upper()
        role = STEP_TARGET_ROLE.get(step, "small_gear")
    target = expected_class(product, role)
    wrong = counterpart_class(product, role)
    housing = expected_class(product, "housing")
    target_conf = identity_conf(detections, target) if target else 0.0
    target_proposal_conf = max_conf(detections, target) if target else 0.0
    role_items = list(role_detections or [])
    coarse_target_conf = role_conf(role_items, (role,))
    fine_target_role_conf = role_conf(detections, role_classes(role))
    coarse_housing_conf = role_conf(role_items, ("housing",))
    fine_housing_role_conf = role_conf(detections, HOUSING_CLASSES)
    target_role_conf = max(coarse_target_conf, fine_target_role_conf)
    wrong_conf = identity_conf(detections, wrong) if wrong else 0.0
    housing_role_conf = max(coarse_housing_conf, fine_housing_role_conf)
    expected_housing_conf = identity_conf(detections, housing) if housing else 0.0
    target_box = best_box(detections, role_classes(role)) or best_box(
        role_items, (role,)
    )
    housing_box = best_box(detections, HOUSING_CLASSES) or best_box(
        role_items, ("housing",)
    )
    height = float(image_shape[0]) if image_shape else 1.0
    width = float(image_shape[1]) if image_shape else 1.0
    frame_area = max(1.0, width * height)

    target_area = box_area(target_box) if target_box else 0.0
    housing_area = box_area(housing_box) if housing_box else 0.0
    inter = intersection_area(target_box, housing_box) if target_box and housing_box else 0.0
    inter_over_target = inter / max(1.0, target_area) if target_box and housing_box else 0.0
    inside = center_inside(target_box, housing_box) if target_box and housing_box else 0.0
    if target_box and housing_box:
        tcx = (target_box[0] + target_box[2]) * 0.5
        tcy = (target_box[1] + target_box[3]) * 0.5
        hcx = (housing_box[0] + housing_box[2]) * 0.5
        hcy = (housing_box[1] + housing_box[3]) * 0.5
        hdiag = max(1.0, ((housing_box[2] - housing_box[0]) ** 2 + (housing_box[3] - housing_box[1]) ** 2) ** 0.5)
        center_dist_norm = ((tcx - hcx) ** 2 + (tcy - hcy) ** 2) ** 0.5 / hdiag
        gap_norm = edge_gap(target_box, housing_box) / max(1.0, hdiag)
        vertical_gap_norm = max(0.0, target_box[1] - housing_box[3], housing_box[1] - target_box[3]) / max(1.0, hdiag)
    else:
        center_dist_norm = 1.0
        gap_norm = 1.0
        vertical_gap_norm = 1.0
    containment = max(0.0, min(1.0, 0.65 * inter_over_target + 0.35 * inside))
    top_conf = max((det_conf(item) for item in detections), default=0.0)
    return {
        "target_conf": target_conf,
        "target_proposal_conf": target_proposal_conf,
        "target_role_conf": target_role_conf,
        "wrong_same_role_conf": wrong_conf,
        "identity_margin": target_conf - wrong_conf,
        "housing_role_conf": housing_role_conf,
        "expected_housing_conf": expected_housing_conf,
        "target_area_ratio": target_area / frame_area,
        "housing_area_ratio": housing_area / frame_area,
        "target_to_housing_area": target_area / max(1.0, housing_area),
        "containment_score": containment,
        "intersection_over_target": inter_over_target,
        "center_inside": inside,
        "center_distance_norm": center_dist_norm,
        "edge_gap_norm": gap_norm,
        "vertical_gap_norm": vertical_gap_norm,
        "det_count": float(len(detections)),
        "top_conf": top_conf,
    }


@dataclass
class PrototypeEvidenceScorer:
    """Nearest-prototype scorer trained from online feedback windows."""

    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    means: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    stds: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _mean(items: List[Mapping[str, float]], feature_names: Sequence[str]) -> Dict[str, float]:
        if not items:
            return {name: 0.0 for name in feature_names}
        return {name: sum(float(item.get(name, 0.0)) for item in items) / len(items) for name in feature_names}

    @classmethod
    def train(cls, samples: Sequence[Mapping[str, Any]], feature_names: Sequence[str] = FEATURE_NAMES) -> "PrototypeEvidenceScorer":
        by_claim_label: Dict[str, Dict[str, List[Mapping[str, float]]]] = {}
        all_features: List[Mapping[str, float]] = []
        for sample in samples:
            claim = normalize_claim(sample.get("claim_id"), sample.get("step_id"))
            label = str(sample.get("label", "")).strip()
            feats = dict(sample.get("features", {}))
            if label not in {"support", "contradiction"} or not claim:
                continue
            by_claim_label.setdefault(claim, {}).setdefault(label, []).append(feats)
            all_features.append(feats)

        means: Dict[str, Dict[str, Dict[str, float]]] = {}
        counts: Dict[str, Dict[str, float]] = {}
        for claim, label_map in by_claim_label.items():
            means[claim] = {}
            counts[claim] = {}
            for label in ("support", "contradiction"):
                means[claim][label] = cls._mean(label_map.get(label, []), feature_names)
                counts[claim][label] = float(len(label_map.get(label, [])))
        global_mean = cls._mean(all_features, feature_names)
        variances: Dict[str, float] = {}
        for name in feature_names:
            if len(all_features) <= 1:
                variances[name] = 1.0
                continue
            mean_value = global_mean[name]
            variances[name] = sum((float(item.get(name, 0.0)) - mean_value) ** 2 for item in all_features) / max(1, len(all_features) - 1)
        stds = {name: max(0.05, math.sqrt(value)) for name, value in variances.items()}
        return cls(
            feature_names=list(feature_names),
            means=means,
            stds=stds,
            counts=counts,
            metadata={"num_samples": len(samples), "num_usable_samples": len(all_features)},
        )

    def _distance(self, features: Mapping[str, float], proto: Mapping[str, float]) -> float:
        total = 0.0
        used = 0
        for name in self.feature_names:
            scale = max(0.05, float(self.stds.get(name, 1.0)))
            total += ((float(features.get(name, 0.0)) - float(proto.get(name, 0.0))) / scale) ** 2
            used += 1
        return (total / max(1, used)) ** 0.5

    def score_features(self, features: Mapping[str, float], *, claim_id: object, step_id: object = "") -> Dict[str, float]:
        claim = normalize_claim(claim_id, step_id)
        claim_means = self.means.get(claim, {})
        claim_counts = self.counts.get(claim, {})
        if not claim_means or max(claim_counts.values(), default=0.0) <= 0:
            support = 0.5 * float(features.get("target_conf", 0.0)) + 0.5 * float(features.get("containment_score", 0.0))
            contradiction = max(0.0, float(features.get("wrong_same_role_conf", 0.0)) - float(features.get("target_conf", 0.0)))
            return {
                "support_score": max(0.0, min(1.0, support)),
                "contradiction_score": max(0.0, min(1.0, contradiction)),
                "visibility_score": max(float(features.get("target_conf", 0.0)), float(features.get("housing_role_conf", 0.0))),
                "model_source": 0.0,
            }

        support_proto = claim_means.get("support")
        contradiction_proto = claim_means.get("contradiction")
        support_count = float(claim_counts.get("support", 0.0))
        contradiction_count = float(claim_counts.get("contradiction", 0.0))
        if not support_proto or support_count <= 0:
            support_score = 0.0
        elif not contradiction_proto or contradiction_count <= 0:
            support_score = max(0.0, min(1.0, 1.0 - self._distance(features, support_proto) / 3.0))
        else:
            d_pos = self._distance(features, support_proto)
            d_neg = self._distance(features, contradiction_proto)
            support_score = 1.0 / (1.0 + math.exp(2.0 * (d_pos - d_neg)))

        if not contradiction_proto or contradiction_count <= 0:
            contradiction_score = 0.0
        elif not support_proto or support_count <= 0:
            contradiction_score = max(0.0, min(1.0, 1.0 - self._distance(features, contradiction_proto) / 3.0))
        else:
            d_pos = self._distance(features, support_proto)
            d_neg = self._distance(features, contradiction_proto)
            contradiction_score = 1.0 / (1.0 + math.exp(2.0 * (d_neg - d_pos)))
        visibility = max(
            float(features.get("target_conf", 0.0)),
            float(features.get("housing_role_conf", 0.0)),
            float(features.get("containment_score", 0.0)),
        )
        return {
            "support_score": max(0.0, min(1.0, support_score)),
            "contradiction_score": max(0.0, min(1.0, contradiction_score)),
            "visibility_score": max(0.0, min(1.0, visibility)),
            "model_source": 1.0,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "means": self.means,
            "stds": self.stds,
            "counts": self.counts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrototypeEvidenceScorer":
        return cls(
            feature_names=[str(item) for item in payload.get("feature_names", FEATURE_NAMES)],
            means={
                str(claim): {
                    str(label): {str(name): float(value) for name, value in dict(values).items()}
                    for label, values in dict(label_map).items()
                }
                for claim, label_map in dict(payload.get("means", {})).items()
            },
            stds={str(name): float(value) for name, value in dict(payload.get("stds", {})).items()},
            counts={
                str(claim): {str(label): float(value) for label, value in dict(label_map).items()}
                for claim, label_map in dict(payload.get("counts", {})).items()
            },
            metadata=dict(payload.get("metadata", {})),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PrototypeEvidenceScorer":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
