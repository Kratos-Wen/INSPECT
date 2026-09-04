"""TwinSwap distillation trainer for Ultralytics YOLO detectors.

TwinSwap trains a detector on visual minimal pairs: two images share the same
Generative Nuisance Field (or optional real background), position, apparent
scale, blur, lighting, occlusion, and compression, but the pasted object
identity is swapped. During training the student receives normal YOLO detection
supervision plus a counterfactual response loss that makes its class response
change like the teacher's response across each swap pair.

Expected asset layout:

    object_root/
      spur_gear/*.png
      helical_gear/*.png
      screwdriver/*.png

PNG assets with alpha masks are preferred. RGB/JPEG assets are also accepted and
will be pasted as rectangular cutouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PAIR_RE = re.compile(r"twinswap_p(?P<pair>\d+)_[ab]_c(?P<class>\d+)_")

# Weak ordinal scene-scale priors used only for non-causal context/dense
# placements. They encode ordering, not metric ratios. TwinSwap swap slots
# deliberately keep the same apparent bbox.
ORDINAL_SCALE_GROUP_BY_CLASS = {
    "type_7_gear": "gear_small",
    "type_3_gear": "gear_small",
    "type_2_gear": "gear_large",
    "type_8_gear": "gear_large",
    "type_6_gearbox_cover": "enclosure",
    "type_5_gearbox_cover": "enclosure",
    "type_6_gearbox_housing": "enclosure",
    "type_5_gearbox_housing": "enclosure",
}
ORDINAL_SCALE_MULTIPLIER_RANGE = {
    "gear_small": (0.92, 1.06),
    "gear_large": (1.00, 1.18),
    "enclosure": (1.18, 1.55),
}

IDENTITY_HARD_PAIR_NAMES = frozenset(
    frozenset(pair)
    for pair in (
        ("type_2_gear", "type_8_gear"),
        ("type_3_gear", "type_7_gear"),
        ("type_5_gearbox_cover", "type_6_gearbox_cover"),
        ("type_5_gearbox_housing", "type_6_gearbox_housing"),
    )
)


def _class_family(class_name: str) -> str:
    if class_name.endswith("_gear"):
        return "gear"
    if "gearbox_cover" in class_name:
        return "cover"
    if "gearbox_housing" in class_name:
        return "housing"
    return "other"


def _is_identity_hard_pair(class_a: str, class_b: str) -> bool:
    return frozenset((class_a, class_b)) in IDENTITY_HARD_PAIR_NAMES


def _is_cross_form_pair(class_a: str, class_b: str) -> bool:
    return _class_family(class_a) != _class_family(class_b)


@dataclass(frozen=True)
class ObjectAsset:
    """One object cutout asset."""

    path: Path
    class_id: int
    class_name: str


@dataclass
class PairRecord:
    """Manifest entry for one TwinSwap pair."""

    pair_id: int
    image_a: str
    image_b: str
    class_a: int
    class_b: int
    name_a: str
    name_b: str
    bbox_xywhn: List[float]
    slot_bbox_xywhn: List[float]
    visible_bbox_a_xywhn: List[float]
    visible_bbox_b_xywhn: List[float]
    scene_type: str
    pair_role: str
    identity_kd_enabled: bool
    objects_a: List[Dict[str, Any]]
    objects_b: List[Dict[str, Any]]
    nuisance: Dict[str, Any]


@dataclass
class NuisanceMiningResult:
    """Selected model-aware nuisance fields and class-pair weights."""

    pair_weights: Dict[str, float]
    hard_nuisances: List[Dict[str, Any]]
    triaged_nuisances: List[Dict[str, Any]] = field(default_factory=list)
    triage_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class TwinSwapConfig:
    """Configuration for generating and training on TwinSwap minimal pairs."""

    object_root: Path
    output_dir: Path
    student_model: str
    background_root: Optional[Path] = None
    teacher_model: Optional[str] = None
    teacher_base_model: Optional[str] = None
    train_teacher: bool = False
    teacher_epochs: Optional[int] = None
    teacher_batch: Optional[int] = None
    teacher_name: str = "twinswap_teacher"
    classes: List[str] = field(default_factory=list)
    train_pairs: int = 1200
    val_pairs: int = 160
    mine_pairs: int = 240
    rounds: int = 1
    epochs_per_round: int = 40
    imgsz: int = 640
    batch: int = 16
    device: str = "cpu"
    workers: int = 4
    seed: int = 7
    project: str = "runs_twinswap"
    name: str = "twinswap"
    lr0: Optional[float] = None
    patience: int = 40
    kd_weight: float = 0.35
    kd_temperature: float = 2.0
    kd_topk: int = 24
    kd_margin_weight: float = 0.75
    hard_pair_margin_weight: float = 0.75
    hard_pair_target_margin: float = 0.50
    # Legacy PI-TwinSwap calibration terms are kept for ablation only.
    # The mainline trainer is pure TwinSwap: object identity changes across a
    # causal swap, while detection loss handles all boxes.
    presence_invariance_weight: float = 0.0
    presence_alignment_weight: float = 0.0
    real_presence_weight: float = 0.0
    context_invariance_weight: float = 0.0
    identity_kd_hard_only: bool = True
    identity_hard_pair_ratio: float = 0.75
    cross_form_pair_ratio: float = 0.15
    pair_keep_batches: bool = True
    hard_pair_temperature: float = 0.35
    hard_pair_min_weight: float = 0.20
    hard_pair_weight_file: Optional[Path] = None
    background_mode: str = "gnf"
    gnf_real_background_prob: float = 0.10
    hard_nuisance_replay_prob: float = 0.35
    hard_nuisance_profile_prob: float = 0.35
    hard_nuisance_topk: int = 512
    teacher_margin_floor: float = 0.05
    # Legacy PI mining gates. Defaults disable recall-hard triage so mined
    # nuisances are selected by TwinSwap identity regret.
    recall_hard_student_score_floor: float = -1.0
    presence_collapse_margin: float = 1.0e9
    semantic_safety: bool = True
    semantic_safety_conf: float = 0.35
    single_object_ratio: float = 0.30
    multi_object_ratio: float = 0.50
    dense_scene_ratio: float = 0.20
    context_object_count_range: Tuple[int, int] = (2, 5)
    dense_object_count_range: Tuple[int, int] = (4, 8)
    target_scale_range: Tuple[float, float] = (0.18, 0.42)
    distractor_scale_range: Tuple[float, float] = (0.08, 0.28)
    real_anchor_video_root: Optional[Path] = None
    real_anchor_mask_root: Optional[Path] = None
    real_anchor_train_per_class: int = 0
    real_anchor_val_per_class: int = 0
    real_anchor_train_repeat: int = 1
    real_anchor_include_mesh: bool = False
    real_anchor_min_area_ratio: float = 0.0005
    real_anchor_max_area_ratio: float = 0.80
    pseudo_real_source_root: Optional[Path] = None
    pseudo_real_filename_prefix: str = "zzz_real_pseudo_"
    pseudo_real_train_limit: int = 0
    pseudo_real_val_limit: int = 0
    max_slot_iou: float = 0.15
    max_placement_attempts: int = 80
    swap_visible_area_tolerance: float = 0.20
    swap_match_attempts: int = 8
    min_visible_area_ratio: float = 0.08
    use_kg_scale_prior: bool = True
    kg_scale_prior_strength: float = 1.0
    kg_scale_prior_jitter: float = 0.25
    blur_prob: float = 0.35
    occlusion_prob: float = 0.20
    preserve_aspect: bool = True
    clear_output: bool = False
    cache: bool = False


def _slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return value or "class"


def _iter_images(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def _stable_int(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_classes(path: Optional[Path], object_root: Path) -> List[str]:
    if path is not None:
        text = path.read_text(encoding="utf-8")
        names = [line.strip() for line in text.splitlines() if line.strip()]
        if names:
            return names
    return [item.name for item in sorted(object_root.iterdir()) if item.is_dir()]


class TwinSwapDatasetBuilder:
    """Build a YOLO-format TwinSwap visual-minimal-pair dataset."""

    def __init__(self, config: TwinSwapConfig) -> None:
        self.config = config
        self._validate_config()
        self.rng = random.Random(config.seed)
        self.classes = list(config.classes)
        if not self.classes:
            self.classes = _read_classes(None, config.object_root)
        if len(self.classes) < 2:
            raise ValueError("TwinSwap needs at least two object classes.")

        self.assets_by_class = self._load_assets()
        self.background_mode = str(config.background_mode or "gnf").lower()
        self.backgrounds: List[Path] = []
        if self.background_mode in {"directory", "mixed"}:
            if config.background_root is None:
                raise ValueError(f"background_mode={self.background_mode} requires background_root.")
            self.backgrounds = list(_iter_images(config.background_root))
            if not self.backgrounds:
                raise ValueError(f"No background images found under {config.background_root}")
        if self.background_mode in {"gnf", "mixed"}:
            from .gnf import GNFConfig, GNFGenerator

            self.gnf_generator = GNFGenerator(GNFConfig(width=config.imgsz, height=config.imgsz))
        else:
            self.gnf_generator = None
        self.hard_nuisances: List[Dict[str, Any]] = []

    def _validate_config(self) -> None:
        if self.config.train_pairs < 1:
            raise ValueError("train_pairs must be at least 1.")
        if self.config.val_pairs < 1:
            raise ValueError("val_pairs must be at least 1 because Ultralytics expects a validation split.")
        if self.config.mine_pairs < 0:
            raise ValueError("mine_pairs must be non-negative.")
        min_scale, max_scale = self.config.target_scale_range
        if min_scale <= 0 or max_scale <= 0 or min_scale > max_scale:
            raise ValueError("target_scale_range must be positive and ordered as (min, max).")
        min_scale, max_scale = self.config.distractor_scale_range
        if min_scale <= 0 or max_scale <= 0 or min_scale > max_scale:
            raise ValueError("distractor_scale_range must be positive and ordered as (min, max).")
        if not 0.0 <= self.config.single_object_ratio <= 1.0:
            raise ValueError("single_object_ratio must be in [0, 1].")
        if not 0.0 <= self.config.multi_object_ratio <= 1.0:
            raise ValueError("multi_object_ratio must be in [0, 1].")
        if not 0.0 <= self.config.dense_scene_ratio <= 1.0:
            raise ValueError("dense_scene_ratio must be in [0, 1].")
        if self.config.real_anchor_train_per_class < 0:
            raise ValueError("real_anchor_train_per_class must be non-negative.")
        if self.config.real_anchor_val_per_class < 0:
            raise ValueError("real_anchor_val_per_class must be non-negative.")
        if self.config.real_anchor_train_repeat < 1:
            raise ValueError("real_anchor_train_repeat must be at least 1.")
        if not 0.0 <= self.config.real_anchor_min_area_ratio <= self.config.real_anchor_max_area_ratio <= 1.0:
            raise ValueError("real anchor area ratios must satisfy 0 <= min <= max <= 1.")
        if self.config.single_object_ratio + self.config.multi_object_ratio <= 0.0:
            raise ValueError("At least one TwinSwap scene ratio must be positive.")
        if self.config.max_placement_attempts < 1:
            raise ValueError("max_placement_attempts must be at least 1.")
        if not 0.0 <= self.config.max_slot_iou <= 1.0:
            raise ValueError("max_slot_iou must be in [0, 1].")
        if not 0.0 <= self.config.blur_prob <= 1.0:
            raise ValueError("blur_prob must be in [0, 1].")
        if not 0.0 <= self.config.occlusion_prob <= 1.0:
            raise ValueError("occlusion_prob must be in [0, 1].")
        if str(self.config.background_mode).lower() not in {"gnf", "directory", "mixed"}:
            raise ValueError("background_mode must be one of: gnf, directory, mixed.")
        if not 0.0 <= self.config.gnf_real_background_prob <= 1.0:
            raise ValueError("gnf_real_background_prob must be in [0, 1].")
        if not 0.0 <= self.config.hard_nuisance_replay_prob <= 1.0:
            raise ValueError("hard_nuisance_replay_prob must be in [0, 1].")
        if not 0.0 <= self.config.hard_nuisance_profile_prob <= 1.0:
            raise ValueError("hard_nuisance_profile_prob must be in [0, 1].")
        if self.config.hard_nuisance_topk < 1:
            raise ValueError("hard_nuisance_topk must be at least 1.")
        if self.config.teacher_margin_floor < 0.0:
            raise ValueError("teacher_margin_floor must be non-negative.")
        if self.config.recall_hard_student_score_floor < -1.0:
            raise ValueError("recall_hard_student_score_floor must be >= -1.0; -1 disables this legacy gate.")
        if self.config.presence_collapse_margin < 0.0:
            raise ValueError("presence_collapse_margin must be non-negative.")
        if self.config.presence_invariance_weight < 0.0:
            raise ValueError("presence_invariance_weight must be non-negative.")
        if self.config.presence_alignment_weight < 0.0:
            raise ValueError("presence_alignment_weight must be non-negative.")
        if self.config.real_presence_weight < 0.0:
            raise ValueError("real_presence_weight must be non-negative.")
        if self.config.hard_pair_margin_weight < 0.0:
            raise ValueError("hard_pair_margin_weight must be non-negative.")
        if self.config.hard_pair_target_margin < 0.0:
            raise ValueError("hard_pair_target_margin must be non-negative.")
        if not 0.0 <= self.config.semantic_safety_conf <= 1.0:
            raise ValueError("semantic_safety_conf must be in [0, 1].")
        if not 0.0 <= self.config.swap_visible_area_tolerance <= 1.0:
            raise ValueError("swap_visible_area_tolerance must be in [0, 1].")
        if self.config.swap_match_attempts < 1:
            raise ValueError("swap_match_attempts must be at least 1.")
        if not 0.0 <= self.config.min_visible_area_ratio <= 1.0:
            raise ValueError("min_visible_area_ratio must be in [0, 1].")
        if not 0.0 <= self.config.kg_scale_prior_strength <= 1.0:
            raise ValueError("kg_scale_prior_strength must be in [0, 1].")
        if not 0.0 <= self.config.kg_scale_prior_jitter <= 1.0:
            raise ValueError("kg_scale_prior_jitter must be in [0, 1].")
        if self.config.context_invariance_weight < 0.0:
            raise ValueError("context_invariance_weight must be non-negative.")
        if not 0.0 <= self.config.identity_hard_pair_ratio <= 1.0:
            raise ValueError("identity_hard_pair_ratio must be in [0, 1].")
        if not 0.0 <= self.config.cross_form_pair_ratio <= 1.0:
            raise ValueError("cross_form_pair_ratio must be in [0, 1].")
        if self.config.identity_hard_pair_ratio + self.config.cross_form_pair_ratio > 1.0:
            raise ValueError("identity_hard_pair_ratio + cross_form_pair_ratio must be <= 1.")

    def build(self, root: Path, pair_weights: Optional[Dict[str, float]] = None) -> Path:
        """Generate train/val/mine splits and return the generated data.yaml path."""

        if root.exists() and self.config.clear_output:
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        self._write_data_yaml(root)
        self._write_class_file(root)

        train_records = self._build_split(root, "train", self.config.train_pairs, pair_weights)
        val_records = self._build_split(root, "val", self.config.val_pairs, None)
        mine_records = self._build_split(root, "mine", self.config.mine_pairs, None)

        _write_jsonl(root / "twinswap_pairs_train.jsonl", (record.__dict__ for record in train_records))
        _write_jsonl(root / "twinswap_pairs_val.jsonl", (record.__dict__ for record in val_records))
        _write_jsonl(root / "twinswap_pairs_mine.jsonl", (record.__dict__ for record in mine_records))
        return root / "data.yaml"

    def _load_assets(self) -> Dict[int, List[ObjectAsset]]:
        assets_by_class: Dict[int, List[ObjectAsset]] = {}
        for class_id, class_name in enumerate(self.classes):
            class_dir = self.config.object_root / class_name
            if not class_dir.exists():
                class_dir = self.config.object_root / _slug(class_name)
            paths = list(_iter_images(class_dir)) if class_dir.exists() else []
            if not paths:
                raise ValueError(f"No object assets found for class '{class_name}' under {class_dir}")
            assets_by_class[class_id] = [
                ObjectAsset(path=path, class_id=class_id, class_name=class_name) for path in paths
            ]
        return assets_by_class

    def _write_data_yaml(self, root: Path) -> None:
        payload = {
            "path": str(root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {index: name for index, name in enumerate(self.classes)},
            "channels": 3,
        }
        with (root / "data.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)

    def _write_class_file(self, root: Path) -> None:
        (root / "classes.txt").write_text("\n".join(self.classes) + "\n", encoding="utf-8")

    def _build_split(
        self,
        root: Path,
        split: str,
        pair_count: int,
        pair_weights: Optional[Dict[str, float]],
    ) -> List[PairRecord]:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        swap_dir = root / "swaps" / split
        for directory in (image_dir, label_dir, swap_dir):
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)
        records: List[PairRecord] = []
        dense_count = 0 if split == "mine" else int(round(pair_count * self.config.dense_scene_ratio))
        twin_count = max(0, pair_count - dense_count)
        for pair_id in range(twin_count):
            class_a, class_b = self._sample_class_pair(pair_weights)
            record = self._render_pair(root, split, pair_id, class_a, class_b)
            records.append(record)
        for dense_id in range(dense_count):
            self._render_dense_scene(root, split, dense_id)
        self._export_real_presence_anchors(root, split)
        self._import_pseudo_real_frames(root, split)
        return records

    def _sample_class_pair(self, pair_weights: Optional[Dict[str, float]]) -> Tuple[int, int]:
        pairs = [(i, j) for i in range(len(self.classes)) for j in range(i + 1, len(self.classes))]
        hard_pairs = [pair for pair in pairs if self._is_identity_hard_pair_indices(*pair)]
        cross_form_pairs = [pair for pair in pairs if self._is_cross_form_pair_indices(*pair)]
        same_family_pairs = [pair for pair in pairs if pair not in cross_form_pairs]
        roll = self.rng.random()
        if hard_pairs and roll < self.config.identity_hard_pair_ratio:
            pairs = hard_pairs
        elif cross_form_pairs and roll < self.config.identity_hard_pair_ratio + self.config.cross_form_pair_ratio:
            pairs = cross_form_pairs
        elif same_family_pairs:
            pairs = same_family_pairs
        if not pair_weights:
            return self.rng.choice(pairs)
        weights = []
        for i, j in pairs:
            key = f"{i}:{j}"
            weights.append(max(self.config.hard_pair_min_weight, float(pair_weights.get(key, 1.0))))
        return self.rng.choices(pairs, weights=weights, k=1)[0]

    def _is_identity_hard_pair_indices(self, class_a: int, class_b: int) -> bool:
        return _is_identity_hard_pair(self.classes[int(class_a)], self.classes[int(class_b)])

    def _is_cross_form_pair_indices(self, class_a: int, class_b: int) -> bool:
        return _is_cross_form_pair(self.classes[int(class_a)], self.classes[int(class_b)])

    def _pair_role(self, class_a: int, class_b: int) -> str:
        if self._is_identity_hard_pair_indices(class_a, class_b):
            return "identity_hard"
        if self._is_cross_form_pair_indices(class_a, class_b):
            return "cross_form_recall"
        return "same_family_recall"

    def _identity_kd_enabled(self, class_a: int, class_b: int) -> bool:
        return (not self.config.identity_kd_hard_only) or self._is_identity_hard_pair_indices(class_a, class_b)

    def _sample_twinswap_scene_type(self) -> str:
        total = max(1e-6, self.config.single_object_ratio + self.config.multi_object_ratio)
        return "single" if self.rng.random() < self.config.single_object_ratio / total else "multi"

    def _render_pair(self, root: Path, split: str, pair_id: int, class_a: int, class_b: int) -> PairRecord:
        base, background_meta = self._make_background(split, pair_id)
        width, height = base.size
        scene_type = self._sample_twinswap_scene_type()
        pair_role = self._pair_role(class_a, class_b)
        identity_kd_enabled = self._identity_kd_enabled(class_a, class_b)
        x1, y1, x2, y2 = self._sample_box(width, height, self.config.target_scale_range, [])
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        nuisance = {
            "scene_type": scene_type,
            "background": background_meta,
            "center_xy": [cx, cy],
            "box_xyxy": [x1, y1, x2, y2],
            "target_size": [box_w, box_h],
            "rotation": self.rng.uniform(-12.0, 12.0),
            "brightness": self.rng.uniform(0.82, 1.18),
            "contrast": self.rng.uniform(0.82, 1.18),
            "color": self.rng.uniform(0.90, 1.10),
            "blur_radius": self.rng.uniform(0.5, 1.8) if self.rng.random() < self.config.blur_prob else 0.0,
            "jpeg_quality": self.rng.randint(72, 96),
            "occlusion": None,
            "occlusions": [],
        }
        if self.rng.random() < self.config.occlusion_prob:
            nuisance["occlusions"] = self._sample_occlusions((x1, y1, x2, y2), width, height)

        swap_transform = self._sample_object_transform(nuisance)
        context_template = (
            self._sample_context_placements(width, height, [(x1, y1, x2, y2)])
            if scene_type == "multi"
            else []
        )

        best_scene: Optional[Tuple[float, Any, Any, List[Dict[str, Any]], List[Dict[str, Any]]]] = None
        for _ in range(self.config.swap_match_attempts):
            asset_a = self.rng.choice(self.assets_by_class[class_a])
            asset_b = self.rng.choice(self.assets_by_class[class_b])
            swap_a = self._placement(class_a, asset_a.path, (x1, y1, x2, y2), swap_transform, role="swap")
            swap_b = self._placement(class_b, asset_b.path, (x1, y1, x2, y2), swap_transform, role="swap")
            placements_a = [swap_a, *self._clone_placements(context_template)]
            placements_b = [swap_b, *self._clone_placements(context_template)]
            image_a = self._compose_scene(base, placements_a, nuisance)
            image_b = self._compose_scene(base, placements_b, nuisance)
            area_a = float(placements_a[0].get("visible_area_ratio", 0.0))
            area_b = float(placements_b[0].get("visible_area_ratio", 0.0))
            visibility_penalty = max(0.0, self.config.min_visible_area_ratio - area_a) + max(
                0.0,
                self.config.min_visible_area_ratio - area_b,
            )
            mismatch = abs(area_a - area_b) / max(area_a, area_b, 1e-6) + visibility_penalty
            if best_scene is None or mismatch < best_scene[0]:
                best_scene = (mismatch, image_a, image_b, placements_a, placements_b)
            if (
                mismatch <= self.config.swap_visible_area_tolerance
                and area_a >= self.config.min_visible_area_ratio
                and area_b >= self.config.min_visible_area_ratio
            ):
                break
        if best_scene is None:
            raise RuntimeError("Failed to compose TwinSwap pair.")
        visible_area_mismatch, image_a, image_b, placements_a, placements_b = best_scene
        nuisance["swap_visible_area_mismatch"] = float(visible_area_mismatch)

        safe_a = _slug(self.classes[class_a])
        safe_b = _slug(self.classes[class_b])
        stem_a = f"twinswap_p{pair_id:08d}_a_c{class_a}_{safe_a}"
        stem_b = f"twinswap_p{pair_id:08d}_b_c{class_b}_{safe_b}"
        rel_a = Path("images") / split / f"{stem_a}.jpg"
        rel_b = Path("images") / split / f"{stem_b}.jpg"
        image_a.save(root / rel_a, quality=int(nuisance["jpeg_quality"]))
        image_b.save(root / rel_b, quality=int(nuisance["jpeg_quality"]))

        slot_bbox_xywhn = self._xyxy_to_xywhn((x1, y1, x2, y2), width, height)
        visible_bbox_a_xywhn = self._xyxy_to_xywhn(tuple(placements_a[0]["visible_xyxy"]), width, height)
        visible_bbox_b_xywhn = self._xyxy_to_xywhn(tuple(placements_b[0]["visible_xyxy"]), width, height)
        self._write_labels(root / "labels" / split / f"{stem_a}.txt", placements_a, width, height)
        self._write_labels(root / "labels" / split / f"{stem_b}.txt", placements_b, width, height)
        self._write_swap_meta(
            root / "swaps" / split / f"{stem_a}.json",
            pair_id,
            class_a,
            visible_bbox_a_xywhn,
            slot_bbox_xywhn,
            scene_type,
            placements_a[1:],
            width,
            height,
            pair_role,
            identity_kd_enabled,
        )
        self._write_swap_meta(
            root / "swaps" / split / f"{stem_b}.json",
            pair_id,
            class_b,
            visible_bbox_b_xywhn,
            slot_bbox_xywhn,
            scene_type,
            placements_b[1:],
            width,
            height,
            pair_role,
            identity_kd_enabled,
        )

        return PairRecord(
            pair_id=pair_id,
            image_a=str((root / rel_a).resolve()),
            image_b=str((root / rel_b).resolve()),
            class_a=class_a,
            class_b=class_b,
            name_a=self.classes[class_a],
            name_b=self.classes[class_b],
            bbox_xywhn=[float(v) for v in slot_bbox_xywhn],
            slot_bbox_xywhn=[float(v) for v in slot_bbox_xywhn],
            visible_bbox_a_xywhn=[float(v) for v in visible_bbox_a_xywhn],
            visible_bbox_b_xywhn=[float(v) for v in visible_bbox_b_xywhn],
            scene_type=scene_type,
            pair_role=pair_role,
            identity_kd_enabled=bool(identity_kd_enabled),
            objects_a=self._manifest_objects(placements_a, width, height),
            objects_b=self._manifest_objects(placements_b, width, height),
            nuisance=nuisance,
        )

    def _load_background(self, path: Path):
        from PIL import Image, ImageOps

        image = Image.open(path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        if image.size == (self.config.imgsz, self.config.imgsz):
            return image.copy()
        return ImageOps.fit(image, (self.config.imgsz, self.config.imgsz), method=Image.Resampling.BICUBIC)

    def _render_dense_scene(self, root: Path, split: str, dense_id: int) -> None:
        base, background_meta = self._make_background(split, _stable_int(f"dense:{dense_id}"))
        width, height = base.size
        count = self.rng.randint(*self.config.dense_object_count_range)
        placements: List[Dict[str, Any]] = []
        existing: List[Tuple[int, int, int, int]] = []
        for _ in range(count):
            class_id = self.rng.randrange(len(self.classes))
            asset = self.rng.choice(self.assets_by_class[class_id])
            box = self._sample_box(width, height, self._scale_range_for_class(class_id, self.config.distractor_scale_range), existing)
            existing.append(box)
            transform = self._sample_object_transform(
                {
                    "rotation": self.rng.uniform(-20.0, 20.0),
                    "brightness": self.rng.uniform(0.78, 1.22),
                    "contrast": self.rng.uniform(0.78, 1.22),
                    "color": self.rng.uniform(0.86, 1.14),
                }
            )
            placements.append(self._placement(class_id, asset.path, box, transform, role="dense"))
        nuisance = {
            "scene_type": "dense",
            "background": background_meta,
            "blur_radius": self.rng.uniform(0.2, 1.4) if self.rng.random() < self.config.blur_prob else 0.0,
            "jpeg_quality": self.rng.randint(72, 96),
            "occlusions": [],
        }
        if placements and self.rng.random() < self.config.occlusion_prob:
            target_box = self.rng.choice(placements)["xyxy"]
            nuisance["occlusions"] = self._sample_occlusions(tuple(target_box), width, height)
        stem = f"zz_dense_s{split}_{dense_id:08d}"
        image = self._compose_scene(base, placements, nuisance)
        image.save(root / "images" / split / f"{stem}.jpg", quality=int(nuisance["jpeg_quality"]))
        self._write_labels(root / "labels" / split / f"{stem}.txt", placements, width, height)

    def _make_background(self, split: str, pair_id: int):
        if self.background_mode == "directory":
            bg_path = self.rng.choice(self.backgrounds)
            return self._load_background(bg_path), {"type": "directory", "path": str(bg_path)}
        if self.background_mode == "mixed" and self.backgrounds and self.rng.random() < self.config.gnf_real_background_prob:
            bg_path = self.rng.choice(self.backgrounds)
            return self._load_background(bg_path), {"type": "directory", "path": str(bg_path)}
        replayed = None
        profiled = None
        if self.hard_nuisances and self.rng.random() < self.config.hard_nuisance_replay_prob:
            replayed = self.rng.choice(self.hard_nuisances)
            seed = int(replayed["seed"])
        else:
            seed = _stable_int(f"{self.config.seed}:{split}:{pair_id}:gnf")
            if self.hard_nuisances and self.rng.random() < self.config.hard_nuisance_profile_prob:
                profiled = self.rng.choice(self.hard_nuisances)
        profile = dict((profiled or {}).get("background") or {}) if profiled is not None else None
        field = self.gnf_generator.generate(
            seed=seed,
            width=self.config.imgsz,
            height=self.config.imgsz,
            profile=profile,
        )
        if replayed is not None:
            field.metadata["replayed_hard_nuisance"] = True
            field.metadata["source_regret"] = float(replayed.get("regret", 0.0))
            field.metadata["source_pair"] = replayed.get("pair")
        if profiled is not None:
            field.metadata["profiled_hard_nuisance"] = True
            field.metadata["source_regret"] = float(profiled.get("regret", 0.0))
            field.metadata["source_pair"] = profiled.get("pair")
            field.metadata["source_seed"] = int(profiled.get("seed", -1))
        return field.image, field.metadata

    def _sample_context_placements(
        self,
        width: int,
        height: int,
        existing_boxes: List[Tuple[int, int, int, int]],
    ) -> List[Dict[str, Any]]:
        count = self.rng.randint(*self.config.context_object_count_range)
        placements: List[Dict[str, Any]] = []
        for _ in range(count):
            class_id = self.rng.randrange(len(self.classes))
            asset = self.rng.choice(self.assets_by_class[class_id])
            box = self._sample_box(width, height, self._scale_range_for_class(class_id, self.config.distractor_scale_range), existing_boxes)
            existing_boxes.append(box)
            transform = self._sample_object_transform(
                {
                    "rotation": self.rng.uniform(-18.0, 18.0),
                    "brightness": self.rng.uniform(0.80, 1.20),
                    "contrast": self.rng.uniform(0.80, 1.20),
                    "color": self.rng.uniform(0.88, 1.12),
                }
            )
            placements.append(self._placement(class_id, asset.path, box, transform, role="context"))
        return placements

    def _scale_range_for_class(
        self,
        class_id: int,
        base_range: Tuple[float, float],
    ) -> Tuple[float, float]:
        if not self.config.use_kg_scale_prior:
            return base_range
        class_name = self.classes[int(class_id)] if 0 <= int(class_id) < len(self.classes) else ""
        group = ORDINAL_SCALE_GROUP_BY_CLASS.get(class_name, "gear_small")
        group_low, group_high = ORDINAL_SCALE_MULTIPLIER_RANGE.get(group, (0.92, 1.08))
        sampled_multiplier = self.rng.uniform(float(group_low), float(group_high))
        strength = float(self.config.kg_scale_prior_strength)
        jitter = float(self.config.kg_scale_prior_jitter)
        nominal_multiplier = 1.0 + (sampled_multiplier - 1.0) * strength
        spread = 1.0 + abs(sampled_multiplier - 1.0) * jitter * strength
        low_multiplier = nominal_multiplier / max(1e-6, spread)
        high_multiplier = nominal_multiplier * spread
        low_multiplier = max(0.25, low_multiplier)
        high_multiplier = max(low_multiplier, high_multiplier)
        min_scale = max(0.02, float(base_range[0]) * low_multiplier)
        max_scale = float(base_range[1]) * high_multiplier
        cap = min(0.55, max(float(self.config.target_scale_range[1]), float(base_range[1])))
        max_scale = min(cap, max(max_scale, min_scale + 0.01))
        return min_scale, max_scale

    def _sample_box(
        self,
        width: int,
        height: int,
        scale_range: Tuple[float, float],
        existing_boxes: Sequence[Tuple[int, int, int, int]],
    ) -> Tuple[int, int, int, int]:
        best_box: Optional[Tuple[int, int, int, int]] = None
        best_overlap = float("inf")
        for _ in range(self.config.max_placement_attempts):
            scale = self.rng.uniform(*scale_range)
            size = max(12, int(min(width, height) * scale))
            aspect = self.rng.uniform(0.78, 1.28)
            box_w = max(8, int(size * math.sqrt(aspect)))
            box_h = max(8, int(size / math.sqrt(aspect)))
            box_w = min(box_w, width)
            box_h = min(box_h, height)
            x1 = self.rng.randint(0, max(0, width - box_w))
            y1 = self.rng.randint(0, max(0, height - box_h))
            box = (x1, y1, x1 + box_w, y1 + box_h)
            overlap = max((_box_iou(box, existing) for existing in existing_boxes), default=0.0)
            if overlap <= self.config.max_slot_iou:
                return box
            if overlap < best_overlap:
                best_overlap = overlap
                best_box = box
        if best_box is not None:
            return best_box
        size = max(12, int(min(width, height) * scale_range[0]))
        return 0, 0, min(width, size), min(height, size)

    def _sample_object_transform(self, nuisance: Dict[str, Any]) -> Dict[str, float]:
        return {
            "rotation": float(nuisance.get("rotation", 0.0)),
            "brightness": float(nuisance.get("brightness", 1.0)),
            "contrast": float(nuisance.get("contrast", 1.0)),
            "color": float(nuisance.get("color", 1.0)),
        }

    @staticmethod
    def _placement(
        class_id: int,
        asset_path: Path,
        xyxy: Tuple[int, int, int, int],
        transform: Dict[str, float],
        role: str,
    ) -> Dict[str, Any]:
        return {
            "class_id": int(class_id),
            "asset_path": str(asset_path),
            "xyxy": [int(v) for v in xyxy],
            "transform": dict(transform),
            "role": role,
        }

    @staticmethod
    def _clone_placements(placements: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "class_id": int(placement["class_id"]),
                "asset_path": str(placement["asset_path"]),
                "xyxy": [int(v) for v in placement["xyxy"]],
                "slot_xyxy": [int(v) for v in placement.get("slot_xyxy", placement["xyxy"])],
                "transform": dict(placement.get("transform") or {}),
                "role": str(placement.get("role", "")),
            }
            for placement in placements
        ]

    def _sample_occlusions(
        self,
        xyxy: Tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> List[Dict[str, Any]]:
        x1, y1, x2, y2 = xyxy
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        occlusions: List[Dict[str, Any]] = []
        for _ in range(self.rng.randint(1, 3)):
            kind = self.rng.choice(["rectangle", "ellipse", "band", "edge_cut", "polygon"])
            if kind == "band":
                horizontal = self.rng.random() < 0.5
                if horizontal:
                    band_h = self.rng.randint(max(5, box_h // 10), max(8, box_h // 3))
                    y = self.rng.randint(max(0, y1 - box_h // 6), min(height - band_h, y2))
                    occlusions.append({"kind": kind, "xyxy": [0, y, width, y + band_h], "alpha": self.rng.randint(130, 230)})
                else:
                    band_w = self.rng.randint(max(5, box_w // 10), max(8, box_w // 3))
                    x = self.rng.randint(max(0, x1 - box_w // 6), min(width - band_w, x2))
                    occlusions.append({"kind": kind, "xyxy": [x, 0, x + band_w, height], "alpha": self.rng.randint(130, 230)})
            elif kind == "edge_cut":
                side = self.rng.choice(["left", "right", "top", "bottom"])
                frac = self.rng.uniform(0.08, 0.28)
                if side == "left":
                    rect = [x1, y1, x1 + int(box_w * frac), y2]
                elif side == "right":
                    rect = [x2 - int(box_w * frac), y1, x2, y2]
                elif side == "top":
                    rect = [x1, y1, x2, y1 + int(box_h * frac)]
                else:
                    rect = [x1, y2 - int(box_h * frac), x2, y2]
                occlusions.append({"kind": "rectangle", "xyxy": rect, "alpha": self.rng.randint(130, 240)})
            elif kind == "polygon":
                cx = self.rng.randint(x1, x2)
                cy = self.rng.randint(y1, y2)
                radius = self.rng.randint(max(6, min(box_w, box_h) // 10), max(8, min(box_w, box_h) // 3))
                points = []
                for k in range(self.rng.randint(4, 7)):
                    theta = 2 * math.pi * k / 6.0 + self.rng.uniform(-0.5, 0.5)
                    rr = radius * self.rng.uniform(0.45, 1.0)
                    points.append([int(cx + math.cos(theta) * rr), int(cy + math.sin(theta) * rr)])
                occlusions.append({"kind": kind, "points": points, "alpha": self.rng.randint(120, 230)})
            else:
                occ_w = self.rng.randint(max(6, box_w // 8), max(8, box_w // 3))
                occ_h = self.rng.randint(max(6, box_h // 8), max(8, box_h // 3))
                ox = self.rng.randint(x1, max(x1, x2 - occ_w))
                oy = self.rng.randint(y1, max(y1, y2 - occ_h))
                occlusions.append({"kind": kind, "xyxy": [ox, oy, ox + occ_w, oy + occ_h], "alpha": self.rng.randint(130, 235)})
        return occlusions

    def _compose_image(self, base, asset_path: Path, xyxy: Tuple[int, int, int, int], nuisance: Dict[str, Any]):
        placement = self._placement(
            class_id=-1,
            asset_path=asset_path,
            xyxy=xyxy,
            transform=self._sample_object_transform(nuisance),
            role="legacy",
        )
        return self._compose_scene(base, [placement], nuisance)

    def _compose_scene(self, base, placements: Sequence[Dict[str, Any]], nuisance: Dict[str, Any]):
        from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps

        image = base.copy().convert("RGBA")
        rendered_masks = []
        for placement in placements:
            cutout = Image.open(placement["asset_path"]).convert("RGBA")
            cutout = ImageOps.exif_transpose(cutout)
            cutout = self._trim_alpha(cutout)
            transform = dict(placement.get("transform") or {})
            cutout = cutout.rotate(float(transform.get("rotation", 0.0)), expand=True, resample=Image.Resampling.BICUBIC)

            x1, y1, x2, y2 = [int(v) for v in placement["xyxy"]]
            target_w = max(1, x2 - x1)
            target_h = max(1, y2 - y1)
            if self.config.preserve_aspect:
                cutout.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
                canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
                px = (target_w - cutout.width) // 2
                py = (target_h - cutout.height) // 2
                canvas.alpha_composite(cutout, (px, py))
                cutout = canvas
            else:
                cutout = cutout.resize((target_w, target_h), Image.Resampling.LANCZOS)

            full_alpha = Image.new("L", image.size, 0)
            full_alpha.paste(cutout.getchannel("A"), (x1, y1))
            placement["slot_xyxy"] = [x1, y1, x2, y2]

            cutout_rgb = ImageEnhance.Brightness(cutout).enhance(float(transform.get("brightness", 1.0)))
            cutout_rgb = ImageEnhance.Contrast(cutout_rgb).enhance(float(transform.get("contrast", 1.0)))
            cutout_rgb = ImageEnhance.Color(cutout_rgb).enhance(float(transform.get("color", 1.0)))
            image.alpha_composite(cutout_rgb, (x1, y1))
            rendered_masks.append((placement, full_alpha))

        occlusions = list(nuisance.get("occlusions") or [])
        legacy_occlusion = nuisance.get("occlusion")
        if legacy_occlusion:
            occlusions.append({"kind": "rectangle", "xyxy": legacy_occlusion, "alpha": 235})
        occlusion_mask = Image.new("L", image.size, 0)
        if occlusions:
            draw = ImageDraw.Draw(image)
            mask_draw = ImageDraw.Draw(occlusion_mask)
            for occlusion in occlusions:
                alpha = int(occlusion.get("alpha", 220))
                if "xyxy" in occlusion:
                    ox1, oy1, ox2, oy2 = [int(v) for v in occlusion["xyxy"]]
                    sample_x = max(0, min(base.width - 1, int((ox1 + ox2) / 2)))
                    sample_y = max(0, min(base.height - 1, int((oy1 + oy2) / 2)))
                    fill = base.getpixel((sample_x, sample_y))
                    if occlusion.get("kind") == "ellipse":
                        draw.ellipse([ox1, oy1, ox2, oy2], fill=(*fill, alpha))
                        mask_draw.ellipse([ox1, oy1, ox2, oy2], fill=255)
                    else:
                        draw.rectangle([ox1, oy1, ox2, oy2], fill=(*fill, alpha))
                        mask_draw.rectangle([ox1, oy1, ox2, oy2], fill=255)
                elif "points" in occlusion:
                    points = [(int(x), int(y)) for x, y in occlusion["points"]]
                    sx = int(sum(x for x, _ in points) / max(1, len(points)))
                    sy = int(sum(y for _, y in points) / max(1, len(points)))
                    sx = max(0, min(base.width - 1, sx))
                    sy = max(0, min(base.height - 1, sy))
                    fill = base.getpixel((sx, sy))
                    draw.polygon(points, fill=(*fill, alpha))
                    mask_draw.polygon(points, fill=255)

        later_alpha = Image.new("L", image.size, 0)
        for placement, full_alpha in reversed(rendered_masks):
            combined_mask = ImageChops.lighter(occlusion_mask, later_alpha)
            visible_alpha = ImageChops.multiply(full_alpha, ImageChops.invert(combined_mask))
            visible_bbox = visible_alpha.getbbox() or tuple(int(v) for v in placement["slot_xyxy"])
            visible_bbox = self._clip_xyxy(visible_bbox, image.width, image.height)
            slot = tuple(int(v) for v in placement["slot_xyxy"])
            slot_area = max(1.0, float(max(0, slot[2] - slot[0]) * max(0, slot[3] - slot[1])))
            visible_area = float(max(0, visible_bbox[2] - visible_bbox[0]) * max(0, visible_bbox[3] - visible_bbox[1]))
            if visible_area <= 0.0:
                visible_bbox = slot
                visible_area = slot_area
            placement["visible_xyxy"] = [int(v) for v in visible_bbox]
            placement["bbox_xyxy"] = [int(v) for v in visible_bbox]
            placement["visible_area_ratio"] = visible_area / slot_area
            later_alpha = ImageChops.lighter(later_alpha, full_alpha.point(lambda value: 255 if value > 8 else 0))

        output = image.convert("RGB")
        blur_radius = float(nuisance.get("blur_radius") or 0.0)
        if blur_radius > 0:
            output = output.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return output

    @staticmethod
    def _trim_alpha(image):
        alpha = image.getchannel("A")
        bbox = alpha.getbbox()
        return image.crop(bbox) if bbox else image

    @staticmethod
    def _write_label(path: Path, class_id: int, bbox_xywhn: Sequence[float]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        clipped = [min(1.0, max(0.0, float(v))) for v in bbox_xywhn]
        path.write_text(
            f"{class_id} {clipped[0]:.6f} {clipped[1]:.6f} {clipped[2]:.6f} {clipped[3]:.6f}\n",
            encoding="utf-8",
        )

    def _write_labels(self, path: Path, placements: Sequence[Dict[str, Any]], width: int, height: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for placement in placements:
            class_id = int(placement["class_id"])
            label_xyxy = placement.get("visible_xyxy", placement.get("bbox_xyxy", placement["xyxy"]))
            bbox_xywhn = self._xyxy_to_xywhn(tuple(int(v) for v in label_xyxy), width, height)
            clipped = [min(1.0, max(0.0, float(v))) for v in bbox_xywhn]
            lines.append(f"{class_id} {clipped[0]:.6f} {clipped[1]:.6f} {clipped[2]:.6f} {clipped[3]:.6f}")
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _export_real_presence_anchors(self, root: Path, split: str) -> None:
        if split == "mine":
            return
        limit = (
            int(self.config.real_anchor_train_per_class)
            if split == "train"
            else int(self.config.real_anchor_val_per_class)
        )
        if limit <= 0:
            return
        if self.config.real_anchor_video_root is None or self.config.real_anchor_mask_root is None:
            return
        video_root = Path(self.config.real_anchor_video_root)
        mask_root = Path(self.config.real_anchor_mask_root)
        if not video_root.exists() or not mask_root.exists():
            return

        import cv2
        import numpy as np

        from .mask_video_importer import (
            DEFAULT_SOURCE_TO_CLASS,
            _base_source_name,
            _bbox_from_mask,
            _iter_mask_frames,
            _read_frame,
            _read_mask,
            _source_stem,
        )

        class_to_candidates: Dict[str, List[Tuple[Path, Any]]] = {name: [] for name in self.classes}
        for mask_dir in sorted([item for item in mask_root.iterdir() if item.is_dir()], key=lambda item: item.name.lower()):
            source = _source_stem(mask_dir)
            if source.endswith("_mesh") and not self.config.real_anchor_include_mesh:
                continue
            class_name = DEFAULT_SOURCE_TO_CLASS.get(_base_source_name(source))
            if class_name not in class_to_candidates:
                continue
            class_to_candidates[class_name].extend((mask_dir, frame) for frame in _iter_mask_frames(mask_dir))

        records = []
        class_to_id = {name: index for index, name in enumerate(self.classes)}
        repeat_count = int(self.config.real_anchor_train_repeat) if split == "train" else 1
        for class_name in self.classes:
            candidates = sorted(
                class_to_candidates.get(class_name, []),
                key=lambda item: (_source_stem(item[0]), int(item[1].frame_index)),
            )
            selected = self._select_evenly(candidates, limit)
            for mask_dir, mask_frame in selected:
                source = _source_stem(mask_dir)
                video_path = video_root / f"{source}.mp4"
                if not video_path.exists():
                    continue
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    continue
                try:
                    frame = _read_frame(capture, int(mask_frame.frame_index))
                finally:
                    capture.release()
                if frame is None:
                    continue
                height, width = frame.shape[:2]
                mask = _read_mask(mask_frame.path, (width, height))
                if getattr(mask, "ndim", 2) > 2:
                    mask = mask[..., 0]
                area_ratio = float((mask > 0).mean())
                if (
                    area_ratio < float(self.config.real_anchor_min_area_ratio)
                    or area_ratio > float(self.config.real_anchor_max_area_ratio)
                ):
                    continue
                box = _bbox_from_mask(mask)
                if box is None:
                    continue
                label_line = self._label_line_for_box(class_to_id[class_name], box, width, height)
                for repeat_index in range(max(1, repeat_count)):
                    suffix = f"_r{repeat_index:02d}" if repeat_count > 1 else ""
                    stem = f"zz_real_mask_{_slug(class_name)}_{_slug(source)}_f{int(mask_frame.frame_index):06d}{suffix}"
                    image_path = root / "images" / split / f"{stem}.jpg"
                    label_path = root / "labels" / split / f"{stem}.txt"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    label_path.write_text(label_line + "\n", encoding="utf-8")
                    records.append(
                        {
                            "image": str(image_path.resolve()),
                            "label": str(label_path.resolve()),
                            "split": split,
                            "class_name": class_name,
                            "class_id": int(class_to_id[class_name]),
                            "source": source,
                            "frame_index": int(mask_frame.frame_index),
                            "mask_area_ratio": area_ratio,
                            "repeat": int(repeat_index),
                            "role": "real_presence_anchor",
                        }
                    )
        if records:
            _write_jsonl(root / f"real_presence_anchors_{split}.jsonl", records)

    def _import_pseudo_real_frames(self, root: Path, split: str) -> None:
        if split == "mine" or self.config.pseudo_real_source_root is None:
            return
        source_root = Path(self.config.pseudo_real_source_root)
        source_image_dir = source_root / "images" / split
        source_label_dir = source_root / "labels" / split
        if not source_image_dir.exists() or not source_label_dir.exists():
            return
        limit = (
            int(self.config.pseudo_real_train_limit)
            if split == "train"
            else int(self.config.pseudo_real_val_limit)
        )
        prefix = str(self.config.pseudo_real_filename_prefix)
        candidates = []
        for path in sorted(source_image_dir.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            if prefix and prefix != "*" and not path.name.startswith(prefix):
                continue
            candidates.append(path)
        selected = candidates if limit <= 0 else self._select_evenly(candidates, limit)
        records: List[Dict[str, Any]] = []
        for image_source in selected:
            label_source = source_label_dir / f"{image_source.stem}.txt"
            if not label_source.exists():
                continue
            image_target = root / "images" / split / image_source.name
            label_target = root / "labels" / split / label_source.name
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_source, image_target)
            shutil.copy2(label_source, label_target)
            records.append(
                {
                    "image": str(image_target.resolve()),
                    "label": str(label_target.resolve()),
                    "source_image": str(image_source.resolve()),
                    "source_label": str(label_source.resolve()),
                    "split": split,
                    "role": "pseudo_real_presence_anchor",
                }
            )
        if records:
            _write_jsonl(root / f"pseudo_real_frames_{split}.jsonl", records)

    @staticmethod
    def _select_evenly(items: Sequence[Any], limit: int) -> List[Any]:
        if limit <= 0 or not items:
            return []
        if len(items) <= limit:
            return list(items)
        if limit == 1:
            return [items[len(items) // 2]]
        import numpy as np

        indices = np.linspace(0, len(items) - 1, num=limit, dtype=np.int64)
        return [items[int(index)] for index in indices]

    @staticmethod
    def _label_line_for_box(class_id: int, box_xyxy: Tuple[int, int, int, int], width: int, height: int) -> str:
        x1, y1, x2, y2 = box_xyxy
        x1 = max(0, min(int(x1), width - 1))
        y1 = max(0, min(int(y1), height - 1))
        x2 = max(0, min(int(x2), width))
        y2 = max(0, min(int(y2), height))
        cx = ((x1 + x2) * 0.5) / max(1, width)
        cy = ((y1 + y2) * 0.5) / max(1, height)
        bw = max(1, x2 - x1) / max(1, width)
        bh = max(1, y2 - y1) / max(1, height)
        return f"{int(class_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    @staticmethod
    def _xyxy_to_xywhn(xyxy: Tuple[int, int, int, int], width: int, height: int) -> List[float]:
        x1, y1, x2, y2 = xyxy
        return [
            ((x1 + x2) / 2.0) / width,
            ((y1 + y2) / 2.0) / height,
            (x2 - x1) / width,
            (y2 - y1) / height,
        ]

    @staticmethod
    def _clip_xyxy(xyxy: Sequence[int], width: int, height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        return x1, y1, x2, y2

    def _write_swap_meta(
        self,
        path: Path,
        pair_id: int,
        class_id: int,
        bbox_xywhn: Sequence[float],
        slot_bbox_xywhn: Sequence[float],
        scene_type: str,
        context: Sequence[Dict[str, Any]],
        width: int,
        height: int,
        pair_role: str,
        identity_kd_enabled: bool,
    ) -> None:
        _write_json(
            path,
            {
                "pair_id": int(pair_id),
                "class_id": int(class_id),
                "bbox_xywhn": [float(v) for v in bbox_xywhn],
                "visible_bbox_xywhn": [float(v) for v in bbox_xywhn],
                "slot_bbox_xywhn": [float(v) for v in slot_bbox_xywhn],
                "scene_type": scene_type,
                "role": "swap",
                "pair_role": str(pair_role),
                "identity_kd_enabled": bool(identity_kd_enabled),
                "context": self._manifest_objects(context, width, height),
            },
        )

    def _manifest_objects(self, placements: Sequence[Dict[str, Any]], width: int, height: int) -> List[Dict[str, Any]]:
        objects: List[Dict[str, Any]] = []
        for placement in placements:
            class_id = int(placement["class_id"])
            xyxy = [int(v) for v in placement.get("visible_xyxy", placement.get("bbox_xyxy", placement["xyxy"]))]
            slot_xyxy = [int(v) for v in placement.get("slot_xyxy", placement["xyxy"])]
            objects.append(
                {
                    "class_id": class_id,
                    "class_name": self.classes[class_id] if 0 <= class_id < len(self.classes) else str(class_id),
                    "bbox_xywhn": self._xyxy_to_xywhn(tuple(xyxy), width, height),
                    "visible_bbox_xywhn": self._xyxy_to_xywhn(tuple(xyxy), width, height),
                    "slot_bbox_xywhn": self._xyxy_to_xywhn(tuple(slot_xyxy), width, height),
                    "xyxy": xyxy,
                    "slot_xyxy": slot_xyxy,
                    "visible_area_ratio": float(placement.get("visible_area_ratio", 0.0)),
                    "role": str(placement.get("role", "")),
                }
            )
        return objects


class TwinSwapDetectionTrainerMixin:
    """Mixin adding TwinSwap metadata parsing and pair-preserving dataloaders."""

    def preprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch = super().preprocess_batch(batch)  # type: ignore[misc]
        if "im_file" in batch:
            pair_ids, classes = self._parse_batch_twin_meta(batch["im_file"])
            if pair_ids:
                import torch

                batch["twinswap_pair_id"] = torch.tensor(pair_ids, dtype=torch.long, device=self.device)
                batch["twinswap_class"] = torch.tensor(classes, dtype=torch.long, device=self.device)
                swap_bboxes, contexts, identity_flags = self._load_batch_swap_metadata(batch["im_file"])
                if swap_bboxes:
                    batch["twinswap_bbox"] = torch.tensor(swap_bboxes, dtype=torch.float32, device=self.device)
                if contexts:
                    batch["twinswap_context"] = contexts
                if identity_flags:
                    batch["twinswap_identity_kd_enabled"] = torch.tensor(
                        identity_flags,
                        dtype=torch.bool,
                        device=self.device,
                    )
        return batch

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        keep_pairs = bool(getattr(self, "twinswap_args", {}).get("twinswap_keep_pairs", True))
        if mode != "train" or not keep_pairs:
            return super().get_dataloader(dataset_path, batch_size, rank, mode)  # type: ignore[misc]

        import numpy as np
        from ultralytics.data import build_dataloader
        from ultralytics.utils import LOGGER
        from ultralytics.utils.torch_utils import torch_distributed_zero_first

        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        if getattr(dataset, "rect", False) and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with TwinSwap pair-preserving batches; disabling shuffle.")
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=False,
            rank=rank,
            drop_last=False,
        )

    @staticmethod
    def _parse_batch_twin_meta(paths: Sequence[str]) -> Tuple[List[int], List[int]]:
        pair_ids: List[int] = []
        classes: List[int] = []
        for path in paths:
            match = PAIR_RE.search(Path(str(path)).name)
            if match is None:
                return [], []
            pair_ids.append(int(match.group("pair")))
            classes.append(int(match.group("class")))
        return pair_ids, classes

    @staticmethod
    def _load_batch_swap_bboxes(paths: Sequence[str]) -> List[List[float]]:
        bboxes, _, _ = TwinSwapDetectionTrainerMixin._load_batch_swap_metadata(paths)
        return bboxes

    @staticmethod
    def _load_batch_swap_metadata(paths: Sequence[str]) -> Tuple[List[List[float]], List[List[Dict[str, Any]]], List[bool]]:
        bboxes: List[List[float]] = []
        contexts: List[List[Dict[str, Any]]] = []
        identity_flags: List[bool] = []
        for path in paths:
            image_path = Path(str(path))
            try:
                root = image_path.parents[2]
                split = image_path.parent.name
            except IndexError:
                return [], [], []
            meta_path = root / "swaps" / split / f"{image_path.stem}.json"
            if not meta_path.exists():
                return [], [], []
            try:
                meta = _load_json(meta_path)
                bbox = [float(v) for v in meta.get("visible_bbox_xywhn", meta["bbox_xywhn"])]
                context = list(meta.get("context") or [])
                identity_enabled = bool(meta.get("identity_kd_enabled", True))
            except Exception:
                return [], [], []
            if len(bbox) != 4:
                return [], [], []
            bboxes.append(bbox)
            contexts.append(context)
            identity_flags.append(identity_enabled)
        return bboxes, contexts, identity_flags


def _xywh_to_xyxy(boxes: Any) -> Any:
    import torch

    output = boxes.clone() if isinstance(boxes, torch.Tensor) else torch.as_tensor(boxes).clone()
    output[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    output[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    output[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    output[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return output


def _box_iou(first: Tuple[float, float, float, float], second: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-6, area_a + area_b - inter)


def _responses_for_boxes_from_parsed(
    preds: Dict[str, Any],
    boxes_xywhn: Any,
    stride: Any,
    device: Any,
    topk: int,
) -> Any:
    import torch
    from ultralytics.utils.tal import make_anchors

    pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
    stride = torch.as_tensor(stride, device=pred_scores.device, dtype=pred_scores.dtype)
    batch_size, _, num_classes = pred_scores.shape
    responses = torch.zeros(batch_size, num_classes, device=device, dtype=pred_scores.dtype)

    anchor_points, stride_tensor = make_anchors(preds["feats"], stride, 0.5)
    anchor_xy = anchor_points * stride_tensor
    image_hw = torch.tensor(preds["feats"][0].shape[2:], device=device, dtype=pred_scores.dtype) * stride[0]
    image_h, image_w = image_hw[0], image_hw[1]

    boxes_tensor = None
    if boxes_xywhn is not None:
        boxes_tensor = torch.as_tensor(boxes_xywhn, device=device, dtype=pred_scores.dtype).reshape(-1, 4)
        boxes_xyxy = _xywh_to_xyxy(boxes_tensor)
        scale = torch.tensor([image_w, image_h, image_w, image_h], device=device, dtype=pred_scores.dtype)
        boxes_xyxy = boxes_xyxy * scale
    else:
        boxes_xyxy = None

    for image_index in range(batch_size):
        if boxes_xyxy is None or image_index >= boxes_xyxy.shape[0]:
            values = pred_scores[image_index].topk(min(topk, pred_scores.shape[1]), dim=0).values
            responses[image_index] = values.mean(dim=0)
            continue

        x1, y1, x2, y2 = boxes_xyxy[image_index]
        inside = (
            (anchor_xy[:, 0] >= x1)
            & (anchor_xy[:, 0] <= x2)
            & (anchor_xy[:, 1] >= y1)
            & (anchor_xy[:, 1] <= y2)
        )
        if not bool(inside.any()):
            center = torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
            distance = (anchor_xy - center).pow(2).sum(dim=1)
            anchor_index = torch.topk(distance, k=min(topk, distance.numel()), largest=False).indices
        else:
            anchor_index = torch.where(inside)[0]
            if anchor_index.numel() > topk:
                local_scores = pred_scores[image_index, anchor_index].amax(dim=1)
                keep = torch.topk(local_scores, k=topk, largest=True).indices
                anchor_index = anchor_index[keep]
        responses[image_index] = pred_scores[image_index, anchor_index].mean(dim=0)
    return responses


def _responses_for_label_boxes_from_parsed(
    preds: Dict[str, Any],
    boxes_xywhn: Any,
    batch_indices: Any,
    stride: Any,
    device: Any,
    topk: int,
) -> Any:
    import torch
    from ultralytics.utils.tal import make_anchors

    pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
    stride = torch.as_tensor(stride, device=pred_scores.device, dtype=pred_scores.dtype)
    batch_size, _, num_classes = pred_scores.shape
    boxes_tensor = torch.as_tensor(boxes_xywhn, device=device, dtype=pred_scores.dtype).reshape(-1, 4)
    batch_tensor = torch.as_tensor(batch_indices, device=device, dtype=torch.long).reshape(-1)
    if boxes_tensor.numel() == 0 or batch_tensor.numel() == 0:
        return torch.zeros(0, num_classes, device=device, dtype=pred_scores.dtype)

    anchor_points, stride_tensor = make_anchors(preds["feats"], stride, 0.5)
    anchor_xy = anchor_points * stride_tensor
    image_hw = torch.tensor(preds["feats"][0].shape[2:], device=device, dtype=pred_scores.dtype) * stride[0]
    image_h, image_w = image_hw[0], image_hw[1]
    boxes_xyxy = _xywh_to_xyxy(boxes_tensor)
    scale = torch.tensor([image_w, image_h, image_w, image_h], device=device, dtype=pred_scores.dtype)
    boxes_xyxy = boxes_xyxy * scale

    responses = []
    for box, image_index_tensor in zip(boxes_xyxy, batch_tensor):
        image_index = int(image_index_tensor.item())
        if image_index < 0 or image_index >= batch_size:
            continue
        x1, y1, x2, y2 = box
        inside = (
            (anchor_xy[:, 0] >= x1)
            & (anchor_xy[:, 0] <= x2)
            & (anchor_xy[:, 1] >= y1)
            & (anchor_xy[:, 1] <= y2)
        )
        if not bool(inside.any()):
            center = torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
            distance = (anchor_xy - center).pow(2).sum(dim=1)
            anchor_index = torch.topk(distance, k=min(topk, distance.numel()), largest=False).indices
        else:
            anchor_index = torch.where(inside)[0]
            if anchor_index.numel() > topk:
                local_scores = pred_scores[image_index, anchor_index].amax(dim=1)
                keep = torch.topk(local_scores, k=topk, largest=True).indices
                anchor_index = anchor_index[keep]
        responses.append(pred_scores[image_index, anchor_index].mean(dim=0))
    if not responses:
        return torch.zeros(0, num_classes, device=device, dtype=pred_scores.dtype)
    return torch.stack(responses, dim=0)


class TwinSwapDetectionLoss:
    """Wrapper around Ultralytics v8 detection loss with response distillation."""

    def __init__(
        self,
        model: Any,
        teacher_model: Any,
        base_loss: Optional[Any] = None,
        kd_weight: float = 0.35,
        temperature: float = 2.0,
        topk: int = 24,
        margin_weight: float = 0.75,
        hard_pair_margin_weight: float = 0.75,
        hard_pair_target_margin: float = 0.50,
        presence_weight: float = 0.0,
        presence_alignment_weight: float = 0.0,
        real_presence_weight: float = 0.0,
        context_weight: float = 0.0,
    ) -> None:
        from ultralytics.utils.loss import v8DetectionLoss

        self.base = base_loss if base_loss is not None else v8DetectionLoss(model)
        self.teacher_model = teacher_model
        self.kd_weight = max(0.0, float(kd_weight))
        self.temperature = max(0.25, float(temperature))
        self.topk = max(1, int(topk))
        self.margin_weight = max(0.0, float(margin_weight))
        self.hard_pair_margin_weight = max(0.0, float(hard_pair_margin_weight))
        self.hard_pair_target_margin = max(0.0, float(hard_pair_target_margin))
        self.presence_weight = max(0.0, float(presence_weight))
        self.presence_alignment_weight = max(0.0, float(presence_alignment_weight))
        self.real_presence_weight = max(0.0, float(real_presence_weight))
        self.context_weight = max(0.0, float(context_weight))
        attr_loss = getattr(self.base, "one2many", self.base)
        self.device = attr_loss.device
        self.stride = attr_loss.stride
        self.nc = attr_loss.nc
        self.teacher_stride = getattr(getattr(teacher_model, "model", [None])[-1], "stride", self.stride)
        self.teacher_half = False
        if teacher_model is not None:
            try:
                first_param = next(teacher_model.parameters())
                self.teacher_half = bool(first_param.is_cuda and first_param.dtype == __import__("torch").float16)
            except StopIteration:
                self.teacher_half = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def __call__(self, preds: Any, batch: Dict[str, Any]):
        loss, loss_items = self.base(preds, batch)
        if self.kd_weight <= 0 or self.teacher_model is None:
            return loss, loss_items

        parsed = self._parse_response_output(preds)
        if "twinswap_pair_id" in batch and "twinswap_class" in batch:
            kd_loss = self._counterfactual_response_loss(parsed, batch)
        else:
            kd_loss = self._real_presence_anchor_loss(parsed, batch)
        if kd_loss is None:
            return loss, loss_items

        batch_size = int(parsed["scores"].shape[0])
        loss = loss.clone()
        loss_items = loss_items.clone()
        loss[1] = loss[1] + kd_loss * batch_size
        loss_items[1] = loss_items[1] + kd_loss.detach()
        return loss, loss_items

    def _parse_response_output(self, preds: Any) -> Dict[str, Any]:
        parsed = preds
        parse_output = getattr(self.base, "parse_output", None)
        if callable(parse_output):
            try:
                parsed = parse_output(preds)
            except (KeyError, TypeError, IndexError):
                parsed = preds
        if isinstance(parsed, tuple):
            parsed = parsed[1]
        if isinstance(parsed, dict):
            if "scores" in parsed and "feats" in parsed:
                return parsed
            for branch in ("one2many", "one2one"):
                branch_preds = parsed.get(branch)
                if isinstance(branch_preds, dict) and "scores" in branch_preds and "feats" in branch_preds:
                    return branch_preds
        raise KeyError("TwinSwap KD could not find YOLO response tensors in model output.")

    @staticmethod
    def _presence_energy(responses: Any, class_count: int) -> Any:
        import torch

        return torch.logsumexp(responses[:, :class_count], dim=1)

    @staticmethod
    def _identity_distribution(responses: Any, class_count: int, temperature: float) -> Any:
        import torch.nn.functional as F

        return F.softmax(responses[:, :class_count] / max(0.25, float(temperature)), dim=1)

    def _teacher_forward(self, images: Any) -> Any:
        import torch

        teacher_images = images.detach()
        if self.teacher_half and teacher_images.is_cuda:
            teacher_images = teacher_images.half()
        with torch.inference_mode():
            if teacher_images.is_cuda:
                try:
                    with torch.amp.autocast("cuda", enabled=True):
                        return self.teacher_model(teacher_images)
                except AttributeError:
                    with torch.cuda.amp.autocast(enabled=True):
                        return self.teacher_model(teacher_images)
            return self.teacher_model(teacher_images)

    def _counterfactual_response_loss(self, student_preds: Dict[str, Any], batch: Dict[str, Any]):
        import torch
        import torch.nn.functional as F

        pair_ids = batch["twinswap_pair_id"]
        twin_classes = batch["twinswap_class"]
        if pair_ids.numel() < 2:
            return None

        teacher_raw = self._teacher_forward(batch["img"])
        teacher_preds = self._parse_response_output(teacher_raw)

        student_resp = self._object_class_response(student_preds, batch, stride=self.stride)
        with torch.no_grad():
            teacher_resp = self._object_class_response(teacher_preds, batch, stride=self.teacher_stride)
        teacher_resp = teacher_resp.to(device=student_resp.device, dtype=student_resp.dtype)
        del teacher_raw, teacher_preds

        losses: List[Any] = []
        unique_pair_ids = torch.unique(pair_ids)
        temp = self.temperature
        identity_flags = batch.get("twinswap_identity_kd_enabled")
        for pair_id in unique_pair_ids.tolist():
            idx = torch.where(pair_ids == int(pair_id))[0]
            if idx.numel() != 2:
                continue
            left, right = int(idx[0].item()), int(idx[1].item())
            cls_left = int(twin_classes[left].item())
            cls_right = int(twin_classes[right].item())
            response_classes = min(student_resp.shape[1], teacher_resp.shape[1])
            if cls_left == cls_right or cls_left >= response_classes or cls_right >= response_classes:
                continue
            class_index = torch.tensor([cls_left, cls_right], device=self.device, dtype=torch.long)
            identity_enabled = True
            if identity_flags is not None and int(identity_flags.numel()) > max(left, right):
                identity_enabled = bool(identity_flags[left].item()) and bool(identity_flags[right].item())

            pair_terms: List[Any] = []
            if identity_enabled:
                s_identity = self._identity_distribution(student_resp[[left, right]], response_classes, temp)
                t_identity = self._identity_distribution(teacher_resp[[left, right]], response_classes, temp).detach()
                identity_delta_loss = F.smooth_l1_loss(s_identity[0] - s_identity[1], t_identity[0] - t_identity[1]) * (
                    temp * temp
                )

                s_delta = student_resp[left, class_index] - student_resp[right, class_index]
                t_delta = teacher_resp[left, class_index] - teacher_resp[right, class_index]
                response_loss = F.kl_div(
                    F.log_softmax(s_delta / temp, dim=0),
                    F.softmax(t_delta.detach() / temp, dim=0),
                    reduction="batchmean",
                ) * (temp * temp)

                s_margin = torch.stack(
                    [
                        student_resp[left, cls_left] - student_resp[left, cls_right],
                        student_resp[right, cls_right] - student_resp[right, cls_left],
                    ]
                )
                t_margin = torch.stack(
                    [
                        teacher_resp[left, cls_left] - teacher_resp[left, cls_right],
                        teacher_resp[right, cls_right] - teacher_resp[right, cls_left],
                    ]
                )
                margin_loss = F.smooth_l1_loss(s_margin / temp, t_margin.detach() / temp) * (temp * temp)
                supervised_margin_loss = F.relu(self.hard_pair_target_margin - s_margin).mean()
                pair_terms.append(
                    response_loss
                    + identity_delta_loss
                    + self.margin_weight * margin_loss
                    + self.hard_pair_margin_weight * supervised_margin_loss
                )

            # Legacy PI-TwinSwap presence invariance/alignment. Disabled by
            # default so the main loss remains pure TwinSwap identity KD.
            if self.presence_weight > 0:
                s_energy = self._presence_energy(student_resp[[left, right]], response_classes)
                t_energy = self._presence_energy(teacher_resp[[left, right]], response_classes).detach()
                presence_inv = F.smooth_l1_loss(s_energy[0] - s_energy[1], t_energy[0] - t_energy[1])
                presence_align = F.smooth_l1_loss(s_energy, t_energy)
                pair_terms.append(self.presence_weight * (presence_inv + self.presence_alignment_weight * presence_align))

            if pair_terms:
                losses.append(torch.stack(pair_terms).sum())

        # Legacy PI-TwinSwap context invariance. Disabled by default.
        if self.context_weight > 0 and "twinswap_context" in batch:
            context_loss = self._context_invariance_loss(student_preds, batch, pair_ids)
            if context_loss is not None:
                losses.append(self.context_weight * context_loss)

        if not losses:
            return None
        return self.kd_weight * torch.stack(losses).mean()

    def _object_class_response(self, preds: Dict[str, Any], batch: Dict[str, Any], stride: Any):
        import torch

        batch_idx = batch["batch_idx"].long()
        bboxes = batch["bboxes"].to(self.device)
        swap_bboxes = batch.get("twinswap_bbox")
        if swap_bboxes is not None:
            return _responses_for_boxes_from_parsed(preds, swap_bboxes, stride, self.device, self.topk)

        batch_size = int(preds["scores"].shape[0])
        fallback_boxes = []
        for image_index in range(batch_size):
            box_indices = torch.where(batch_idx == image_index)[0]
            if box_indices.numel() > 0:
                fallback_boxes.append(bboxes[box_indices[0]].detach().cpu().tolist())
            else:
                fallback_boxes.append([0.5, 0.5, 1.0, 1.0])
        return _responses_for_boxes_from_parsed(preds, fallback_boxes, stride, self.device, self.topk)

    def _real_presence_anchor_loss(self, student_preds: Dict[str, Any], batch: Dict[str, Any]):
        import torch
        import torch.nn.functional as F

        if self.real_presence_weight <= 0:
            return None
        paths = [Path(str(path)).name for path in batch.get("im_file", [])]
        if not paths:
            return None
        real_image_indices = {
            index
            for index, name in enumerate(paths)
            if name.startswith("zz_real_mask_") or name.startswith("zzz_real_pseudo_")
        }
        if not real_image_indices:
            return None
        batch_idx = batch["batch_idx"].long().to(self.device)
        bboxes = batch["bboxes"].to(self.device)
        if batch_idx.numel() == 0 or bboxes.numel() == 0:
            return None
        selected = torch.tensor(
            [int(index.item()) in real_image_indices for index in batch_idx],
            device=self.device,
            dtype=torch.bool,
        )
        if not bool(selected.any()):
            return None
        selected_boxes = bboxes[selected]
        selected_batch_idx = batch_idx[selected]
        teacher_raw = self._teacher_forward(batch["img"])
        teacher_preds = self._parse_response_output(teacher_raw)
        with torch.no_grad():
            teacher_resp = _responses_for_label_boxes_from_parsed(
                teacher_preds,
                selected_boxes,
                selected_batch_idx,
                self.teacher_stride,
                self.device,
                self.topk,
            )
        del teacher_raw, teacher_preds
        student_resp = _responses_for_label_boxes_from_parsed(
            student_preds,
            selected_boxes,
            selected_batch_idx,
            self.stride,
            self.device,
            self.topk,
        )
        teacher_resp = teacher_resp.to(device=student_resp.device, dtype=student_resp.dtype)
        class_count = min(student_resp.shape[1], teacher_resp.shape[1], self.nc)
        if class_count <= 0 or student_resp.shape[0] == 0 or teacher_resp.shape[0] == 0:
            return None
        count = min(student_resp.shape[0], teacher_resp.shape[0])
        student_energy = self._presence_energy(student_resp[:count], class_count)
        teacher_energy = self._presence_energy(teacher_resp[:count], class_count).detach()
        return self.kd_weight * self.real_presence_weight * F.smooth_l1_loss(student_energy, teacher_energy)

    def _context_invariance_loss(self, student_preds: Dict[str, Any], batch: Dict[str, Any], pair_ids: Any):
        import torch
        import torch.nn.functional as F

        contexts = batch.get("twinswap_context") or []
        if len(contexts) < int(pair_ids.numel()):
            return None

        losses: List[Any] = []
        temp = self.temperature
        for pair_id in torch.unique(pair_ids).tolist():
            idx = torch.where(pair_ids == int(pair_id))[0]
            if idx.numel() != 2:
                continue
            left, right = int(idx[0].item()), int(idx[1].item())
            left_context = contexts[left] if left < len(contexts) else []
            right_context = contexts[right] if right < len(contexts) else []
            for obj_left, obj_right in zip(left_context, right_context):
                if int(obj_left.get("class_id", -1)) != int(obj_right.get("class_id", -2)):
                    continue
                bbox_left = obj_left.get("visible_bbox_xywhn") or obj_left.get("bbox_xywhn")
                bbox_right = obj_right.get("visible_bbox_xywhn") or obj_right.get("bbox_xywhn")
                if not bbox_left or not bbox_right:
                    continue
                boxes = torch.tensor([bbox_left, bbox_right], device=self.device, dtype=student_preds["scores"].dtype)
                responses = _responses_for_boxes_from_parsed(
                    {
                        "scores": student_preds["scores"][[left, right]],
                        "feats": student_preds["feats"],
                    },
                    boxes,
                    self.stride,
                    self.device,
                    self.topk,
                )
                class_count = min(responses.shape[1], self.nc)
                if class_count <= 0:
                    continue
                energy = self._presence_energy(responses, class_count)
                identity = self._identity_distribution(responses, class_count, temp)
                presence_loss = F.smooth_l1_loss(energy[0], energy[1])
                identity_loss = F.smooth_l1_loss(identity[0], identity[1]) * (temp * temp)
                losses.append(presence_loss + identity_loss)
        if not losses:
            return None
        return torch.stack(losses).mean()


def _build_twinswap_detection_trainer_class():
    from ultralytics.cfg import DEFAULT_CFG
    from ultralytics.models.yolo.detect import DetectionTrainer

    twinswap_defaults = {
        "twinswap_teacher": None,
        "twinswap_kd_weight": 0.0,
        "twinswap_temperature": 2.0,
        "twinswap_topk": 24,
        "twinswap_margin_weight": 0.75,
        "twinswap_hard_pair_margin_weight": 0.75,
        "twinswap_hard_pair_target_margin": 0.50,
        "twinswap_presence_weight": 0.0,
        "twinswap_presence_alignment_weight": 0.0,
        "twinswap_real_presence_weight": 0.0,
        "twinswap_context_weight": 0.0,
        "twinswap_keep_pairs": True,
    }

    class TwinSwapDetectionTrainer(TwinSwapDetectionTrainerMixin, DetectionTrainer):
        """Ultralytics DetectionTrainer with TwinSwap distillation enabled."""

        def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
            overrides = dict(overrides or {})
            twinswap_args = {}
            for key in list(overrides):
                if key in twinswap_defaults:
                    twinswap_args[key] = overrides.pop(key)
            super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
            self.twinswap_args = {
                key: twinswap_args.get(key, default) for key, default in twinswap_defaults.items()
            }

        def _setup_train(self) -> None:
            super()._setup_train()
            twinswap_args = getattr(self, "twinswap_args", twinswap_defaults)
            teacher_path = twinswap_args.get("twinswap_teacher")
            kd_weight = float(twinswap_args.get("twinswap_kd_weight", 0.0) or 0.0)
            if not teacher_path or kd_weight <= 0:
                return

            from ultralytics import YOLO
            from ultralytics.utils import LOGGER
            from ultralytics.utils.torch_utils import unwrap_model

            teacher = YOLO(str(teacher_path)).model
            teacher = teacher.to(self.device).eval()
            if str(self.device).startswith("cuda"):
                teacher = teacher.half()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)

            base_model = unwrap_model(self.model)
            base_loss = getattr(base_model, "criterion", None)
            if base_loss is None and hasattr(base_model, "init_criterion"):
                base_loss = base_model.init_criterion()
            base_model.criterion = TwinSwapDetectionLoss(
                model=base_model,
                teacher_model=teacher,
                base_loss=base_loss,
                kd_weight=kd_weight,
                temperature=float(twinswap_args.get("twinswap_temperature", 2.0)),
                topk=int(twinswap_args.get("twinswap_topk", 24)),
                margin_weight=float(twinswap_args.get("twinswap_margin_weight", 0.75)),
                hard_pair_margin_weight=float(twinswap_args.get("twinswap_hard_pair_margin_weight", 0.75)),
                hard_pair_target_margin=float(twinswap_args.get("twinswap_hard_pair_target_margin", 0.50)),
                presence_weight=float(twinswap_args.get("twinswap_presence_weight", 0.0)),
                presence_alignment_weight=float(twinswap_args.get("twinswap_presence_alignment_weight", 0.0)),
                real_presence_weight=float(twinswap_args.get("twinswap_real_presence_weight", 0.0)),
                context_weight=float(twinswap_args.get("twinswap_context_weight", 0.0)),
            )
            LOGGER.info(
                "TwinSwap KD enabled: teacher=%s, weight=%.3f, temperature=%.2f",
                teacher_path,
                kd_weight,
                float(twinswap_args.get("twinswap_temperature", 2.0)),
            )

    return TwinSwapDetectionTrainer


TwinSwapDetectionTrainer = _build_twinswap_detection_trainer_class()


class _RawSlotResponseScorer:
    """Compute slot-local class margins from raw YOLO head responses."""

    def __init__(self, weights: str | Path, config: TwinSwapConfig) -> None:
        import torch
        from ultralytics import YOLO
        from ultralytics.utils.loss import v8DetectionLoss

        self.config = config
        device_text = str(config.device)
        if device_text.isdigit():
            self.device = torch.device(f"cuda:{device_text}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_text)
        self.yolo = YOLO(str(weights))
        self.model = self.yolo.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.parser = v8DetectionLoss(self.model)
        self.stride = self.parser.stride

    def pair_margin(self, record: Dict[str, Any], class_a: int, class_b: int) -> float:
        return float(self.pair_stats(record, class_a, class_b)["margin"])

    def pair_stats(self, record: Dict[str, Any], class_a: int, class_b: int) -> Dict[str, float]:
        import torch

        images = [record["image_a"], record["image_b"]]
        bboxes = [
            record.get("visible_bbox_a_xywhn") or record.get("bbox_xywhn"),
            record.get("visible_bbox_b_xywhn") or record.get("bbox_xywhn"),
        ]
        tensor = self._load_images(images)
        with torch.no_grad():
            raw = self.model(tensor)
            parsed = self.parser.parse_output(raw)
            responses = _responses_for_boxes_from_parsed(
                parsed,
                bboxes,
                self.stride,
                self.device,
                self.config.kd_topk,
            )
        class_count = int(responses.shape[1])
        if class_a >= class_count or class_b >= class_count:
            return {"margin": 0.0, "correct_score": 0.0, "confusion_score": 0.0, "presence_energy": 0.0}
        score_a_a = float(responses[0, class_a].item())
        score_a_b = float(responses[0, class_b].item())
        score_b_a = float(responses[1, class_a].item())
        score_b_b = float(responses[1, class_b].item())
        correct_score = 0.5 * (score_a_a + score_b_b)
        confusion_score = 0.5 * (score_a_b + score_b_a)
        presence_energy = 0.5 * float(torch.logsumexp(responses[:, :class_count], dim=1).sum().item())
        return {
            "margin": float(correct_score - confusion_score),
            "correct_score": float(correct_score),
            "confusion_score": float(confusion_score),
            "presence_energy": float(presence_energy),
        }

    def _load_images(self, paths: Sequence[str]) -> Any:
        import numpy as np
        import torch
        from PIL import Image, ImageOps

        tensors = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            image = ImageOps.exif_transpose(image)
            if image.size != (self.config.imgsz, self.config.imgsz):
                image = ImageOps.fit(
                    image,
                    (self.config.imgsz, self.config.imgsz),
                    method=Image.Resampling.BICUBIC,
                )
            arr = np.asarray(image, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
        return torch.stack(tensors, dim=0).to(self.device)


class NuisanceMiner:
    """Select GNF seeds that expose teacher-student counterfactual gaps."""

    def __init__(self, config: TwinSwapConfig, teacher_weights: str, student_weights: Path) -> None:
        from ultralytics import YOLO

        self.config = config
        self.teacher = YOLO(str(teacher_weights))
        self.teacher_scorer = _RawSlotResponseScorer(teacher_weights, config)
        self.student_scorer = _RawSlotResponseScorer(student_weights, config)
        if config.semantic_safety:
            from .gnf import GNFConfig, GNFGenerator

            self.gnf_generator = GNFGenerator(GNFConfig(width=config.imgsz, height=config.imgsz))
        else:
            self.gnf_generator = None

    def mine(self, manifest_path: Path) -> NuisanceMiningResult:
        records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        scores: Dict[str, List[float]] = {}
        nuisance_scores: List[Dict[str, Any]] = []
        triaged_nuisances: List[Dict[str, Any]] = []
        triage_counts: Dict[str, int] = {"identity_hard": 0, "recall_hard": 0, "empty_hard": 0, "ignored": 0}

        for record in records:
            class_a = int(record["class_a"])
            class_b = int(record["class_b"])
            key = f"{min(class_a, class_b)}:{max(class_a, class_b)}"
            teacher_stats = self.teacher_scorer.pair_stats(record, class_a, class_b)
            student_stats = self.student_scorer.pair_stats(record, class_a, class_b)
            teacher_margin = float(teacher_stats["margin"])
            student_margin = float(student_stats["margin"])
            if teacher_margin < self.config.teacher_margin_floor:
                triage_counts["ignored"] += 1
                continue
            regret = max(0.0, teacher_margin - student_margin)
            scores.setdefault(key, []).append(regret)

            background = dict((record.get("nuisance") or {}).get("background") or {})
            if background.get("type") != "generative_nuisance_field" or "seed" not in background:
                triage_counts["ignored"] += 1
                continue
            if regret <= 0.0:
                triage_counts["ignored"] += 1
                continue
            semantic_safe = (not self.config.semantic_safety) or self._background_is_semantically_safe(background)
            presence_gap = float(teacher_stats["presence_energy"]) - float(student_stats["presence_energy"])
            if not semantic_safe:
                bucket = "empty_hard"
            elif presence_gap > float(self.config.presence_collapse_margin) or float(student_stats["correct_score"]) < float(
                self.config.recall_hard_student_score_floor
            ):
                bucket = "recall_hard"
            else:
                bucket = "identity_hard"
            triage_counts[bucket] += 1
            triaged = {
                "seed": int(background["seed"]),
                "bucket": bucket,
                "regret": float(regret),
                "pair": key,
                "teacher_margin": float(teacher_margin),
                "student_margin": float(student_margin),
                "teacher_correct_score": float(teacher_stats["correct_score"]),
                "student_correct_score": float(student_stats["correct_score"]),
                "teacher_confusion_score": float(teacher_stats["confusion_score"]),
                "student_confusion_score": float(student_stats["confusion_score"]),
                "teacher_presence_energy": float(teacher_stats["presence_energy"]),
                "student_presence_energy": float(student_stats["presence_energy"]),
                "presence_energy_gap": float(presence_gap),
                "semantic_safe": bool(semantic_safe),
                "background": background,
            }
            triaged_nuisances.append(triaged)
            if bucket != "identity_hard":
                continue
            nuisance_scores.append(triaged)

        if not scores:
            return NuisanceMiningResult(pair_weights={}, hard_nuisances=[], triaged_nuisances=triaged_nuisances, triage_counts=triage_counts)

        regrets = {key: sum(values) / max(1, len(values)) for key, values in scores.items()}
        max_regret = max(regrets.values()) if regrets else 0.0
        if max_regret <= 1e-8:
            pair_weights = {key: 1.0 for key in regrets}
        else:
            temperature = max(1e-4, float(self.config.hard_pair_temperature))
            pair_weights = {
                key: float(self.config.hard_pair_min_weight + math.exp((value / max_regret) / temperature))
                for key, value in regrets.items()
            }

        nuisance_scores.sort(key=lambda item: item["regret"], reverse=True)
        hard_nuisances = nuisance_scores[: self.config.hard_nuisance_topk]
        triaged_nuisances.sort(key=lambda item: item["regret"], reverse=True)
        return NuisanceMiningResult(
            pair_weights=pair_weights,
            hard_nuisances=hard_nuisances,
            triaged_nuisances=triaged_nuisances,
            triage_counts=triage_counts,
        )

    def _background_is_semantically_safe(self, background: Dict[str, Any]) -> bool:
        if self.gnf_generator is None:
            return True
        import numpy as np

        seed = int(background["seed"])
        field = self.gnf_generator.generate(seed=seed, width=self.config.imgsz, height=self.config.imgsz)
        result = self.teacher.predict(
            source=np.asarray(field.image),
            imgsz=self.config.imgsz,
            device=self.config.device,
            verbose=False,
            conf=float(self.config.semantic_safety_conf),
        )[0]
        boxes = getattr(result, "boxes", None)
        return boxes is None or len(boxes) == 0

    def _pair_margin(self, model: Any, record: Dict[str, Any], class_a: int, class_b: int) -> float:
        result_a = model.predict(
            source=record["image_a"],
            imgsz=self.config.imgsz,
            device=self.config.device,
            verbose=False,
        )[0]
        result_b = model.predict(
            source=record["image_b"],
            imgsz=self.config.imgsz,
            device=self.config.device,
            verbose=False,
        )[0]
        expected_a = self._record_box_xyxy(record, result_a)
        expected_b = self._record_box_xyxy(record, result_b)
        score_a_a = self._class_score(result_a, class_a, expected_a)
        score_a_b = self._class_score(result_a, class_b, expected_a)
        score_b_a = self._class_score(result_b, class_a, expected_b)
        score_b_b = self._class_score(result_b, class_b, expected_b)
        return 0.5 * ((score_a_a - score_a_b) + (score_b_b - score_b_a))

    @staticmethod
    def _record_box_xyxy(record: Dict[str, Any], result: Any) -> Optional[Tuple[float, float, float, float]]:
        bbox = record.get("bbox_xywhn")
        if not bbox or len(bbox) != 4:
            return None
        shape = getattr(result, "orig_shape", None)
        if not shape or len(shape) < 2:
            return None
        height, width = float(shape[0]), float(shape[1])
        cx, cy, bw, bh = [float(value) for value in bbox]
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        return x1, y1, x2, y2

    @staticmethod
    def _class_score(
        result: Any,
        class_id: int,
        expected_xyxy: Optional[Tuple[float, float, float, float]] = None,
    ) -> float:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return 0.0
        best = 0.0
        for box in boxes:
            try:
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
            except Exception:
                continue
            if cls == int(class_id):
                if expected_xyxy is not None:
                    predicted_xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
                    if _box_iou(predicted_xyxy, expected_xyxy) < 0.05:
                        continue
                best = max(best, conf)
        return best


class TwinSwapTrainingRunner:
    """High-level round-based TwinSwap data generation, training, and hard-pair mining."""

    def __init__(self, config: TwinSwapConfig) -> None:
        self.config = config
        self._validate_config()
        self.builder = TwinSwapDatasetBuilder(config)

    def _validate_config(self) -> None:
        if self.config.pair_keep_batches and self.config.batch % 2 != 0:
            raise ValueError("TwinSwap pair-preserving training requires an even --batch value.")
        if self.config.rounds < 1:
            raise ValueError("rounds must be at least 1.")
        if self.config.epochs_per_round < 1:
            raise ValueError("epochs_per_round must be at least 1.")
        if self.config.teacher_epochs is not None and self.config.teacher_epochs < 1:
            raise ValueError("teacher_epochs must be at least 1.")
        if self.config.teacher_batch is not None and self.config.teacher_batch < 1:
            raise ValueError("teacher_batch must be at least 1.")
        if self.config.kd_weight > 0 and not (self.config.teacher_model or self.config.train_teacher):
            raise ValueError("kd_weight > 0 requires --teacher-model or --train-teacher.")
        if self.config.train_teacher and not (self.config.teacher_base_model or self.config.teacher_model):
            raise ValueError("--train-teacher requires --teacher-base-model or --teacher-model.")

    def run(self) -> Path:
        pair_weights = self._load_initial_pair_weights()
        current_model = self.config.student_model
        teacher_weights = self.config.teacher_model
        final_weights = Path(current_model)
        for round_index in range(self.config.rounds):
            round_root = self.config.output_dir / f"round_{round_index:02d}"
            data_yaml = self.builder.build(round_root, pair_weights)
            if round_index == 0 and self.config.train_teacher:
                teacher_weights = str(
                    self._train_teacher(
                        data_yaml=data_yaml,
                        model=self.config.teacher_base_model or self.config.teacher_model or "",
                    )
                )
            run_name = f"{self.config.name}_r{round_index:02d}"
            final_weights = self._train_round(data_yaml, current_model, run_name, teacher_weights)
            current_model = str(final_weights)
            if round_index + 1 < self.config.rounds and teacher_weights:
                pair_weights = self.mine_hard_pairs(
                    round_root / "twinswap_pairs_mine.jsonl",
                    final_weights,
                    teacher_weights=teacher_weights,
                )
                _write_json(self.config.output_dir / f"hard_pairs_round_{round_index:02d}.json", pair_weights)
                hard_nuisances = getattr(self, "_last_hard_nuisances", [])
                self.builder.hard_nuisances = hard_nuisances
                _write_jsonl(self.config.output_dir / f"hard_nuisances_round_{round_index:02d}.jsonl", hard_nuisances)
                triaged_nuisances = getattr(self, "_last_triaged_nuisances", [])
                if triaged_nuisances:
                    _write_jsonl(
                        self.config.output_dir / f"triaged_nuisances_round_{round_index:02d}.jsonl",
                        triaged_nuisances,
                    )
                    _write_json(
                        self.config.output_dir / f"triage_counts_round_{round_index:02d}.json",
                        getattr(self, "_last_triage_counts", {}),
                    )
        return final_weights

    def _load_initial_pair_weights(self) -> Optional[Dict[str, float]]:
        if self.config.hard_pair_weight_file and self.config.hard_pair_weight_file.exists():
            payload = _load_json(self.config.hard_pair_weight_file)
            return {str(key): float(value) for key, value in dict(payload).items()}
        return None

    def _base_train_overrides(self, data_yaml: Path, model: str, run_name: str, epochs: int) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {
            "model": model,
            "data": str(data_yaml),
            "epochs": epochs,
            "imgsz": self.config.imgsz,
            "batch": self.config.batch,
            "device": self.config.device,
            "workers": self.config.workers,
            "project": self.config.project,
            "name": run_name,
            "exist_ok": True,
            "patience": self.config.patience,
            "cache": self.config.cache,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "cutmix": 0.0,
            "copy_paste": 0.0,
            "auto_augment": None,
            "erasing": 0.0,
        }
        if self.config.lr0 is not None:
            overrides["lr0"] = self.config.lr0
        return overrides

    def _train_teacher(self, data_yaml: Path, model: str) -> Path:
        if not model:
            raise ValueError("Teacher training requires --teacher-base-model or --teacher-model.")
        overrides = self._base_train_overrides(
            data_yaml=data_yaml,
            model=model,
            run_name=self.config.teacher_name,
            epochs=int(self.config.teacher_epochs or self.config.epochs_per_round),
        )
        if self.config.teacher_batch is not None:
            overrides["batch"] = int(self.config.teacher_batch)
        overrides.update(
            {
                "twinswap_teacher": None,
                "twinswap_kd_weight": 0.0,
                "twinswap_keep_pairs": False,
            }
        )
        trainer = TwinSwapDetectionTrainer(overrides=overrides)
        trainer.train()
        return self._best_or_last(Path(str(trainer.save_dir)))

    def _train_round(
        self,
        data_yaml: Path,
        model: str,
        run_name: str,
        teacher_weights: Optional[str],
    ) -> Path:
        overrides = self._base_train_overrides(
            data_yaml=data_yaml,
            model=model,
            run_name=run_name,
            epochs=self.config.epochs_per_round,
        )
        overrides.update(
            {
                "twinswap_teacher": teacher_weights,
                "twinswap_kd_weight": self.config.kd_weight if teacher_weights else 0.0,
                "twinswap_temperature": self.config.kd_temperature,
                "twinswap_topk": self.config.kd_topk,
                "twinswap_margin_weight": self.config.kd_margin_weight,
                "twinswap_hard_pair_margin_weight": self.config.hard_pair_margin_weight,
                "twinswap_hard_pair_target_margin": self.config.hard_pair_target_margin,
                "twinswap_presence_weight": self.config.presence_invariance_weight,
                "twinswap_presence_alignment_weight": self.config.presence_alignment_weight,
                "twinswap_real_presence_weight": self.config.real_presence_weight,
                "twinswap_context_weight": self.config.context_invariance_weight,
                "twinswap_keep_pairs": self.config.pair_keep_batches,
            }
        )

        trainer = TwinSwapDetectionTrainer(overrides=overrides)
        trainer.train()
        return self._best_or_last(Path(str(trainer.save_dir)))

    @staticmethod
    def _best_or_last(save_dir: Path) -> Path:
        best = save_dir / "weights" / "best.pt"
        last = save_dir / "weights" / "last.pt"
        return best if best.exists() else last

    def mine_hard_pairs(
        self,
        manifest_path: Path,
        student_weights: Path,
        teacher_weights: Optional[str] = None,
    ) -> Dict[str, float]:
        """Run model-aware nuisance mining and return class-pair sampling weights."""
        teacher_path = teacher_weights or self.config.teacher_model
        if not teacher_path:
            self._last_hard_nuisances = []
            self._last_triaged_nuisances = []
            self._last_triage_counts = {}
            return {}
        result = NuisanceMiner(
            config=self.config,
            teacher_weights=str(teacher_path),
            student_weights=student_weights,
        ).mine(manifest_path)
        self._last_hard_nuisances = result.hard_nuisances
        self._last_triaged_nuisances = result.triaged_nuisances
        self._last_triage_counts = result.triage_counts
        return result.pair_weights


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("TwinSwap YOLO distillation trainer")
    parser.add_argument("--object-root", required=True, type=Path, help="Directory with one subdirectory per class.")
    parser.add_argument("--data-yaml", type=Path, default=None, help="Train on an existing generated YOLO data.yaml instead of building a round dataset.")
    parser.add_argument("--background-root", type=Path, default=None, help="Directory of background images for directory/mixed background modes.")
    parser.add_argument("--background-mode", choices=["gnf", "directory", "mixed"], default="gnf", help="Use procedural GNF fields, real background images, or a mix.")
    parser.add_argument("--gnf-real-background-prob", type=float, default=0.10, help="Probability of using a real background when --background-mode mixed.")
    parser.add_argument("--hard-nuisance-replay-prob", type=float, default=0.35, help="Probability of replaying mined hard GNF seeds after round 0.")
    parser.add_argument("--hard-nuisance-profile-prob", type=float, default=0.35, help="Probability of sampling new GNF seeds from mined hard style profiles after round 0.")
    parser.add_argument("--hard-nuisance-topk", type=int, default=512, help="Number of mined hard GNF seeds retained after each round.")
    parser.add_argument("--teacher-margin-floor", type=float, default=0.05, help="Reject nuisance candidates the teacher cannot separate confidently.")
    parser.add_argument("--recall-hard-student-score-floor", type=float, default=-1.0, help="Legacy PI mining gate; disabled by default for pure TwinSwap.")
    parser.add_argument("--presence-collapse-margin", type=float, default=1.0e9, help="Legacy PI mining gate; disabled by default for pure TwinSwap.")
    parser.add_argument("--semantic-safety-conf", type=float, default=0.35, help="Reject empty GNF fields where the teacher detects a target above this confidence.")
    parser.add_argument("--disable-semantic-safety", action="store_true", help="Disable teacher empty-background safety filtering for mined GNF seeds.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where generated TwinSwap datasets are written.")
    parser.add_argument("--student-model", required=True, help="YOLO student cfg or weights, e.g. yolo11n.pt or best.pt.")
    parser.add_argument("--teacher-model", default=None, help="Existing YOLO teacher weights, or base model when --train-teacher is set.")
    parser.add_argument("--teacher-base-model", default=None, help="Base largest YOLO model to train as teacher, e.g. yolo26x.pt or yolo11x.pt.")
    parser.add_argument("--train-teacher", action="store_true", help="Train the teacher first, then distill the student from its best.pt.")
    parser.add_argument("--teacher-epochs", type=int, default=None, help="Teacher training epochs. Defaults to --epochs-per-round.")
    parser.add_argument("--teacher-batch", type=int, default=None, help="Optional smaller batch for teacher training to avoid OOM.")
    parser.add_argument("--teacher-name", default="twinswap_teacher", help="Run name for teacher training.")
    parser.add_argument("--classes", type=Path, default=None, help="Optional classes.txt. Defaults to object-root folders.")
    parser.add_argument("--train-pairs", type=int, default=1200)
    parser.add_argument("--val-pairs", type=int, default=160)
    parser.add_argument("--mine-pairs", type=int, default=240)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--epochs-per-round", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--project", default="runs_twinswap")
    parser.add_argument("--name", default="twinswap")
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--kd-weight", type=float, default=0.35)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--kd-topk", type=int, default=24)
    parser.add_argument("--kd-margin-weight", type=float, default=0.75)
    parser.add_argument("--hard-pair-margin-weight", type=float, default=0.75, help="Supervised hinge loss weight for the true class vs its hard-pair counterpart.")
    parser.add_argument("--hard-pair-target-margin", type=float, default=0.50, help="Target raw-logit margin for hard-pair identity commitment.")
    parser.add_argument("--presence-invariance-weight", type=float, default=0.0, help="Legacy PI ablation term; disabled for pure TwinSwap.")
    parser.add_argument("--presence-alignment-weight", type=float, default=0.0, help="Legacy PI ablation term; disabled for pure TwinSwap.")
    parser.add_argument("--real-presence-weight", type=float, default=0.0, help="Legacy PI ablation term; disabled for pure TwinSwap.")
    parser.add_argument("--context-invariance-weight", type=float, default=0.0, help="Legacy PI ablation term; disabled for pure TwinSwap.")
    parser.add_argument("--identity-kd-all-pairs", action="store_true", help="Apply identity KD to every TwinSwap pair, matching the older TwinSwap behavior.")
    parser.add_argument("--identity-hard-pair-ratio", type=float, default=0.75, help="Fraction of generated TwinSwap pairs drawn from canonical hard identity pairs.")
    parser.add_argument("--cross-form-pair-ratio", type=float, default=0.15, help="Fraction of generated TwinSwap pairs reserved for cross-form recall calibration.")
    parser.add_argument("--hard-pair-weight-file", type=Path, default=None)
    parser.add_argument("--hard-pair-temperature", type=float, default=0.35)
    parser.add_argument("--hard-pair-min-weight", type=float, default=0.20)
    parser.add_argument("--min-scale", type=float, default=0.18)
    parser.add_argument("--max-scale", type=float, default=0.42)
    parser.add_argument("--single-object-ratio", type=float, default=0.30)
    parser.add_argument("--multi-object-ratio", type=float, default=0.50)
    parser.add_argument("--dense-scene-ratio", type=float, default=0.20)
    parser.add_argument("--min-context-objects", type=int, default=2)
    parser.add_argument("--max-context-objects", type=int, default=5)
    parser.add_argument("--min-dense-objects", type=int, default=4)
    parser.add_argument("--max-dense-objects", type=int, default=8)
    parser.add_argument("--min-distractor-scale", type=float, default=0.08)
    parser.add_argument("--max-distractor-scale", type=float, default=0.28)
    parser.add_argument("--real-anchor-video-root", type=Path, default=None, help="Original video directory for real positive presence anchors.")
    parser.add_argument("--real-anchor-mask-root", type=Path, default=None, help="Mask directory for real positive presence anchors.")
    parser.add_argument("--real-anchor-train-per-class", type=int, default=0)
    parser.add_argument("--real-anchor-val-per-class", type=int, default=0)
    parser.add_argument("--real-anchor-train-repeat", type=int, default=1)
    parser.add_argument("--real-anchor-include-mesh", action="store_true")
    parser.add_argument("--real-anchor-min-area-ratio", type=float, default=0.0005)
    parser.add_argument("--real-anchor-max-area-ratio", type=float, default=0.80)
    parser.add_argument("--pseudo-real-source-root", type=Path, default=None, help="Optional existing YOLO dataset containing zzz_real_pseudo_* frames to copy into train/val.")
    parser.add_argument("--pseudo-real-prefix", default="zzz_real_pseudo_", help="Filename prefix for pseudo-real imported frames; use '*' to import every image from the YOLO source split.")
    parser.add_argument("--pseudo-real-train-limit", type=int, default=0, help="Max pseudo-real train frames to import; 0 means all.")
    parser.add_argument("--pseudo-real-val-limit", type=int, default=0, help="Max pseudo-real val frames to import; 0 means all.")
    parser.add_argument("--max-slot-iou", type=float, default=0.15)
    parser.add_argument("--swap-visible-area-tolerance", type=float, default=0.20)
    parser.add_argument("--swap-match-attempts", type=int, default=8)
    parser.add_argument("--min-visible-area-ratio", type=float, default=0.08)
    parser.add_argument("--disable-kg-scale-prior", action="store_true", help="Disable weak class-level relative scale priors in context/dense scenes.")
    parser.add_argument("--kg-scale-prior-strength", type=float, default=1.0, help="Blend strength for weak KG scale priors in non-swap placements.")
    parser.add_argument("--kg-scale-prior-jitter", type=float, default=0.25, help="Relative jitter around weak KG scale priors.")
    parser.add_argument("--blur-prob", type=float, default=0.35)
    parser.add_argument("--occlusion-prob", type=float, default=0.20)
    parser.add_argument("--preserve-aspect", action="store_true", default=True)
    parser.add_argument("--stretch-to-slot", action="store_false", dest="preserve_aspect")
    parser.add_argument("--shuffle-pairs", action="store_true", help="Allow normal shuffled YOLO batches; KD only applies when both twins land in one batch.")
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--build-dataset-only", action="store_true", help="Generate round_00 YOLO data and exit without training.")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TwinSwapConfig:
    classes = _read_classes(args.classes, args.object_root)
    return TwinSwapConfig(
        object_root=args.object_root,
        background_root=args.background_root,
        output_dir=args.output_dir,
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        teacher_base_model=args.teacher_base_model,
        train_teacher=args.train_teacher,
        teacher_epochs=args.teacher_epochs,
        teacher_batch=args.teacher_batch,
        teacher_name=args.teacher_name,
        classes=classes,
        train_pairs=args.train_pairs,
        val_pairs=args.val_pairs,
        mine_pairs=args.mine_pairs,
        rounds=args.rounds,
        epochs_per_round=args.epochs_per_round,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=args.project,
        name=args.name,
        lr0=args.lr0,
        patience=args.patience,
        kd_weight=args.kd_weight,
        kd_temperature=args.kd_temperature,
        kd_topk=args.kd_topk,
        kd_margin_weight=args.kd_margin_weight,
        hard_pair_margin_weight=args.hard_pair_margin_weight,
        hard_pair_target_margin=args.hard_pair_target_margin,
        presence_invariance_weight=args.presence_invariance_weight,
        presence_alignment_weight=args.presence_alignment_weight,
        real_presence_weight=args.real_presence_weight,
        context_invariance_weight=args.context_invariance_weight,
        identity_kd_hard_only=not args.identity_kd_all_pairs,
        identity_hard_pair_ratio=args.identity_hard_pair_ratio,
        cross_form_pair_ratio=args.cross_form_pair_ratio,
        pair_keep_batches=not args.shuffle_pairs,
        hard_pair_temperature=args.hard_pair_temperature,
        hard_pair_min_weight=args.hard_pair_min_weight,
        hard_pair_weight_file=args.hard_pair_weight_file,
        background_mode=args.background_mode,
        gnf_real_background_prob=args.gnf_real_background_prob,
        hard_nuisance_replay_prob=args.hard_nuisance_replay_prob,
        hard_nuisance_profile_prob=args.hard_nuisance_profile_prob,
        hard_nuisance_topk=args.hard_nuisance_topk,
        teacher_margin_floor=args.teacher_margin_floor,
        recall_hard_student_score_floor=args.recall_hard_student_score_floor,
        presence_collapse_margin=args.presence_collapse_margin,
        semantic_safety=not args.disable_semantic_safety,
        semantic_safety_conf=args.semantic_safety_conf,
        single_object_ratio=args.single_object_ratio,
        multi_object_ratio=args.multi_object_ratio,
        dense_scene_ratio=args.dense_scene_ratio,
        context_object_count_range=(args.min_context_objects, args.max_context_objects),
        dense_object_count_range=(args.min_dense_objects, args.max_dense_objects),
        target_scale_range=(args.min_scale, args.max_scale),
        distractor_scale_range=(args.min_distractor_scale, args.max_distractor_scale),
        real_anchor_video_root=args.real_anchor_video_root,
        real_anchor_mask_root=args.real_anchor_mask_root,
        real_anchor_train_per_class=args.real_anchor_train_per_class,
        real_anchor_val_per_class=args.real_anchor_val_per_class,
        real_anchor_train_repeat=args.real_anchor_train_repeat,
        real_anchor_include_mesh=args.real_anchor_include_mesh,
        real_anchor_min_area_ratio=args.real_anchor_min_area_ratio,
        real_anchor_max_area_ratio=args.real_anchor_max_area_ratio,
        pseudo_real_source_root=args.pseudo_real_source_root,
        pseudo_real_filename_prefix=args.pseudo_real_prefix,
        pseudo_real_train_limit=args.pseudo_real_train_limit,
        pseudo_real_val_limit=args.pseudo_real_val_limit,
        max_slot_iou=args.max_slot_iou,
        swap_visible_area_tolerance=args.swap_visible_area_tolerance,
        swap_match_attempts=args.swap_match_attempts,
        min_visible_area_ratio=args.min_visible_area_ratio,
        use_kg_scale_prior=not args.disable_kg_scale_prior,
        kg_scale_prior_strength=args.kg_scale_prior_strength,
        kg_scale_prior_jitter=args.kg_scale_prior_jitter,
        blur_prob=args.blur_prob,
        occlusion_prob=args.occlusion_prob,
        preserve_aspect=args.preserve_aspect,
        clear_output=args.clear_output,
        cache=args.cache,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    config = config_from_args(args)
    if args.build_dataset_only:
        data_yaml = TwinSwapDatasetBuilder(config).build(config.output_dir / "round_00", pair_weights=None)
        print(f"TwinSwap dataset ready: {data_yaml}")
        return
    if args.data_yaml is not None:
        runner = TwinSwapTrainingRunner(config)
        teacher_weights = config.teacher_model
        if config.train_teacher:
            teacher_weights = str(
                runner._train_teacher(
                    data_yaml=args.data_yaml,
                    model=config.teacher_base_model or config.teacher_model or "",
                )
            )
        if not teacher_weights:
            raise ValueError("--data-yaml training requires --teacher-model or --train-teacher.")
        final_weights = runner._train_round(args.data_yaml, config.student_model, config.name, teacher_weights)
        print(f"TwinSwap training complete: teacher={teacher_weights} student={final_weights}")
        return
    final_weights = TwinSwapTrainingRunner(config).run()
    print(f"TwinSwap training complete: {final_weights}")


if __name__ == "__main__":
    main()
