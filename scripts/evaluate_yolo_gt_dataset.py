"""Evaluate detector weights on a standard YOLO image/label dataset.

This is used for INSPECT object-evidence tests after manually correcting
YOLO-format boxes. It reports object-level recall, class accuracy on matched
boxes, hard-pair errors, false positives, and detection density.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2


HARD_COUNTERPART = {
    "type_2_gear": "type_8_gear",
    "type_8_gear": "type_2_gear",
    "type_3_gear": "type_7_gear",
    "type_7_gear": "type_3_gear",
    "type_5_gearbox_cover": "type_6_gearbox_cover",
    "type_6_gearbox_cover": "type_5_gearbox_cover",
    "type_5_gearbox_housing": "type_6_gearbox_housing",
    "type_6_gearbox_housing": "type_5_gearbox_housing",
}


@dataclass
class Box:
    cls_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0


def read_classes(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_yolo_label(path: Path, width: int, height: int) -> List[Box]:
    boxes: List[Box] = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, bw, bh = [float(x) for x in parts[1:5]]
        boxes.append(
            Box(
                cls_id=cls_id,
                x1=(cx - bw / 2.0) * width,
                y1=(cy - bh / 2.0) * height,
                x2=(cx + bw / 2.0) * width,
                y2=(cy + bh / 2.0) * height,
            )
        )
    return boxes


def box_iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    return float(inter / (area_a + area_b - inter + 1e-9))


def load_model(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


def model_names(model: Any) -> Dict[int, str]:
    names = getattr(model, "names", {}) or {}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(name) for i, name in enumerate(names)}


def predict(
    *,
    model: Any,
    image_path: Path,
    class_names: Sequence[str],
    conf: float,
    iou: float,
    device: str,
    end2end: Optional[bool],
) -> List[Box]:
    kwargs: Dict[str, Any] = {
        "source": str(image_path),
        "conf": float(conf),
        "iou": float(iou),
        "device": device,
        "verbose": False,
    }
    if end2end is not None:
        kwargs["end2end"] = bool(end2end)
    try:
        results = model.predict(**kwargs)
    except TypeError:
        kwargs.pop("end2end", None)
        results = model.predict(**kwargs)
    if not results:
        return []
    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    names = getattr(result, "names", None) or model_names(model)
    if not isinstance(names, dict):
        names = {i: str(name) for i, name in enumerate(names)}
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    out: List[Box] = []
    for item in boxes:
        raw_id = int(item.cls[0].item()) if getattr(item, "cls", None) is not None else -1
        raw_name = str(names.get(raw_id, raw_id))
        cls_id = class_to_id.get(raw_name, raw_id if 0 <= raw_id < len(class_names) else -1)
        if cls_id < 0 or cls_id >= len(class_names):
            continue
        x1, y1, x2, y2 = [float(v) for v in item.xyxy[0].tolist()]
        score = float(item.conf[0].item()) if getattr(item, "conf", None) is not None else 0.0
        out.append(Box(cls_id, x1, y1, x2, y2, score))
    return out


def boxes_from_result(result: Any, model: Any, class_names: Sequence[str]) -> List[Box]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    names = getattr(result, "names", None) or model_names(model)
    if not isinstance(names, dict):
        names = {i: str(name) for i, name in enumerate(names)}
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    out: List[Box] = []
    for item in boxes:
        raw_id = int(item.cls[0].item()) if getattr(item, "cls", None) is not None else -1
        raw_name = str(names.get(raw_id, raw_id))
        cls_id = class_to_id.get(raw_name, raw_id if 0 <= raw_id < len(class_names) else -1)
        if cls_id < 0 or cls_id >= len(class_names):
            continue
        x1, y1, x2, y2 = [float(v) for v in item.xyxy[0].tolist()]
        score = float(item.conf[0].item()) if getattr(item, "conf", None) is not None else 0.0
        out.append(Box(cls_id, x1, y1, x2, y2, score))
    return out


def predict_many(
    *,
    model: Any,
    image_paths: Sequence[Path],
    class_names: Sequence[str],
    conf: float,
    iou: float,
    device: str,
    end2end: Optional[bool],
    batch: int,
) -> List[List[Box]]:
    kwargs: Dict[str, Any] = {
        "source": [str(path) for path in image_paths],
        "conf": float(conf),
        "iou": float(iou),
        "device": device,
        "batch": int(batch),
        "stream": True,
        "verbose": False,
    }
    if end2end is not None:
        kwargs["end2end"] = bool(end2end)
    try:
        results = model.predict(**kwargs)
    except TypeError:
        kwargs.pop("end2end", None)
        results = model.predict(**kwargs)
    return [boxes_from_result(result, model, class_names) for result in results]


def greedy_match(gt: Sequence[Box], pred: Sequence[Box], iou_threshold: float) -> List[Tuple[int, int, float]]:
    pairs: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            iou = box_iou(g, p)
            if iou >= iou_threshold:
                pairs.append((iou, gi, pi))
    pairs.sort(reverse=True, key=lambda item: item[0])
    used_g: set[int] = set()
    used_p: set[int] = set()
    matches: List[Tuple[int, int, float]] = []
    for iou, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi, iou))
    return matches


def empty_stats(class_names: Sequence[str]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "images": 0,
        "gt": 0,
        "pred": 0,
        "matched": 0,
        "correct": 0,
        "hard_error": 0,
        "other_class_error": 0,
        "miss": 0,
        "fp": 0,
        "exact_image_success": 0,
        "images_with_gt": 0,
        "top1_image_correct": 0,
        "sum_identity_margin": 0.0,
        "identity_margin_count": 0,
    }
    for name in class_names:
        stats[f"gt::{name}"] = 0
        stats[f"matched::{name}"] = 0
        stats[f"correct::{name}"] = 0
        stats[f"hard::{name}"] = 0
        stats[f"miss::{name}"] = 0
    return stats


def add_image_stats(
    *,
    stats: Dict[str, Any],
    gt: Sequence[Box],
    pred: Sequence[Box],
    matches: Sequence[Tuple[int, int, float]],
    class_names: Sequence[str],
) -> None:
    stats["images"] += 1
    stats["gt"] += len(gt)
    stats["pred"] += len(pred)
    stats["matched"] += len(matches)
    matched_g = {gi for gi, _, _ in matches}
    matched_p = {pi for _, pi, _ in matches}
    if gt:
        stats["images_with_gt"] += 1
    correct_this = 0
    for g in gt:
        stats[f"gt::{class_names[g.cls_id]}"] += 1
    for gi, pi, _ in matches:
        g = gt[gi]
        p = pred[pi]
        gt_name = class_names[g.cls_id]
        pred_name = class_names[p.cls_id]
        stats[f"matched::{gt_name}"] += 1
        if p.cls_id == g.cls_id:
            stats["correct"] += 1
            stats[f"correct::{gt_name}"] += 1
            correct_this += 1
        elif HARD_COUNTERPART.get(gt_name) == pred_name:
            stats["hard_error"] += 1
            stats[f"hard::{gt_name}"] += 1
        else:
            stats["other_class_error"] += 1
    for gi, g in enumerate(gt):
        if gi not in matched_g:
            stats["miss"] += 1
            stats[f"miss::{class_names[g.cls_id]}"] += 1
    stats["fp"] += max(0, len(pred) - len(matched_p))
    if gt and correct_this == len(gt) and len(pred) == len(gt):
        stats["exact_image_success"] += 1
    if gt and pred:
        top = max(pred, key=lambda box: box.conf)
        gt_classes = {g.cls_id for g in gt}
        if top.cls_id in gt_classes:
            stats["top1_image_correct"] += 1

        for g in gt:
            gt_name = class_names[g.cls_id]
            counterpart = HARD_COUNTERPART.get(gt_name)
            if counterpart is None:
                continue
            counterpart_id = class_names.index(counterpart) if counterpart in class_names else -1
            gt_conf = max((p.conf for p in pred if p.cls_id == g.cls_id), default=0.0)
            hard_conf = max((p.conf for p in pred if p.cls_id == counterpart_id), default=0.0)
            stats["sum_identity_margin"] += gt_conf - hard_conf
            stats["identity_margin_count"] += 1


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def summarize(stats: Mapping[str, Any], *, model_name: str, weights: str, conf: float, iou_threshold: float) -> Dict[str, Any]:
    gt = float(stats["gt"])
    matched = float(stats["matched"])
    pred = float(stats["pred"])
    images = float(stats["images"])
    correct_recall = safe_div(stats["correct"], gt)
    correct_precision = safe_div(stats["correct"], pred)
    correct_f1 = safe_div(2.0 * correct_recall * correct_precision, correct_recall + correct_precision)
    return {
        "model": model_name,
        "weights": weights,
        "conf": conf,
        "iou_match": iou_threshold,
        "images": int(stats["images"]),
        "gt_boxes": int(stats["gt"]),
        "pred_boxes": int(stats["pred"]),
        "box_oracle_recall": safe_div(stats["matched"], gt),
        "correct_class_recall": correct_recall,
        "matched_class_accuracy": safe_div(stats["correct"], matched),
        "hard_pair_error_per_gt": safe_div(stats["hard_error"], gt),
        "hard_pair_error_per_matched": safe_div(stats["hard_error"], matched),
        "other_class_error_per_gt": safe_div(stats["other_class_error"], gt),
        "miss_rate": safe_div(stats["miss"], gt),
        "false_positive_per_frame": safe_div(stats["fp"], images),
        "avg_detections_per_frame": safe_div(pred, images),
        "precision_correct_class": correct_precision,
        "correct_evidence_f1": correct_f1,
        "net_correct_recall_minus_hard_pair_error": correct_recall - safe_div(stats["hard_error"], gt),
        "exact_image_success": safe_div(stats["exact_image_success"], stats["images_with_gt"]),
        "top1_image_accuracy": safe_div(stats["top1_image_correct"], stats["images_with_gt"]),
        "avg_identity_margin": safe_div(stats["sum_identity_margin"], stats["identity_margin_count"]),
    }


def parse_weight_items(values: Optional[Sequence[str]]) -> Dict[str, str]:
    if not values:
        raise ValueError("At least one --weights name=path entry is required.")
    out: Dict[str, str] = {}
    for item in values:
        if "=" in item:
            name, path = item.split("=", 1)
            out[name.strip()] = path.strip()
        else:
            path = Path(item)
            out[path.parent.parent.name if path.name == "best.pt" else path.stem] = str(path)
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        default="test",
        help="Image/label split below dataset-root (for example val or test).",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=None,
        help="Optional class-name file; defaults to dataset-root/classes.txt.",
    )
    parser.add_argument("--weights", nargs="+", required=True, help="One or more name=path checkpoint entries.")
    parser.add_argument("--confs", default="0.25,0.10,0.05")
    parser.add_argument("--device", default="0")
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--end2end-mode", choices=("default", "end2end", "one2many"), default="one2many")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help=(
            "Optional deterministic calibration subset size. Images are "
            "selected by SHA-1 filename order without reading labels."
        ),
    )
    parser.add_argument(
        "--single-pass-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run each model once at the minimum threshold and filter predictions for higher thresholds.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/detector_evaluation"))
    args = parser.parse_args()

    classes_path = args.classes or (args.dataset_root / "classes.txt")
    class_names = read_classes(classes_path)
    split = str(args.split).strip()
    images = sorted((args.dataset_root / "images" / split).glob("*"))
    images = [path for path in images if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if int(args.max_images) > 0 and len(images) > int(args.max_images):
        images = sorted(
            images,
            key=lambda path: hashlib.sha1(path.name.encode("utf-8")).hexdigest(),
        )[: int(args.max_images)]
        images = sorted(images)
    if not images:
        raise RuntimeError(f"No images found under {args.dataset_root / 'images' / split}")
    weights = parse_weight_items(args.weights)
    confs = [float(item.strip()) for item in args.confs.split(",") if item.strip()]
    end2end: Optional[bool]
    if args.end2end_mode == "end2end":
        end2end = True
    elif args.end2end_mode == "one2many":
        end2end = False
    else:
        end2end = None

    image_records: List[Tuple[Path, int, int, List[Box]]] = []
    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gt = read_yolo_label(args.dataset_root / "labels" / split / f"{image_path.stem}.txt", w, h)
        image_records.append((image_path, w, h, gt))
    print(f"[dataset] images={len(image_records)}")

    summary_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_index, (model_name, weights_path) in enumerate(weights.items(), start=1):
        print(f"[model {model_index}/{len(weights)}] {model_name}: {weights_path}")
        model = load_model(weights_path)
        base_preds_by_image: Optional[List[List[Box]]] = None
        if args.single_pass_sweep:
            min_conf = min(confs)
            print(f"[single-pass] predict once at conf={min_conf}")
            base_preds_by_image = predict_many(
                model=model,
                image_paths=[record[0] for record in image_records],
                class_names=class_names,
                conf=min_conf,
                iou=float(args.nms_iou),
                device=str(args.device),
                end2end=end2end,
                batch=int(args.batch),
            )
        for conf in confs:
            stats = empty_stats(class_names)
            if base_preds_by_image is None:
                preds_by_image = predict_many(
                    model=model,
                    image_paths=[record[0] for record in image_records],
                    class_names=class_names,
                    conf=conf,
                    iou=float(args.nms_iou),
                    device=str(args.device),
                    end2end=end2end,
                    batch=int(args.batch),
                )
            else:
                preds_by_image = [[box for box in boxes if float(box.conf) >= conf] for boxes in base_preds_by_image]
            for (image_path, _w, _h, gt), pred in zip(image_records, preds_by_image):
                matches = greedy_match(gt, pred, float(args.match_iou))
                add_image_stats(stats=stats, gt=gt, pred=pred, matches=matches, class_names=class_names)
                if conf == confs[-1]:
                    detail_rows.append(
                        {
                            "model": model_name,
                            "image": image_path.name,
                            "gt": len(gt),
                            "pred": len(pred),
                            "matched": len(matches),
                            "correct": sum(1 for gi, pi, _ in matches if gt[gi].cls_id == pred[pi].cls_id),
                        }
                    )
            summary_rows.append(
                summarize(
                    stats,
                    model_name=model_name,
                    weights=weights_path,
                    conf=conf,
                    iou_threshold=float(args.match_iou),
                )
            )
            for cls_id, cls_name in enumerate(class_names):
                per_class_rows.append(
                    {
                        "model": model_name,
                        "conf": conf,
                        "class": cls_name,
                        "gt": int(stats[f"gt::{cls_name}"]),
                        "box_recall": safe_div(stats[f"matched::{cls_name}"], stats[f"gt::{cls_name}"]),
                        "correct_recall": safe_div(stats[f"correct::{cls_name}"], stats[f"gt::{cls_name}"]),
                        "matched_acc": safe_div(stats[f"correct::{cls_name}"], stats[f"matched::{cls_name}"]),
                        "hard_pair_error_per_gt": safe_div(stats[f"hard::{cls_name}"], stats[f"gt::{cls_name}"]),
                        "miss_rate": safe_div(stats[f"miss::{cls_name}"], stats[f"gt::{cls_name}"]),
                    }
                )
            print(json.dumps(summary_rows[-1], indent=2))
            write_csv(args.output_dir / "summary.csv", summary_rows)
            write_csv(args.output_dir / "per_class.csv", per_class_rows)
            write_csv(args.output_dir / "image_details_at_lowest_conf.csv", detail_rows)
            (args.output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
