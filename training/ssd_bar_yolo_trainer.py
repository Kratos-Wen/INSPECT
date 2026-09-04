"""YOLO26s SSD/BAR baseline trainer for Gear8.

This script implements a compact reproduction-oriented baseline inspired by
Snap, Segment, Deploy with Background-Agnostic Refinement (BAR):

1. Train YOLO26s on the provided Gear8 YOLO dataset.
2. Use the first-stage model to generate background-agnostic refinement images:
   detected object crops are pasted at the same normalized locations on a
   uniform canvas.
3. Fine-tune YOLO26s on the BAR refinement dataset.

It is intentionally separate from the TwinSwap trainer so the comparison
baseline remains clean and auditable.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DatasetSpec:
    data_root: Path
    yaml_path: Path
    class_names: dict[int, str]


def _load_names(source_yaml: Path) -> dict[int, str]:
    data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def _iter_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def prepare_local_yaml(data_root: Path) -> DatasetSpec:
    source_yaml = data_root / "gear8.yaml"
    if not source_yaml.exists():
        raise FileNotFoundError(f"Missing Gear8 yaml: {source_yaml}")

    for rel in ["images/train", "images/val", "labels/train", "labels/val"]:
        path = data_root / rel
        if not path.exists():
            raise FileNotFoundError(
                f"Gear8 is not trainable yet: missing {path}. "
                "Expected a standard YOLO dataset with images/train and images/val."
            )

    train_images = _iter_images(data_root / "images" / "train")
    val_images = _iter_images(data_root / "images" / "val")
    if not train_images or not val_images:
        raise FileNotFoundError(
            f"Gear8 image splits are empty. Found {len(train_images)} train and {len(val_images)} val images."
        )

    names = _load_names(source_yaml)
    local_yaml = data_root / "gear8_local.yaml"
    payload = {
        "path": str(data_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    local_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return DatasetSpec(data_root=data_root, yaml_path=local_yaml, class_names=names)


def train_yolo(
    model_path: str | Path,
    data_yaml: Path,
    project: Path,
    name: str,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str,
    workers: int,
    stop_map50: float = 0.0,
    stop_min_epochs: int = 1,
) -> Path:
    model = YOLO(str(model_path))
    if stop_map50 > 0:
        target = float(stop_map50)
        min_epochs = max(1, int(stop_min_epochs))

        def _stop_when_ready(trainer) -> None:
            metrics = getattr(trainer, "metrics", {}) or {}
            value = metrics.get("metrics/mAP50(B)")
            if value is None:
                value = metrics.get("mAP50")
            try:
                map50 = float(value)
            except (TypeError, ValueError):
                return
            epoch = int(getattr(trainer, "epoch", -1)) + 1
            if epoch >= min_epochs and map50 >= target:
                print(
                    f"[BAR] Stage1 reached mAP50={map50:.4f} at epoch {epoch}; "
                    "stopping stage1 and switching to BAR stage2.",
                    flush=True,
                )
                setattr(trainer, "stop", True)

        try:
            model.add_callback("on_fit_epoch_end", _stop_when_ready)
        except Exception as exc:
            print(f"[BAR] Warning: could not register early stage switch callback: {exc}", flush=True)

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        project=str(project),
        name=name,
        exist_ok=True,
        cache=False,
        amp=True,
    )
    best = project / name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training finished but best.pt was not found: {best}")
    return best


def _read_best_metric(results_csv: Path, column: str) -> float:
    if not results_csv.exists():
        return 0.0
    best = 0.0
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get(column)
            if value is None:
                # Ultralytics sometimes preserves leading spaces in CSV headers.
                for key, candidate in row.items():
                    if key.strip() == column:
                        value = candidate
                        break
            try:
                best = max(best, float(str(value).strip()))
            except (TypeError, ValueError):
                continue
    return best


def _stage1_is_ready(project: Path, name: str, target_map50: float) -> bool:
    if target_map50 <= 0:
        return False
    run_dir = project / name
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        return False
    best_map50 = _read_best_metric(run_dir / "results.csv", "metrics/mAP50(B)")
    if best_map50 >= float(target_map50):
        print(
            f"[BAR] Existing stage1 run already reached mAP50={best_map50:.4f}; "
            "reusing best.pt and entering BAR stage2.",
            flush=True,
        )
        return True
    return False


def _read_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    labels: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        x, y, w, h = (float(v) for v in parts[1:5])
        labels.append((cls, x, y, w, h))
    return labels


def _paste_crop(
    source: np.ndarray,
    canvas: np.ndarray,
    xyxy: tuple[int, int, int, int],
    target_xyxy: tuple[int, int, int, int],
) -> bool:
    x1, y1, x2, y2 = xyxy
    tx1, ty1, tx2, ty2 = target_xyxy
    if x2 <= x1 or y2 <= y1 or tx2 <= tx1 or ty2 <= ty1:
        return False
    crop = source[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    resized = cv2.resize(crop, (tx2 - tx1, ty2 - ty1), interpolation=cv2.INTER_AREA)
    canvas[ty1:ty2, tx1:tx2] = resized
    return True


def build_bar_dataset(
    spec: DatasetSpec,
    stage1_weights: Path,
    bar_root: Path,
    imgsz: int,
    conf: float,
    device: str,
) -> Path:
    if bar_root.exists():
        shutil.rmtree(bar_root)

    names = spec.class_names
    model = YOLO(str(stage1_weights))
    for split in ["train", "val"]:
        src_img_dir = spec.data_root / "images" / split
        src_label_dir = spec.data_root / "labels" / split
        dst_img_dir = bar_root / "images" / split
        dst_label_dir = bar_root / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        for image_path in _iter_images(src_img_dir):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            h0, w0 = image.shape[:2]
            canvas = np.full((imgsz, imgsz, 3), 255, dtype=np.uint8)
            out_labels: list[str] = []

            result = model.predict(
                source=str(image_path),
                imgsz=imgsz,
                conf=conf,
                device=device,
                verbose=False,
            )[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy()
                for box, cls_float in zip(boxes, clss):
                    x1, y1, x2, y2 = box.tolist()
                    cls = int(cls_float)
                    sx1 = int(max(0, min(w0 - 1, round(x1))))
                    sy1 = int(max(0, min(h0 - 1, round(y1))))
                    sx2 = int(max(0, min(w0, round(x2))))
                    sy2 = int(max(0, min(h0, round(y2))))
                    nx1 = int(round(sx1 / max(1, w0) * imgsz))
                    ny1 = int(round(sy1 / max(1, h0) * imgsz))
                    nx2 = int(round(sx2 / max(1, w0) * imgsz))
                    ny2 = int(round(sy2 / max(1, h0) * imgsz))
                    nx1, ny1 = max(0, nx1), max(0, ny1)
                    nx2, ny2 = min(imgsz, nx2), min(imgsz, ny2)
                    if _paste_crop(image, canvas, (sx1, sy1, sx2, sy2), (nx1, ny1, nx2, ny2)):
                        cx = ((nx1 + nx2) / 2.0) / imgsz
                        cy = ((ny1 + ny2) / 2.0) / imgsz
                        bw = (nx2 - nx1) / imgsz
                        bh = (ny2 - ny1) / imgsz
                        out_labels.append(f"{cls} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")

            # Keep the refinement set complete. If the first-stage detector
            # misses all objects, fall back to the original labels and crops.
            if not out_labels:
                for cls, cx, cy, bw, bh in _read_label_file(src_label_dir / f"{image_path.stem}.txt"):
                    sx1 = int(max(0, (cx - bw / 2) * w0))
                    sy1 = int(max(0, (cy - bh / 2) * h0))
                    sx2 = int(min(w0, (cx + bw / 2) * w0))
                    sy2 = int(min(h0, (cy + bh / 2) * h0))
                    nx1 = int(max(0, (cx - bw / 2) * imgsz))
                    ny1 = int(max(0, (cy - bh / 2) * imgsz))
                    nx2 = int(min(imgsz, (cx + bw / 2) * imgsz))
                    ny2 = int(min(imgsz, (cy + bh / 2) * imgsz))
                    if _paste_crop(image, canvas, (sx1, sy1, sx2, sy2), (nx1, ny1, nx2, ny2)):
                        out_labels.append(f"{cls} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")

            out_image = dst_img_dir / f"{image_path.stem}.jpg"
            out_label = dst_label_dir / f"{image_path.stem}.txt"
            cv2.imwrite(str(out_image), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            out_label.write_text("\n".join(out_labels) + ("\n" if out_labels else ""), encoding="utf-8")

    data_yaml = bar_root / "data.yaml"
    payload = {
        "path": str(bar_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    data_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return data_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26s SSD/BAR baseline on Gear8.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--bar-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--stage1-epochs", type=int, default=80)
    parser.add_argument("--stage2-epochs", type=int, default=40)
    parser.add_argument(
        "--stage1-target-map50",
        type=float,
        default=0.95,
        help="Stop stage1 and switch to BAR stage2 once validation mAP50 reaches this value. Set <=0 to disable.",
    )
    parser.add_argument(
        "--stage1-min-epochs",
        type=int,
        default=1,
        help="Minimum completed stage1 epochs before the mAP50 threshold can trigger.",
    )
    parser.add_argument("--bar-conf", type=float, default=0.35)
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-bar-build", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = prepare_local_yaml(args.data_root)
    args.project.mkdir(parents=True, exist_ok=True)

    stage1_best = args.project / "yolo26s_gear8_ssd_stage1" / "weights" / "best.pt"
    if not args.skip_stage1:
        stage1_name = "yolo26s_gear8_ssd_stage1"
        if _stage1_is_ready(args.project, stage1_name, args.stage1_target_map50):
            stage1_best = args.project / stage1_name / "weights" / "best.pt"
        else:
            stage1_best = train_yolo(
                model_path=args.model,
                data_yaml=spec.yaml_path,
                project=args.project,
                name=stage1_name,
                epochs=args.stage1_epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device,
                workers=args.workers,
                stop_map50=args.stage1_target_map50,
                stop_min_epochs=args.stage1_min_epochs,
            )
    elif not stage1_best.exists():
        raise FileNotFoundError(f"--skip-stage1 was set, but missing {stage1_best}")

    bar_yaml = args.bar_root / "data.yaml"
    if not args.skip_bar_build:
        bar_yaml = build_bar_dataset(
            spec=spec,
            stage1_weights=stage1_best,
            bar_root=args.bar_root,
            imgsz=args.imgsz,
            conf=args.bar_conf,
            device=args.device,
        )
    elif not bar_yaml.exists():
        raise FileNotFoundError(f"--skip-bar-build was set, but missing {bar_yaml}")

    if not args.skip_stage2:
        train_yolo(
            model_path=stage1_best,
            data_yaml=bar_yaml,
            project=args.project,
            name="yolo26s_gear8_bar_stage2",
            epochs=args.stage2_epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
