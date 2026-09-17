"""Re-evaluate historical paired checkpoints with per-image preprocessing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import torch
import ultralytics
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_yolo_gt_dataset as evaluator


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/object_detector_test_yolo_gt"),
                        help="Corrected detector test set (images/ + labels/ YOLO layout).")
    parser.add_argument("--pair-root", type=Path, default=Path("data/paper_detector_fair"),
                        help="Root that contains the paired detector training runs.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/detector_per_image")
    args = parser.parse_args()
    runs = {
        "paired_no_pi": args.pair_root / "runs_twinswap/yolo26s_twinswap_matched_4000pairs_xteacher_no_pi",
        "pi_twinswap": args.pair_root / "runs_pi_twinswap/yolo26s_pi_twinswap_matched_4000pairs_xteacher_stage2_ampfix",
    }
    configs = {name: yaml.safe_load((path / "args.yaml").read_text()) for name, path in runs.items()}
    a, b = configs.values()
    differences = {k: [a.get(k), b.get(k)] for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    substantive = {k: v for k, v in differences.items() if k not in {"name", "project", "save_dir"}}
    classes = evaluator.read_classes(args.dataset / "classes.txt")
    images = sorted(p for p in (args.dataset / "images/test").iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"})
    dataset_manifest = [
        {"image": p.name, "image_sha256": digest(p),
         "label_sha256": digest(args.dataset / "labels/test" / (p.stem + ".txt"))}
        for p in images
    ]
    results, checkpoints, details = [], {}, []
    for name, path in runs.items():
        weights = path / "weights/best.pt"
        model = evaluator.load_model(str(weights))
        stats = evaluator.empty_stats(classes)
        for index, image in enumerate(images):
            frame = cv2.imread(str(image))
            assert frame is not None, image
            height, width = frame.shape[:2]
            gt = evaluator.read_yolo_label(args.dataset / "labels/test" / (image.stem + ".txt"), width, height)
            pred = evaluator.predict(model=model, image_path=image, class_names=classes,
                                     conf=0.10, iou=0.45, device="0", end2end=False)
            matches = evaluator.greedy_match(gt, pred, 0.50)
            evaluator.add_image_stats(stats=stats, gt=gt, pred=pred, matches=matches, class_names=classes)
            details.append({"model": name, "image": image.name, "gt": len(gt), "pred": len(pred),
                            "matched": len(matches), "correct": sum(gt[g].cls_id == pred[p].cls_id for g, p, _ in matches)})
            if (index + 1) % 100 == 0:
                print(f"{name}: {index + 1}/{len(images)}", flush=True)
        result = evaluator.summarize(stats, model_name=name, weights=str(weights), conf=0.1, iou_threshold=0.5)
        results.append(result)
        with (path / "results.csv").open(newline="") as stream:
            epochs = list(csv.DictReader(stream))
        checkpoints[name] = {"path": str(weights), "sha256": digest(weights),
                             "completed_epochs": len(epochs), "args_sha256": digest(path / "args.yaml")}
        del model
        torch.cuda.empty_cache()
    receipt = {
        "protocol": {"conf": 0.1, "nms_iou": 0.45, "match_iou": 0.5, "end2end": False,
                     "device": 0, "preprocessing": "one image per predict call, native aspect-ratio letterbox"},
        "versions": {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                     "gpu": torch.cuda.get_device_name(0)},
        "checkpoints": checkpoints, "training_argument_differences": differences,
        "substantive_standard_argument_differences": substantive,
        "custom_loss_coefficients_verified": False,
        "attribution_limit": "Standard args and budget match. Historical custom-loss CLI coefficients are absent from saved args; run names alone do not certify PI-only attribution.",
        "dataset_manifest": dataset_manifest, "script_sha256": digest(__file__),
        "results": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    evaluator.write_csv(args.output / "image_details.csv", details)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
