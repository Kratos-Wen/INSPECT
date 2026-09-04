"""Convert phone videos plus binary masks into TwinSwap training assets.

The importer reads frame-aligned masks such as ``mask_000123.png``, seeks the
matching video frame, writes RGBA object cutouts by class, and writes inpainted
background frames. It never modifies the source videos or masks.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_CLASS_ORDER = [
    "type_6_gearbox_cover",
    "type_5_gearbox_cover",
    "type_6_gearbox_housing",
    "type_5_gearbox_housing",
    "type_7_gear",
    "type_2_gear",
    "type_8_gear",
    "type_3_gear",
]

DEFAULT_SOURCE_TO_CLASS = {
    "Getriebedeckel_typ_6": "type_6_gearbox_cover",
    "Getriebedeckel_typ_5": "type_5_gearbox_cover",
    "Getriebegehaeuse_typ_6": "type_6_gearbox_housing",
    "Getriebegehaeuse_typ_5": "type_5_gearbox_housing",
    "Zahnrad_typ_7": "type_7_gear",
    "Zahnrad_typ_2": "type_2_gear",
    "Zahnrad_typ_8": "type_8_gear",
    "Zahnrad_typ_3": "type_3_gear",
}

MASK_RE = re.compile(r"mask_(?P<frame>\d+)\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class MaskFrame:
    path: Path
    frame_index: int


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_") or "item"


def _source_stem(mask_dir: Path) -> str:
    name = mask_dir.name
    return name[:-5] if name.endswith("_mask") else name


def _base_source_name(stem: str) -> str:
    return stem[:-5] if stem.endswith("_mesh") else stem


def _iter_mask_frames(mask_dir: Path) -> Iterator[MaskFrame]:
    for path in sorted(mask_dir.glob("mask_*.png")):
        match = MASK_RE.search(path.name)
        if match:
            yield MaskFrame(path=path, frame_index=int(match.group("frame")))


def _select_evenly(items: Sequence[MaskFrame], limit: int) -> List[MaskFrame]:
    if limit <= 0:
        return []
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[len(items) // 2]]
    indices = np.linspace(0, len(items) - 1, num=limit, dtype=np.int64)
    return [items[int(index)] for index in indices]


def _read_mask(path: Path, size: Tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")
    width, height = size
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8) * 255


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _pad_box(box: Tuple[int, int, int, int], width: int, height: int, pad_ratio: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pad = int(round(max(x2 - x1, y2 - y1) * pad_ratio))
    return max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)


def _open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def _read_frame(cap: cv2.VideoCapture, frame_index: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    return frame if ok else None


def _safe_clear_directory(path: Path) -> None:
    resolved = path.resolve()
    if len(resolved.parts) < 4:
        raise ValueError(f"Refusing to clear suspiciously broad path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class MaskVideoTwinSwapAssetBuilder:
    """Build object cutouts and inpainted backgrounds from aligned mask/video pairs."""

    def __init__(
        self,
        video_root: Path,
        mask_root: Path,
        output_root: Path,
        include_mesh: bool = False,
        max_cutouts_per_class: int = 500,
        max_backgrounds: int = 1000,
        cutout_pad_ratio: float = 0.08,
        min_mask_area_ratio: float = 0.002,
        max_mask_area_ratio: float = 0.80,
        inpaint_radius: int = 5,
        clear_output: bool = False,
    ) -> None:
        self.video_root = video_root
        self.mask_root = mask_root
        self.output_root = output_root
        self.include_mesh = bool(include_mesh)
        self.max_cutouts_per_class = max(1, int(max_cutouts_per_class))
        self.max_backgrounds = max(0, int(max_backgrounds))
        self.cutout_pad_ratio = max(0.0, float(cutout_pad_ratio))
        self.min_mask_area_ratio = max(0.0, float(min_mask_area_ratio))
        self.max_mask_area_ratio = min(1.0, float(max_mask_area_ratio))
        self.inpaint_radius = max(1, int(inpaint_radius))
        self.clear_output = bool(clear_output)

    def build(self) -> None:
        if self.clear_output:
            _safe_clear_directory(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        object_root = self.output_root / "objects"
        background_root = self.output_root / "backgrounds"
        object_root.mkdir(parents=True, exist_ok=True)
        background_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "classes.txt").write_text("\n".join(DEFAULT_CLASS_ORDER) + "\n", encoding="utf-8")

        cutout_records = self._export_cutouts(object_root)
        background_records = self._export_backgrounds(background_root)
        _write_jsonl(self.output_root / "cutouts.jsonl", cutout_records)
        _write_jsonl(self.output_root / "backgrounds.jsonl", background_records)

    def _export_cutouts(self, object_root: Path) -> List[dict]:
        records: List[dict] = []
        by_class: Dict[str, List[Tuple[Path, MaskFrame]]] = {name: [] for name in DEFAULT_CLASS_ORDER}
        for mask_dir in self._mask_dirs():
            source = _source_stem(mask_dir)
            if source.endswith("_mesh") and not self.include_mesh:
                continue
            class_name = DEFAULT_SOURCE_TO_CLASS.get(_base_source_name(source))
            if not class_name:
                continue
            frames = list(_iter_mask_frames(mask_dir))
            by_class[class_name].extend((mask_dir, frame) for frame in frames)

        for class_name in DEFAULT_CLASS_ORDER:
            candidates = _select_evenly(by_class[class_name], self.max_cutouts_per_class)
            class_dir = object_root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for mask_dir, mask_frame in candidates:
                record = self._write_cutout(class_dir, mask_dir, mask_frame, class_name)
                if record:
                    records.append(record)
        return records

    def _write_cutout(
        self,
        class_dir: Path,
        mask_dir: Path,
        mask_frame: MaskFrame,
        class_name: str,
    ) -> Optional[dict]:
        source = _source_stem(mask_dir)
        video = self.video_root / f"{source}.mp4"
        if not video.exists():
            return None
        cap = _open_video(video)
        try:
            frame = _read_frame(cap, mask_frame.frame_index)
        finally:
            cap.release()
        if frame is None:
            return None
        height, width = frame.shape[:2]
        mask = _read_mask(mask_frame.path, (width, height))
        area_ratio = float((mask > 0).mean())
        if area_ratio < self.min_mask_area_ratio or area_ratio > self.max_mask_area_ratio:
            return None
        box = _bbox_from_mask(mask)
        if box is None:
            return None
        x1, y1, x2, y2 = _pad_box(box, width, height, self.cutout_pad_ratio)
        crop_bgr = frame[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgba = np.dstack([crop_rgb, crop_mask])
        name = f"{_slug(source)}_f{mask_frame.frame_index:06d}.png"
        output = class_dir / name
        cv2.imwrite(str(output), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        return {
            "path": str(output),
            "class_name": class_name,
            "source_video": str(video),
            "mask": str(mask_frame.path),
            "frame_index": int(mask_frame.frame_index),
            "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "mask_area_ratio": area_ratio,
        }

    def _export_backgrounds(self, background_root: Path) -> List[dict]:
        mask_frames: List[Tuple[Path, MaskFrame]] = []
        for mask_dir in self._mask_dirs():
            frames = list(_iter_mask_frames(mask_dir))
            if not frames:
                continue
            source = _source_stem(mask_dir)
            if source.endswith("_mesh"):
                continue
            mask_frames.extend((mask_dir, frame) for frame in frames)
        selected = _select_evenly(mask_frames, self.max_backgrounds)
        records: List[dict] = []
        for mask_dir, mask_frame in selected:
            record = self._write_background(background_root, mask_dir, mask_frame)
            if record:
                records.append(record)
        return records

    def _write_background(self, background_root: Path, mask_dir: Path, mask_frame: MaskFrame) -> Optional[dict]:
        source = _source_stem(mask_dir)
        video = self.video_root / f"{source}.mp4"
        if not video.exists():
            return None
        cap = _open_video(video)
        try:
            frame = _read_frame(cap, mask_frame.frame_index)
        finally:
            cap.release()
        if frame is None:
            return None
        height, width = frame.shape[:2]
        mask = _read_mask(mask_frame.path, (width, height))
        dilate_kernel = np.ones((9, 9), np.uint8)
        inpaint_mask = cv2.dilate(mask, dilate_kernel, iterations=1)
        background = cv2.inpaint(frame, inpaint_mask, self.inpaint_radius, cv2.INPAINT_TELEA)
        output = background_root / f"{_slug(source)}_bg_f{mask_frame.frame_index:06d}.jpg"
        cv2.imwrite(str(output), background, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return {
            "path": str(output),
            "source_video": str(video),
            "mask": str(mask_frame.path),
            "frame_index": int(mask_frame.frame_index),
        }

    def _mask_dirs(self) -> List[Path]:
        return sorted(path for path in self.mask_root.iterdir() if path.is_dir())


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build TwinSwap assets from videos and binary masks")
    parser.add_argument("--video-root", type=Path, default=Path("data/raw_videos"))
    parser.add_argument("--mask-root", type=Path, default=Path("data/masks"))
    parser.add_argument("--output-root", type=Path, default=Path("data/twinswap_assets"))
    parser.add_argument("--include-mesh", action="store_true", help="Also use *_mesh videos as object cutout sources.")
    parser.add_argument("--max-cutouts-per-class", type=int, default=500)
    parser.add_argument("--max-backgrounds", type=int, default=1000)
    parser.add_argument("--cutout-pad-ratio", type=float, default=0.08)
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.002)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.80)
    parser.add_argument("--inpaint-radius", type=int, default=5)
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    builder = MaskVideoTwinSwapAssetBuilder(
        video_root=args.video_root,
        mask_root=args.mask_root,
        output_root=args.output_root,
        include_mesh=args.include_mesh,
        max_cutouts_per_class=args.max_cutouts_per_class,
        max_backgrounds=args.max_backgrounds,
        cutout_pad_ratio=args.cutout_pad_ratio,
        min_mask_area_ratio=args.min_mask_area_ratio,
        max_mask_area_ratio=args.max_mask_area_ratio,
        inpaint_radius=args.inpaint_radius,
        clear_output=args.clear_output,
    )
    builder.build()
    print(f"TwinSwap assets written to {args.output_root}")


if __name__ == "__main__":
    main()
