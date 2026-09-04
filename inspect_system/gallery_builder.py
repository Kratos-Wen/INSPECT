"""Build retrieval galleries from annotated ego videos."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2

from .ego_evaluation import load_ego_state_segments, normalize_state


NEGATIVE_LABELS = {"WRONG", "INVALID", "NO_STEP", "NEGATIVE"}


@dataclass(frozen=True)
class GalleryBuildSummary:
    """Summary for one gallery build."""

    output_dir: str
    positives: Dict[str, int]
    negatives: int
    skipped: int
    fps: float
    num_segments: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "output_dir": self.output_dir,
            "positives": dict(self.positives),
            "negatives": int(self.negatives),
            "skipped": int(self.skipped),
            "fps": float(self.fps),
            "num_segments": int(self.num_segments),
        }


def _video_meta(video_path: Path) -> tuple[float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    return fps, num_frames


def _sample_frames(start: int, end: int, samples: int, margin: int) -> List[int]:
    lo = max(start, start + margin)
    hi = min(end, end - margin)
    if hi < lo:
        lo, hi = start, end
    if hi < lo:
        return []
    count = max(1, int(samples))
    if count == 1:
        return [int(round((lo + hi) / 2.0))]
    span = max(0, hi - lo)
    return [int(round(lo + span * index / float(count - 1))) for index in range(count)]


def _state_to_dir(state: str, steps: Iterable[str]) -> Optional[str]:
    raw = str(state or "").strip().lower().replace("-", "_").replace(" ", "_")
    normalized = normalize_state(state)
    step_set = {str(step).strip().upper() for step in steps}
    if normalized in step_set:
        return normalized
    if raw in {"wrong", "invalid", "no_step", "negative", "hard_negative"}:
        return "wrong"
    if raw.startswith("wrong") or raw.startswith("invalid") or raw.startswith("no_step") or raw.startswith("negative"):
        return raw
    if normalized in NEGATIVE_LABELS or normalized.startswith("WRONG") or normalized.startswith("INVALID"):
        return "wrong"
    return None


def build_gallery_from_ego_video(
    *,
    video_path: Path,
    gt_path: Path,
    output_dir: Path,
    steps: Iterable[str] = ("S1", "S2", "S3", "S4"),
    samples_per_segment: int = 18,
    margin_sec: float = 1.0,
    image_quality: int = 92,
    overwrite: bool = False,
) -> GalleryBuildSummary:
    """Sample positive step references and wrong hard negatives from an annotated ego video."""

    video_path = Path(video_path)
    gt_path = Path(gt_path)
    output_dir = Path(output_dir)
    fps, num_frames = _video_meta(video_path)
    segments = load_ego_state_segments(gt_path, fps=fps, max_frame=max(0, num_frames - 1))
    margin_frames = int(round(max(0.0, float(margin_sec)) * fps))
    step_list = [str(step).strip().upper() for step in steps]
    positives: Dict[str, int] = {step: 0 for step in step_list}
    negatives = 0
    skipped = 0

    if overwrite and output_dir.exists():
        for path in output_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    try:
        for segment_index, segment in enumerate(segments):
            if segment.end_frame is None:
                continue
            target_dir_name = _state_to_dir(segment.state, step_list)
            if target_dir_name is None:
                skipped += 1
                continue
            target_dir = output_dir / target_dir_name
            target_dir.mkdir(parents=True, exist_ok=True)
            frame_ids = _sample_frames(
                int(segment.start_frame),
                int(segment.end_frame),
                samples=int(samples_per_segment),
                margin=margin_frames,
            )
            for sample_index, frame_id in enumerate(frame_ids):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
                ok, frame = capture.read()
                if not ok or frame is None:
                    skipped += 1
                    continue
                output_path = target_dir / f"{video_path.stem}_seg{segment_index:03d}_f{frame_id:06d}_{sample_index:02d}.jpg"
                cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(image_quality)])
                if target_dir_name == "wrong":
                    negatives += 1
                else:
                    positives[target_dir_name] = positives.get(target_dir_name, 0) + 1
    finally:
        capture.release()

    summary = GalleryBuildSummary(
        output_dir=str(output_dir),
        positives=positives,
        negatives=negatives,
        skipped=skipped,
        fps=fps,
        num_segments=len(segments),
    )
    (output_dir / "gallery_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary
