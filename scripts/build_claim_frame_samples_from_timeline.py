"""Build frame-level claim samples from annotated procedural intervals.

This is the correct bridge from timeline annotations to INSPECT-Active view
learning. A timeline row defines a claim/world-outcome interval. It does not
define a successful view transition. This script therefore samples multiple
frames inside each annotated interval and preserves the interval label as
frame-level supervision. Downstream scripts should run the detector/verifier on
these frames, form evidence traces, and mine low-evidence to high-evidence
transitions automatically from those traces.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2


def normalize_key(value: object) -> str:
    text = str(value or "").lower().strip().replace("-", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text).strip("_")


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return count


def resolve_video_path(raw: str, video_dir: Path | None) -> Path:
    path = Path(str(raw or ""))
    if path.exists():
        return path
    if video_dir is not None:
        candidate = video_dir / path.name
        if candidate.exists():
            return candidate
    return path


def parse_frame(value: object, default: int = 0) -> int:
    try:
        return int(round(float(str(value).strip())))
    except Exception:
        return int(default)


def evenly_limit(frames: List[int], max_count: int) -> List[int]:
    if max_count <= 0 or len(frames) <= max_count:
        return frames
    if max_count == 1:
        return [frames[len(frames) // 2]]
    selected: List[int] = []
    last = len(frames) - 1
    for index in range(max_count):
        selected.append(frames[round(index * last / (max_count - 1))])
    return sorted(set(selected))


def sample_interval(start: int, end: int, stride: int, max_count: int, edge_margin: int) -> List[int]:
    start = start + max(0, edge_margin)
    end = end - max(0, edge_margin)
    if end < start:
        return []
    step = max(1, stride)
    frames = list(range(start, end + 1, step))
    if frames and frames[-1] != end:
        frames.append(end)
    if not frames:
        frames = [start]
    return evenly_limit(frames, max_count)


def row_is_skipped(row: Dict[str, str]) -> bool:
    return normalize_key(row.get("skip", "")) in {"1", "true", "yes", "y"}


def make_samples(
    rows: Iterable[Dict[str, str]],
    video_dir: Path | None,
    stride: int,
    max_samples_per_event: int,
    edge_margin: int,
    include_unresolved: bool,
) -> tuple[List[Dict[str, Any]], List[str]]:
    samples: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for row_index, row in enumerate(rows, start=1):
        if row_is_skipped(row):
            continue
        outcome = normalize_key(row.get("outcome", ""))
        if outcome == "unresolved" and not include_unresolved:
            continue
        video = resolve_video_path(row.get("video", ""), video_dir)
        count = frame_count(video)
        if count <= 0:
            skipped.append(f"row {row_index}: cannot read video {video}")
            continue
        start = max(0, min(count - 1, parse_frame(row.get("start_frame", 0))))
        end = max(0, min(count - 1, parse_frame(row.get("end_frame", start), start)))
        if end < start:
            skipped.append(f"row {row_index}: invalid interval {start}->{end} for {video.name}")
            continue
        frames = sample_interval(start, end, stride, max_samples_per_event, edge_margin)
        if not frames:
            skipped.append(f"row {row_index}: no frames after margin for {video.name}")
            continue
        event_id = row.get("event_id") or f"{video.stem}_event_{row_index:04d}"
        for sample_index, frame in enumerate(frames):
            samples.append(
                {
                    "sample_id": f"{event_id}_f{frame:06d}",
                    "event_id": event_id,
                    "video": str(video),
                    "frame": int(frame),
                    "sample_index": int(sample_index),
                    "num_samples_in_event": int(len(frames)),
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "assembly_set": row.get("assembly_set", ""),
                    "step_id": row.get("step_id", ""),
                    "claim_id": row.get("claim_id", ""),
                    "world_outcome": row.get("outcome", ""),
                    "note": row.get("note", ""),
                    "source": "timeline_interval_frame_sample",
                    "metadata": {
                        "uses_human_segment_boundary": True,
                        "not_relative_action": True,
                        "not_view_transition": True,
                        "uses_robot_view_training": False,
                        "intended_next_stage": "run detector/verifier evidence, then mine evidence-trace transitions",
                    },
                }
            )
    return samples, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-events-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=15, help="Frame stride inside each labeled interval.")
    parser.add_argument("--max-samples-per-event", type=int, default=12, help="Evenly cap samples per interval; <=0 keeps all.")
    parser.add_argument("--edge-margin-frames", type=int, default=0, help="Ignore this many frames at each interval edge.")
    parser.add_argument("--exclude-unresolved", action="store_true", help="Skip unresolved intervals.")
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples, skipped = make_samples(
        rows=read_rows(args.claim_events_csv),
        video_dir=args.video_dir,
        stride=args.stride,
        max_samples_per_event=args.max_samples_per_event,
        edge_margin=args.edge_margin_frames,
        include_unresolved=not args.exclude_unresolved,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "output": str(args.output),
        "samples": len(samples),
        "skipped": skipped[:30],
        "by_world_outcome": {},
        "by_step_id": {},
        "by_claim_id": {},
    }
    for sample in samples:
        for key, field in (
            ("by_world_outcome", "world_outcome"),
            ("by_step_id", "step_id"),
            ("by_claim_id", "claim_id"),
        ):
            value = str(sample.get(field, "")) or "UNKNOWN"
            report[key][value] = int(report[key].get(value, 0)) + 1
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
