"""Create past-only refresh schedules for frozen appearance embeddings.

The input cache is produced from online predicted-box ROIs. This utility never
reads task labels, filenames as features, or future observations. At a refresh
frame it stores the current embedding; between refreshes it reuses only the
most recent stored embedding from the same video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


FORBIDDEN_PROPOSAL_FIELDS = {
    "target",
    "truth",
    "label",
    "outcome",
    "outcome_norm",
    "step_id",
}


def video_key(value: object) -> str:
    """Return a case-normalized basename for either path separator."""
    return Path(str(value or "").replace(chr(92), "/")).name.strip().lower()


def causal_reuse(
    videos: np.ndarray,
    frames: np.ndarray,
    embeddings: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return causally cached embeddings and a refresh indicator."""

    output = np.asarray(embeddings).copy()
    refreshed = np.zeros(len(frames), dtype=np.uint8)
    by_video: dict[str, list[int]] = {}
    for index, video in enumerate(videos):
        by_video.setdefault(str(video), []).append(index)
    for indices in by_video.values():
        ordered = sorted(indices, key=lambda index: int(frames[index]))
        cached: np.ndarray | None = None
        for observation_index, index in enumerate(ordered):
            if cached is None or observation_index % stride == 0:
                cached = np.asarray(embeddings[index]).copy()
                refreshed[index] = 1
            else:
                output[index] = cached
    return output, refreshed


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1])
    )
    second_area = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1])
    )
    return intersection / max(1e-8, first_area + second_area - intersection)


def load_proposal_steps(path: Path | None) -> dict[tuple[str, int], str]:
    if path is None:
        return {}
    output: dict[tuple[str, int], str] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record: dict[str, Any] = json.loads(line)
            leaked = FORBIDDEN_PROPOSAL_FIELDS.intersection(record)
            if leaked:
                names = ", ".join(sorted(leaked))
                raise ValueError(f"Proposal cache contains forbidden fields: {names}")
            video = video_key(record.get("video", ""))
            frame = int(record.get("frame", -1))
            step = str(record.get("proposed_step", "")).strip().upper()
            if not video or frame < 0 or step not in {"S1", "S2", "S3", "S4"}:
                raise ValueError("Proposal rows require video, frame, and proposed_step")
            output[(video, frame)] = step
    return output


def adaptive_causal_reuse(
    videos: np.ndarray,
    frames: np.ndarray,
    embeddings: np.ndarray,
    roi_valid: np.ndarray,
    detection_count: np.ndarray,
    roi_boxes: np.ndarray,
    proposal_steps: dict[tuple[str, int], str],
    *,
    max_interval: int,
    roi_iou_threshold: float,
    enabled_signals: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Refresh on cheap evidence changes, otherwise reuse only past evidence."""

    signals = enabled_signals or {
        "proposal",
        "detection",
        "visibility",
        "roi",
    }
    output = np.asarray(embeddings).copy()
    refreshed = np.zeros(len(frames), dtype=np.uint8)
    reasons = {
        "initial": 0,
        "max_interval": 0,
        "proposal_change": 0,
        "detection_change": 0,
        "visibility_change": 0,
        "roi_change": 0,
    }
    by_video: dict[str, list[int]] = {}
    for index, video in enumerate(videos):
        by_video.setdefault(str(video), []).append(index)
    for video, indices in by_video.items():
        ordered = sorted(indices, key=lambda index: int(frames[index]))
        cached: np.ndarray | None = None
        last_refresh_position = -max_interval
        previous_index: int | None = None
        previous_step = ""
        for position, index in enumerate(ordered):
            key = (video_key(video), int(frames[index]))
            current_step = proposal_steps.get(key, previous_step)
            triggers: list[str] = []
            if cached is None:
                triggers.append("initial")
            elif position - last_refresh_position >= max_interval:
                triggers.append("max_interval")
            if previous_index is not None:
                if (
                    "proposal" in signals
                    and current_step
                    and previous_step
                    and current_step != previous_step
                ):
                    triggers.append("proposal_change")
                if (
                    "detection" in signals
                    and int(detection_count[index]) != int(detection_count[previous_index])
                ):
                    triggers.append("detection_change")
                if (
                    "visibility" in signals
                    and int(roi_valid[index]) != int(roi_valid[previous_index])
                ):
                    triggers.append("visibility_change")
                if (
                    "roi" in signals
                    and box_iou(roi_boxes[index], roi_boxes[previous_index])
                    < roi_iou_threshold
                ):
                    triggers.append("roi_change")
            if triggers:
                cached = np.asarray(embeddings[index]).copy()
                refreshed[index] = 1
                last_refresh_position = position
                for reason in set(triggers):
                    reasons[reason] += 1
            else:
                output[index] = cached
            previous_index = index
            if current_step:
                previous_step = current_step
    return output, refreshed, reasons


def apply_refresh_schedule(
    videos: np.ndarray,
    frames: np.ndarray,
    embeddings: np.ndarray,
    refreshed: np.ndarray,
) -> np.ndarray:
    """Apply an existing refresh mask in per-video chronological order."""

    output = np.asarray(embeddings).copy()
    by_video: dict[str, list[int]] = {}
    for index, video in enumerate(videos):
        by_video.setdefault(str(video), []).append(index)
    for indices in by_video.values():
        cached: np.ndarray | None = None
        for index in sorted(indices, key=lambda value: int(frames[value])):
            if bool(refreshed[index]) or cached is None:
                cached = np.asarray(embeddings[index]).copy()
            else:
                output[index] = cached
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--max-interval", type=int, default=4)
    parser.add_argument("--roi-iou-threshold", type=float, default=0.75)
    parser.add_argument("--proposal-jsonl", type=Path)
    parser.add_argument(
        "--change-signals",
        nargs="+",
        choices=("proposal", "detection", "visibility", "roi"),
        default=("proposal", "detection", "visibility", "roi"),
    )
    args = parser.parse_args()

    stride = max(1, int(args.stride))
    payload = np.load(args.input, allow_pickle=False)
    required = {"video", "frame", "roi_embedding"}
    missing = required.difference(payload.files)
    if missing:
        raise KeyError(f"Missing cache fields: {', '.join(sorted(missing))}")

    arrays = {name: np.asarray(payload[name]) for name in payload.files}
    trigger_counts: dict[str, int] = {}
    proposal_steps = load_proposal_steps(args.proposal_jsonl)
    if args.adaptive:
        adaptive_required = {"roi_valid", "detection_count", "roi_box"}
        adaptive_missing = adaptive_required.difference(arrays)
        if adaptive_missing:
            names = ", ".join(sorted(adaptive_missing))
            raise KeyError(f"Adaptive cache requires fields: {names}")
        arrays["roi_embedding"], refreshed, trigger_counts = adaptive_causal_reuse(
            arrays["video"],
            arrays["frame"],
            arrays["roi_embedding"],
            arrays["roi_valid"],
            arrays["detection_count"],
            arrays["roi_box"],
            proposal_steps,
            max_interval=max(1, int(args.max_interval)),
            roi_iou_threshold=max(0.0, min(1.0, float(args.roi_iou_threshold))),
            enabled_signals=set(args.change_signals),
        )
    else:
        arrays["roi_embedding"], refreshed = causal_reuse(
            arrays["video"],
            arrays["frame"],
            arrays["roi_embedding"],
            stride,
        )
    if "frame_embedding" in arrays:
        if args.adaptive:
            arrays["frame_embedding"] = apply_refresh_schedule(
                arrays["video"],
                arrays["frame"],
                arrays["frame_embedding"],
                refreshed,
            )
        else:
            arrays["frame_embedding"], frame_refreshed = causal_reuse(
                arrays["video"],
                arrays["frame"],
                arrays["frame_embedding"],
                stride,
            )
            if not np.array_equal(refreshed, frame_refreshed):
                raise RuntimeError("ROI and frame refresh schedules diverged")
    arrays["appearance_refreshed"] = refreshed

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        "source": str(args.input),
        "samples": int(len(refreshed)),
        "videos": int(len(set(str(value) for value in arrays["video"]))),
        "schedule": "evidence_change" if args.adaptive else "fixed_stride",
        "refresh_stride": None if args.adaptive else stride,
        "max_interval": max(1, int(args.max_interval)) if args.adaptive else None,
        "roi_iou_threshold": (
            max(0.0, min(1.0, float(args.roi_iou_threshold)))
            if args.adaptive
            else None
        ),
        "proposal_cache_used": bool(proposal_steps),
        "change_signals": list(args.change_signals) if args.adaptive else [],
        "trigger_counts": trigger_counts,
        "refreshes": int(refreshed.sum()),
        "refresh_rate": float(refreshed.mean()) if len(refreshed) else 0.0,
        "causal": True,
        "future_frames_used": False,
        "task_labels_used": False,
        "ground_truth_boxes_used": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
