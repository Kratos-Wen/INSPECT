"""Offline evaluation for ego/video INSPECT trace runs."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


_STATE_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "一": "1",
    "二": "2",
    "两": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "十": "10",
}

UNSCORED_STATES = {"UNCERTAIN"}


@dataclass(frozen=True)
class EgoStateSegment:
    """One ground-truth procedural state interval in frame coordinates."""

    state: str
    start_frame: int
    end_frame: Optional[int] = None
    source: str = ""
    text: str = ""

    def contains(self, frame_index: int) -> bool:
        if frame_index < self.start_frame:
            return False
        if self.end_frame is None:
            return True
        return frame_index <= self.end_frame

    def to_dict(self) -> Dict[str, object]:
        return {
            "state": self.state,
            "start_frame": int(self.start_frame),
            "end_frame": (int(self.end_frame) if self.end_frame is not None else None),
            "source": self.source,
            "text": self.text,
        }


def normalize_state(value: object) -> str:
    """Normalize common audio/annotation variants into S1/S2/... labels."""

    text = str(value or "").strip()
    if not text:
        return ""
    lowered_raw = text.lower().strip()
    if lowered_raw in {
        "wrong",
        "invalid",
        "no_step",
        "no step",
        "negative",
        "hard_negative",
        "hold",
        "unknown",
        "abstain",
    }:
        return "INVALID"
    if lowered_raw.startswith("wrong") or lowered_raw.startswith("invalid") or lowered_raw.startswith("no_step"):
        return "INVALID"
    upper = text.upper().replace("STATE_", "STATE ").replace("STEP_", "STEP ")
    match = re.search(r"\bS\s*([0-9]+)\b", upper)
    if match:
        return f"S{int(match.group(1))}"
    match = re.search(r"\b(?:STATE|STEP)\s*([0-9]+)\b", upper)
    if match:
        return f"S{int(match.group(1))}"
    lowered = text.lower()
    for word, number in _STATE_WORDS.items():
        if re.search(rf"(?:state|step|状态|步骤)\s*{re.escape(word)}\b", lowered) or f"状态{word}" in text:
            return f"S{int(number)}"
    if re.fullmatch(r"[0-9]+", text):
        return f"S{int(text)}"
    compact = re.sub(r"[^a-zA-Z0-9]", "", upper)
    match = re.fullmatch(r"(?:STATE|STEP)?([0-9]+)", compact)
    if match:
        return f"S{int(match.group(1))}"
    return upper


def _read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if Path(path).suffix.lower() == ".jsonl":
        records: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(dict(json.loads(line)))
        return records
    if text.startswith("["):
        return [dict(item) for item in json.loads(text)]
    if text.startswith("{"):
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("segments", "annotations", "labels", "events", "items"):
                values = payload.get(key)
                if isinstance(values, list):
                    return [dict(item) for item in values if isinstance(item, dict)]
            return [dict(payload)]
    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(dict(json.loads(line)))
    return records


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_txt_ranges(path: Path) -> List[Dict[str, Any]]:
    """Read simple range annotations such as ``0~24 step1``.

    The range endpoints are interpreted as seconds, which matches the common
    audio-note workflow used for ego videos. Use JSON/CSV with explicit
    ``start_frame``/``end_frame`` fields when frame-accurate labels are needed.
    """

    records: List[Dict[str, Any]] = []
    pattern = re.compile(
        r"^\s*(?P<start>[0-9]+(?:\.[0-9]+)?)\s*(?:~|-|,)\s*(?P<end>[0-9]+(?:\.[0-9]+)?)\s+(?P<label>.+?)\s*$"
    )
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            match = pattern.match(text)
            if not match:
                continue
            records.append(
                {
                    "start_time": float(match.group("start")),
                    "end_time": float(match.group("end")),
                    "state": match.group("label").strip(),
                    "source": f"txt:{index}",
                    "text": text,
                }
            )
    return records


def _read_records(path: Path) -> List[Dict[str, Any]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return _read_csv(path)
    if suffix in {".txt", ".ann", ".label", ".labels"}:
        return _read_txt_ranges(path)
    return _read_json_or_jsonl(path)


def _first_value(record: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _float_value(record: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    value = _first_value(record, keys)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_frame(record: Dict[str, Any], keys: Sequence[str]) -> Optional[int]:
    value = _float_value(record, keys)
    if value is None:
        return None
    return int(round(value))


def _state_from_record(record: Dict[str, Any]) -> str:
    for key in ("state", "gt_state", "ground_truth_state", "label", "step", "state_id"):
        state = normalize_state(record.get(key))
        if state:
            return state
    for key in ("text", "transcript", "utterance", "note", "audio_text"):
        state = normalize_state(record.get(key))
        if state:
            return state
    return ""


def _fps_from_run(run_dir: Path, fallback: float = 30.0) -> float:
    meta_path = Path(run_dir) / "meta.json"
    if not meta_path.exists():
        return float(fallback)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return float(fallback)
    for key in ("fps", "source_fps", "video_fps"):
        if key in meta:
            try:
                value = float(meta[key])
                if value > 1e-6:
                    return value
            except (TypeError, ValueError):
                pass
    video_path = str(meta.get("video_path", "") or "")
    if video_path:
        try:
            import cv2

            capture = cv2.VideoCapture(video_path)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            capture.release()
            if fps > 1e-6:
                return fps
        except Exception:
            return float(fallback)
    return float(fallback)


def load_ego_state_segments(
    gt_path: Path,
    fps: float,
    max_frame: Optional[int] = None,
) -> List[EgoStateSegment]:
    """Load segment or transition-style ego state annotations.

    Supported examples:
    ``{"start_time": 2.1, "end_time": 8.4, "state": "S2"}``
    ``{"start_frame": 63, "end_frame": 252, "label": "State 2"}``
    ``{"timestamp": 8.4, "transcript": "state three"}``
    """

    records = _read_records(Path(gt_path))
    segments: List[EgoStateSegment] = []
    transition_events: List[EgoStateSegment] = []
    fps_value = max(1e-6, float(fps))
    for index, record in enumerate(records):
        state = _state_from_record(record)
        if not state:
            continue
        start_frame = _int_frame(record, ("start_frame", "frame_start", "begin_frame", "frame_index", "frame"))
        end_frame = _int_frame(record, ("end_frame", "frame_end", "stop_frame"))
        if start_frame is None:
            start_time = _float_value(record, ("start_time", "time_start", "begin_time", "start_sec", "start"))
            if start_time is not None:
                start_frame = int(round(start_time * fps_value))
        if end_frame is None:
            end_time = _float_value(record, ("end_time", "time_end", "stop_time", "end_sec", "end"))
            if end_time is not None:
                end_frame = int(round(end_time * fps_value))
        if start_frame is None:
            timestamp = _float_value(record, ("timestamp", "time", "sec", "seconds", "audio_time"))
            if timestamp is not None:
                start_frame = int(round(timestamp * fps_value))
        if start_frame is None:
            continue
        segment = EgoStateSegment(
            state=state,
            start_frame=max(0, int(start_frame)),
            end_frame=(max(0, int(end_frame)) if end_frame is not None else None),
            source=str(record.get("source", f"gt:{index}")),
            text=str(_first_value(record, ("text", "transcript", "utterance", "note", "audio_text")) or ""),
        )
        if segment.end_frame is None:
            transition_events.append(segment)
        else:
            segments.append(segment)

    transition_events.sort(key=lambda item: item.start_frame)
    for index, event in enumerate(transition_events):
        next_start = transition_events[index + 1].start_frame if index + 1 < len(transition_events) else None
        end = (next_start - 1) if next_start is not None else max_frame
        segments.append(
            EgoStateSegment(
                state=event.state,
                start_frame=event.start_frame,
                end_frame=end,
                source=event.source,
                text=event.text,
            )
        )

    segments.sort(key=lambda item: (item.start_frame, item.end_frame if item.end_frame is not None else math.inf))
    return segments


def _read_iterations(run_dir: Path) -> List[Dict[str, Any]]:
    path = Path(run_dir) / "iterations.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"iterations.jsonl not found in run directory: {run_dir}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(dict(json.loads(line)))
    return records


def _truth_for_frame(segments: Sequence[EgoStateSegment], frame_index: int) -> str:
    for segment in segments:
        if segment.contains(frame_index):
            return segment.state
    return ""


def _f1_by_state(pairs: Sequence[tuple[str, str]]) -> Dict[str, Dict[str, float]]:
    labels = sorted({truth for truth, _ in pairs} | {pred for _, pred in pairs})
    metrics: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for truth, pred in pairs if truth == label and pred == label)
        fp = sum(1 for truth, pred in pairs if truth != label and pred == label)
        fn = sum(1 for truth, pred in pairs if truth == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(sum(1 for truth, _ in pairs if truth == label)),
        }
    return metrics


def _confusion(pairs: Sequence[tuple[str, str]]) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = {}
    for truth, pred in pairs:
        matrix.setdefault(truth, {})
        matrix[truth][pred] = matrix[truth].get(pred, 0) + 1
    return matrix


def _transition_metrics(
    records: Sequence[Dict[str, Any]],
    segments: Sequence[EgoStateSegment],
    prediction_key: str,
    fps: float,
    window_sec: float,
) -> Dict[str, object]:
    predictions = [
        (int(record.get("frame_index", 0) or 0), normalize_state(record.get(prediction_key)))
        for record in records
        if normalize_state(record.get(prediction_key))
    ]
    if not predictions:
        return {
            "num_gt_transitions": max(0, len(segments) - 1),
            "num_matched_transitions": 0,
            "transition_recall": None,
            "mean_abs_transition_error_frames": None,
            "mean_abs_transition_error_sec": None,
        }
    window = int(round(max(0.0, float(window_sec)) * max(1e-6, float(fps))))
    errors: List[int] = []
    gt_transitions = list(segments[1:])
    for segment in gt_transitions:
        candidates = [
            frame - segment.start_frame
            for frame, pred in predictions
            if pred == segment.state and abs(frame - segment.start_frame) <= window
        ]
        if candidates:
            errors.append(min(candidates, key=lambda item: abs(item)))
    return {
        "num_gt_transitions": len(gt_transitions),
        "num_matched_transitions": len(errors),
        "transition_recall": (len(errors) / len(gt_transitions) if gt_transitions else None),
        "mean_abs_transition_error_frames": (
            sum(abs(value) for value in errors) / len(errors) if errors else None
        ),
        "mean_abs_transition_error_sec": (
            sum(abs(value) for value in errors) / (len(errors) * max(1e-6, float(fps))) if errors else None
        ),
        "signed_transition_errors_frames": errors,
    }


def evaluate_ego_run(
    run_dir: Path,
    gt_path: Path,
    output_path: Optional[Path] = None,
    prediction_key: str = "fused_step",
    fps: Optional[float] = None,
    transition_window_sec: float = 3.0,
    write_aligned_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Evaluate an INSPECT ego/video run against state timeline annotations."""

    run_dir = Path(run_dir)
    records = _read_iterations(run_dir)
    max_frame = max((int(record.get("frame_index", 0) or 0) for record in records), default=0)
    fps_value = float(fps) if fps is not None else _fps_from_run(run_dir)
    segments = load_ego_state_segments(Path(gt_path), fps=fps_value, max_frame=max_frame)

    scored_segments = [segment for segment in segments if segment.state not in UNSCORED_STATES]
    pairs: List[tuple[str, str]] = []
    aligned: List[Dict[str, object]] = []
    skipped = 0
    correct = 0
    for record in records:
        frame_index = int(record.get("frame_index", 0) or 0)
        truth = _truth_for_frame(segments, frame_index)
        pred = normalize_state(record.get(prediction_key))
        if not truth or truth in UNSCORED_STATES:
            skipped += 1
            continue
        pairs.append((truth, pred))
        if truth == pred:
            correct += 1
        aligned.append(
            {
                "frame_index": frame_index,
                "time_sec": frame_index / max(1e-6, fps_value),
                "truth": truth,
                "prediction": pred,
                "correct": bool(truth == pred),
                "confidence": float(record.get("fused_conf", 0.0) or 0.0),
                "stable": bool(record.get("stable", False)),
                "has_visual_evidence": bool(record.get("has_visual_evidence", False)),
            }
        )

    by_state = _f1_by_state(pairs)
    macro_f1 = sum(item["f1"] for item in by_state.values()) / len(by_state) if by_state else None
    summary: Dict[str, object] = {
        "run_dir": str(run_dir),
        "gt_path": str(gt_path),
        "prediction_key": prediction_key,
        "fps": fps_value,
        "num_iterations": len(records),
        "num_labeled_iterations": len(pairs),
        "num_unlabeled_iterations": skipped,
        "accuracy": (correct / len(pairs) if pairs else None),
        "macro_f1": macro_f1,
        "per_state": by_state,
        "confusion": _confusion(pairs),
        "segments": [segment.to_dict() for segment in segments],
        "unscored_states": sorted(UNSCORED_STATES),
        "transition_metrics": _transition_metrics(
            records,
            scored_segments,
            prediction_key=prediction_key,
            fps=fps_value,
            window_sec=transition_window_sec,
        ),
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if write_aligned_path is not None:
        aligned_path = Path(write_aligned_path)
        aligned_path.parent.mkdir(parents=True, exist_ok=True)
        with aligned_path.open("w", encoding="utf-8") as handle:
            for row in aligned:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return summary
