"""Build and review INSPECT procedural claim events from assistant videos.

This tool sits after step-transition annotation and before verifier training.
It converts video names and optional ``*_anno.jsonl`` step segments into a
claim-level CSV:

    video,start_frame,end_frame,assembly_set,step_id,claim_id,outcome,...

Most specially recorded error/view videos are inferred directly from their
file names. Normal assistant-use videos can reuse ``annotate_step_transitions``
outputs when available.

Examples:
    python scripts/annotate_procedural_claim_events.py ^
        --video-dir data/assistant_videos \
        --output-csv outputs/traces/procedural_claim_events.csv

    python scripts/annotate_procedural_claim_events.py ^
        --video-dir data/assistant_videos \
        --output-csv outputs/traces/procedural_claim_events.csv \
        --review
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

CLAIMS = [
    "object_presence",
    "small_gear_inserted",
    "big_gear_inserted",
    "gear_inserted",
    "cover_fully_seated",
    "cover_aligned",
    "correct_small_gear_identity",
    "correct_big_gear_identity",
    "correct_cover_identity",
    "state_validity",
]

OUTCOMES = ["supported", "contradicted", "unresolved"]
STEPS = ["S1", "S2", "S3", "S4", "UNKNOWN"]
ASSEMBLY_SETS = ["A", "B", "UNKNOWN"]


@dataclass
class VideoInfo:
    path: Path
    fps: float
    total_frames: int


@dataclass
class ClaimEvent:
    video: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    assembly_set: str
    step_id: str
    claim_id: str
    outcome: str
    source: str
    note: str = ""
    needs_review: int = 0
    skip: int = 0


def _fps(capture: cv2.VideoCapture) -> float:
    value = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    return value if value > 1e-6 else 30.0


def _video_info(path: Path) -> Optional[VideoInfo]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    fps = _fps(capture)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if total <= 0:
        return None
    return VideoInfo(path=path, fps=fps, total_frames=total)


def _find_videos(video_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in video_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def _seconds_to_frame(seconds: float, fps: float, total_frames: int) -> int:
    return max(0, min(total_frames - 1, int(round(seconds * fps))))


def _stable_range(info: VideoInfo, trim_seconds: float) -> Tuple[int, int]:
    trim = _seconds_to_frame(trim_seconds, info.fps, info.total_frames)
    start = min(trim, max(0, info.total_frames - 1))
    end = max(start, info.total_frames - 1 - trim)
    if end - start < max(10, int(info.fps * 0.5)):
        return 0, info.total_frames - 1
    return start, end


def _event(
    info: VideoInfo,
    *,
    start_frame: int,
    end_frame: int,
    assembly_set: str,
    step_id: str,
    claim_id: str,
    outcome: str,
    source: str,
    note: str = "",
    needs_review: int = 0,
) -> ClaimEvent:
    start_frame = max(0, min(info.total_frames - 1, int(start_frame)))
    end_frame = max(start_frame, min(info.total_frames - 1, int(end_frame)))
    return ClaimEvent(
        video=str(info.path),
        start_frame=start_frame,
        end_frame=end_frame,
        start_time=start_frame / info.fps,
        end_time=end_frame / info.fps,
        assembly_set=assembly_set,
        step_id=step_id,
        claim_id=claim_id,
        outcome=outcome,
        source=source,
        note=note,
        needs_review=needs_review,
    )


def _assembly_from_name(stem: str) -> str:
    upper = stem.upper()
    if re.search(r"(^|_)A($|_)", upper) or upper.startswith(("ASSIST_A", "ERR_A", "VIEW_A", "OD_A")):
        return "A"
    if re.search(r"(^|_)B($|_)", upper) or upper.startswith(("ASSIST_B", "ERR_B", "VIEW_B", "OD_B")):
        return "B"
    return "UNKNOWN"


def _step_for_claim(claim_id: str) -> str:
    if claim_id in {"object_presence", "state_validity"}:
        return "S1"
    if claim_id in {"small_gear_inserted", "gear_inserted", "correct_small_gear_identity"}:
        return "S2" if claim_id != "gear_inserted" else "UNKNOWN"
    if claim_id in {"big_gear_inserted", "correct_big_gear_identity"}:
        return "S3"
    if claim_id in {"cover_fully_seated", "cover_aligned", "correct_cover_identity"}:
        return "S4"
    return "UNKNOWN"


def _claim_from_tokens(stem: str) -> str:
    text = stem.lower()
    if "wrong_smallgear" in text or "wrong_small_gear" in text:
        return "correct_small_gear_identity"
    if "wrong_biggear" in text or "wrong_big_gear" in text:
        return "correct_big_gear_identity"
    if "wrong_cover" in text:
        return "correct_cover_identity"
    if "smallgear_partial_insert" in text or "small_gear_partial_insert" in text:
        return "small_gear_inserted"
    if "biggear_partial_insert" in text or "big_gear_partial_insert" in text:
        return "big_gear_inserted"
    if "cover_gap" in text or "cover_seated" in text:
        return "cover_fully_seated"
    if "smallgear_inserted" in text or "small_gear_inserted" in text:
        return "small_gear_inserted"
    if "biggear_inserted" in text or "big_gear_inserted" in text:
        return "big_gear_inserted"
    if "gear_inserted" in text:
        return "gear_inserted"
    if "cover_aligned" in text:
        return "cover_aligned"
    if "object_blocks_slot" in text:
        return "object_presence"
    if "hand_blocks_gear" in text:
        return "gear_inserted"
    if "hand_blocks_cover" in text or "cover_edge" in text:
        return "cover_fully_seated"
    return "state_validity"


def _infer_special_video(info: VideoInfo, trim_seconds: float, include_od: bool) -> List[ClaimEvent]:
    stem = info.path.stem
    lower = stem.lower()
    assembly = _assembly_from_name(stem)
    start, end = _stable_range(info, trim_seconds)
    events: List[ClaimEvent] = []

    if lower.startswith("od_"):
        if not include_od:
            return []
        claim = "object_presence"
        return [
            _event(
                info,
                start_frame=start,
                end_frame=end,
                assembly_set=assembly,
                step_id="S1",
                claim_id=claim,
                outcome="supported",
                source="filename_rule",
                note="object_detector_video_included_as_presence_claim",
                needs_review=1,
            )
        ]

    if lower.startswith("occ_"):
        claim = _claim_from_tokens(stem)
        ambiguous_gear = claim == "gear_inserted"
        is_empty_slot_s1 = "object_blocks_slot" in lower
        return [
            _event(
                info,
                start_frame=start,
                end_frame=end,
                assembly_set=assembly,
                step_id="S1" if is_empty_slot_s1 else _step_for_claim(claim),
                claim_id=claim,
                outcome="supported" if is_empty_slot_s1 else "unresolved",
                source="filename_rule",
                note=("empty_slot_s1" if is_empty_slot_s1 else "occlusion_or_view_insufficient")
                + ("; review_small_vs_big_gear_step" if ambiguous_gear else ""),
                needs_review=0 if is_empty_slot_s1 else (1 if ambiguous_gear else 0),
            )
        ]

    if lower.startswith("err_"):
        claim = _claim_from_tokens(stem)
        return [
            _event(
                info,
                start_frame=start,
                end_frame=end,
                assembly_set=assembly,
                step_id=_step_for_claim(claim),
                claim_id=claim,
                outcome="contradicted",
                source="filename_rule",
                note="special_error_video",
            )
        ]

    if lower.startswith("view_fail_"):
        claim = _claim_from_tokens(stem)
        ambiguous_gear = claim == "gear_inserted"
        return [
            _event(
                info,
                start_frame=start,
                end_frame=end,
                assembly_set=assembly,
                step_id=_step_for_claim(claim),
                claim_id=claim,
                outcome="unresolved",
                source="filename_rule",
                note="failed_uncertainty_resolution"
                + ("; review_small_vs_big_gear_step" if ambiguous_gear else ""),
                needs_review=1 if ambiguous_gear else 0,
            )
        ]

    if lower.startswith("view_"):
        claim = _claim_from_tokens(stem)
        if "supported" in lower:
            outcome = "supported"
        elif "contradicted" in lower:
            outcome = "contradicted"
        elif "unresolved" in lower:
            outcome = "unresolved"
        else:
            outcome = "unresolved"
        return [
            _event(
                info,
                start_frame=start,
                end_frame=end,
                assembly_set=assembly,
                step_id=_step_for_claim(claim),
                claim_id=claim,
                outcome=outcome,
                source="filename_rule",
                note="active_view_video",
                needs_review=0 if outcome in {"supported", "contradicted", "unresolved"} else 1,
            )
        ]

    return events


def _load_step_segments(video_path: Path, step_anno_dir: Optional[Path]) -> List[Dict[str, object]]:
    candidate_dirs = []
    if step_anno_dir is not None:
        candidate_dirs.append(step_anno_dir)
    candidate_dirs.append(video_path.parent)
    for directory in candidate_dirs:
        path = directory / f"{video_path.stem}_anno.jsonl"
        if not path.exists():
            continue
        rows: List[Dict[str, object]] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                rows.append(json.loads(text))
        return rows
    return []


def _claim_for_step(state: str) -> Tuple[str, str]:
    state = state.upper()
    if state == "S1":
        return "object_presence", "supported"
    if state == "S2":
        return "small_gear_inserted", "supported"
    if state == "S3":
        return "big_gear_inserted", "supported"
    if state == "S4":
        return "cover_fully_seated", "supported"
    if state in {"WRONG", "INVALID"}:
        return "state_validity", "contradicted"
    if state == "UNCERTAIN":
        return "state_validity", "unresolved"
    return "state_validity", "unresolved"


def _infer_assist_video(info: VideoInfo, step_anno_dir: Optional[Path], trim_seconds: float) -> List[ClaimEvent]:
    stem = info.path.stem
    lower = stem.lower()
    if not lower.startswith("assist_"):
        return []
    assembly = _assembly_from_name(stem)
    segments = _load_step_segments(info.path, step_anno_dir)
    events: List[ClaimEvent] = []
    if not segments:
        start, end = _stable_range(info, trim_seconds)
        events.append(
            _event(
                info,
                start_frame=start,
                end_frame=end,
                assembly_set=assembly,
                step_id="UNKNOWN",
                claim_id="state_validity",
                outcome="supported",
                source="filename_rule",
                note="normal_assist_video_without_step_segments",
                needs_review=1,
            )
        )
        return events

    for seg in segments:
        state = str(seg.get("state", "")).upper()
        if state in {"NO_STEP", ""}:
            continue
        claim, outcome = _claim_for_step(state)
        start = int(seg.get("start_frame", 0))
        end = int(seg.get("end_frame", start))
        events.append(
            _event(
                info,
                start_frame=start,
                end_frame=end,
                assembly_set=assembly,
                step_id=state if state.startswith("S") else "UNKNOWN",
                claim_id=claim,
                outcome=outcome,
                source="step_annotation",
                note=f"from_{state}",
                needs_review=0 if state.startswith("S") else 1,
            )
        )
    return events


def build_events(
    video_dir: Path,
    *,
    step_anno_dir: Optional[Path],
    trim_seconds: float,
    include_od: bool,
) -> List[ClaimEvent]:
    events: List[ClaimEvent] = []
    for video in _find_videos(video_dir):
        info = _video_info(video)
        if info is None:
            continue
        inferred = _infer_assist_video(info, step_anno_dir, trim_seconds)
        if not inferred:
            inferred = _infer_special_video(info, trim_seconds, include_od)
        events.extend(inferred)
    return sorted(events, key=lambda item: (item.video, item.start_frame, item.claim_id))


def read_events(path: Path) -> List[ClaimEvent]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        events = []
        for row in reader:
            events.append(
                ClaimEvent(
                    video=row.get("video", ""),
                    start_frame=int(float(row.get("start_frame", 0) or 0)),
                    end_frame=int(float(row.get("end_frame", 0) or 0)),
                    start_time=float(row.get("start_time", 0.0) or 0.0),
                    end_time=float(row.get("end_time", 0.0) or 0.0),
                    assembly_set=row.get("assembly_set", "UNKNOWN") or "UNKNOWN",
                    step_id=row.get("step_id", "UNKNOWN") or "UNKNOWN",
                    claim_id=row.get("claim_id", "state_validity") or "state_validity",
                    outcome=row.get("outcome", "unresolved") or "unresolved",
                    source=row.get("source", "") or "",
                    note=row.get("note", "") or "",
                    needs_review=int(float(row.get("needs_review", 0) or 0)),
                    skip=int(float(row.get("skip", 0) or 0)),
                )
            )
    return events


def write_events(path: Path, events: Sequence[ClaimEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(events[0]).keys()) if events else [field.name for field in ClaimEvent.__dataclass_fields__.values()]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))


def _cycle(value: str, values: Sequence[str], direction: int = 1) -> str:
    if value not in values:
        return values[0]
    return values[(values.index(value) + direction) % len(values)]


def _read_frame(path: Path, frame_index: int):
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def _draw_overlay(frame, lines: Sequence[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 1
    x = 12
    y = 24
    for i, line in enumerate(lines):
        yy = y + i * 24
        cv2.putText(frame, line, (x + 1, yy + 1), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, line, (x, yy), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def review_events(
    events: List[ClaimEvent],
    output_csv: Path,
    *,
    all_events: Optional[List[ClaimEvent]] = None,
    step_frames: int = 10,
    scale: float = 0.8,
) -> None:
    if not events:
        print("[review] no events")
        return
    index = 0
    preview_frame: Dict[int, int] = {}
    window = "INSPECT procedural claim events"
    save_events = all_events if all_events is not None else events
    print(
        "Keys: Space play/pause, s set start, e set end, n/Enter next, p previous, "
        "1 supported, 2 contradicted, 3 unresolved, c claim, t step, g assembly, "
        "a/d step frame, j/l jump 1s, [/] nudge start, ,/. nudge end, "
        "x skip, r review flag, q save+quit"
    )
    playing = False
    while 0 <= index < len(events):
        event = events[index]
        path = Path(event.video)
        frame_id = preview_frame.get(index, (event.start_frame + event.end_frame) // 2)
        info = _video_info(path)
        fps = info.fps if info else 30.0
        total_frames = info.total_frames if info else max(event.end_frame + 1, 1)
        frame_id = max(0, min(total_frames - 1, frame_id))
        frame = _read_frame(path, frame_id)
        if frame is None:
            frame = 255 * cv2.UMat(360, 640, cv2.CV_8UC3).get()
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _draw_overlay(
            frame,
            [
                f"{index + 1}/{len(events)} {path.name}",
                f"frames {event.start_frame}-{event.end_frame} preview={frame_id}",
                f"time {frame_id / max(fps, 1e-6):.2f}s play={'on' if playing else 'off'}",
                f"set={event.assembly_set} step={event.step_id} claim={event.claim_id}",
                f"outcome={event.outcome} skip={event.skip} review={event.needs_review}",
                f"source={event.source} note={event.note}",
            ],
        )
        cv2.imshow(window, frame)
        if playing:
            key = cv2.waitKey(max(1, int(round(1000.0 / max(fps, 1e-6))))) & 0xFF
            if key == 255:
                next_frame = min(total_frames - 1, frame_id + 1)
                preview_frame[index] = next_frame
                if next_frame >= total_frames - 1:
                    playing = False
                continue
        else:
            key = cv2.waitKey(0) & 0xFF
        if key in {ord("q"), 27}:
            write_events(output_csv, save_events)
            break
        if key == 32:
            playing = not playing
            continue
        if key in {ord("n"), 13, 10}:
            event.needs_review = 0
            write_events(output_csv, save_events)
            index += 1
            playing = False
            continue
        if key == ord("p"):
            write_events(output_csv, save_events)
            index = max(0, index - 1)
            playing = False
            continue
        if key == ord("1"):
            event.outcome = "supported"
        elif key == ord("2"):
            event.outcome = "contradicted"
        elif key == ord("3"):
            event.outcome = "unresolved"
        elif key == ord("c"):
            event.claim_id = _cycle(event.claim_id, CLAIMS)
            if event.step_id == "UNKNOWN":
                event.step_id = _step_for_claim(event.claim_id)
        elif key == ord("t"):
            event.step_id = _cycle(event.step_id, STEPS)
        elif key == ord("g"):
            event.assembly_set = _cycle(event.assembly_set, ASSEMBLY_SETS)
        elif key == ord("s"):
            event.start_frame = min(frame_id, event.end_frame)
            event.start_time = event.start_frame / max(fps, 1e-6)
        elif key == ord("e"):
            event.end_frame = max(frame_id, event.start_frame)
            event.end_time = event.end_frame / max(fps, 1e-6)
        elif key == ord("["):
            event.start_frame = max(0, event.start_frame - step_frames)
            event.start_time = event.start_frame / max(fps, 1e-6)
        elif key == ord("]"):
            event.start_frame = min(event.end_frame, event.start_frame + step_frames)
            event.start_time = event.start_frame / max(fps, 1e-6)
        elif key == ord(","):
            event.end_frame = max(event.start_frame, event.end_frame - step_frames)
            event.end_time = event.end_frame / max(fps, 1e-6)
        elif key == ord("."):
            event.end_frame = min(total_frames - 1, event.end_frame + step_frames)
            event.end_time = event.end_frame / fps
        elif key == ord("a"):
            preview_frame[index] = max(0, frame_id - step_frames)
        elif key == ord("d"):
            preview_frame[index] = min(total_frames - 1, frame_id + step_frames)
        elif key == ord("j"):
            preview_frame[index] = max(0, frame_id - int(round(fps)))
        elif key == ord("l"):
            preview_frame[index] = min(total_frames - 1, frame_id + int(round(fps)))
        elif key == ord("x"):
            event.skip = 0 if event.skip else 1
        elif key == ord("r"):
            event.needs_review = 0 if event.needs_review else 1
        write_events(output_csv, save_events)
    cv2.destroyWindow(window)


def _default_timeline_metadata(info: VideoInfo, step_anno_dir: Optional[Path], trim_seconds: float) -> ClaimEvent:
    inferred = _infer_assist_video(info, step_anno_dir, trim_seconds)
    if not inferred:
        inferred = _infer_special_video(info, trim_seconds, include_od=False)
    if inferred:
        return inferred[0]
    start, end = _stable_range(info, trim_seconds)
    return _event(
        info,
        start_frame=start,
        end_frame=end,
        assembly_set=_assembly_from_name(info.path.stem),
        step_id="UNKNOWN",
        claim_id="state_validity",
        outcome="unresolved",
        source="timeline_default",
        note="manual_timeline_default",
        needs_review=1,
    )


def timeline_annotate(
    video_paths: Sequence[Path],
    output_csv: Path,
    events: List[ClaimEvent],
    *,
    step_anno_dir: Optional[Path],
    trim_seconds: float,
    scale: float = 0.8,
) -> None:
    if not video_paths:
        print("[timeline] no videos")
        return

    video_index = 0
    window = "INSPECT full-video claim timeline"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    print(
        "Timeline keys: drag trackbar or Space play/pause; s=set start, e=set end, "
        "m=save now, n=save+next, p=save+previous, u=clear current video, z=new blank, "
        "1/2/3 outcome, c claim, t step, g assembly, a/d step frame, j/l jump 1s, q save+quit"
    )

    while 0 <= video_index < len(video_paths):
        info = _video_info(video_paths[video_index])
        if info is None:
            video_index += 1
            continue

        default = _default_timeline_metadata(info, step_anno_dir, trim_seconds)
        current = 0
        segment_start: Optional[int] = None
        segment_end: Optional[int] = None
        assembly_set = default.assembly_set
        step_id = default.step_id
        claim_id = default.claim_id
        outcome = default.outcome
        note = default.note
        playing = False
        draft_dirty = False
        editing_index: Optional[int] = None
        frame_state = {"pos": 0, "changed": False}

        def _saved_indices() -> List[int]:
            return [i for i, event in enumerate(events) if Path(event.video) == info.path and not event.skip]

        def _load_saved() -> bool:
            nonlocal segment_start, segment_end, assembly_set, step_id, claim_id, outcome
            nonlocal note, current, editing_index, draft_dirty
            indices = _saved_indices()
            if not indices:
                return False
            editing_index = indices[0]
            event = events[editing_index]
            segment_start = int(event.start_frame)
            segment_end = int(event.end_frame)
            assembly_set = event.assembly_set
            step_id = event.step_id
            claim_id = event.claim_id
            outcome = event.outcome
            note = event.note
            current = max(0, min(info.total_frames - 1, segment_start))
            draft_dirty = False
            return True

        def _new_blank() -> None:
            nonlocal segment_start, segment_end, assembly_set, step_id, claim_id, outcome
            nonlocal note, editing_index, draft_dirty
            segment_start = None
            segment_end = None
            assembly_set = default.assembly_set
            step_id = default.step_id
            claim_id = default.claim_id
            outcome = default.outcome
            note = default.note
            editing_index = None
            draft_dirty = False

        def _save_current(action: str) -> bool:
            nonlocal editing_index, draft_dirty
            if segment_start is None or segment_end is None:
                return False
            event = _event(
                info,
                start_frame=segment_start,
                end_frame=segment_end,
                assembly_set=assembly_set,
                step_id=step_id,
                claim_id=claim_id,
                outcome=outcome,
                source="human_timeline",
                note=note,
                needs_review=0,
            )
            kept = [existing for existing in events if Path(existing.video) != info.path]
            kept.append(event)
            events[:] = kept
            editing_index = len(events) - 1
            draft_dirty = False
            write_events(output_csv, events)
            print(f"[{action}] {info.path.name} {segment_start}-{segment_end} {claim_id} {outcome}")
            return True

        _load_saved()

        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        def _on_trackbar(value: int) -> None:
            frame_state["pos"] = int(value)
            frame_state["changed"] = True

        cv2.createTrackbar("frame", window, 0, max(1, info.total_frames - 1), _on_trackbar)
        cv2.setTrackbarPos("frame", window, current)

        while True:
            if frame_state["changed"]:
                current = max(0, min(info.total_frames - 1, int(frame_state["pos"])))
                frame_state["changed"] = False

            frame = _read_frame(info.path, current)
            if frame is None:
                frame = 255 * cv2.UMat(360, 640, cv2.CV_8UC3).get()
            if scale != 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            committed = sum(1 for event in events if Path(event.video) == info.path and not event.skip)
            start_text = "--" if segment_start is None else str(segment_start)
            end_text = "--" if segment_end is None else str(segment_end)
            indices = _saved_indices()
            edit_text = "new"
            if editing_index is not None and editing_index in indices:
                edit_text = "editing saved"
            _draw_overlay(
                frame,
                [
                    f"video {video_index + 1}/{len(video_paths)} {info.path.name}",
                    f"frame {current}/{info.total_frames - 1} time={current / max(info.fps, 1e-6):.2f}s play={'on' if playing else 'off'}",
                    f"segment start={start_text} end={end_text} committed_for_video={committed} {edit_text}",
                    f"set={assembly_set} step={step_id} claim={claim_id} outcome={outcome}",
                    "n/p/q auto-save if start and end are set; one row per video",
                    "s=start e=end m=save z=new n=next p=prev Space=play q=quit",
                ],
            )
            cv2.imshow(window, frame)

            if playing:
                key = cv2.waitKey(max(1, int(round(1000.0 / max(info.fps, 1e-6))))) & 0xFF
                if key == 255:
                    current = min(info.total_frames - 1, current + 1)
                    cv2.setTrackbarPos("frame", window, current)
                    if current >= info.total_frames - 1:
                        playing = False
                    continue
            else:
                key = cv2.waitKey(0) & 0xFF

            if key in {ord("q"), 27}:
                _save_current("auto-save")
                write_events(output_csv, events)
                cv2.destroyWindow(window)
                return
            if key == 32:
                playing = not playing
                continue
            if key == ord("n"):
                _save_current("auto-save")
                write_events(output_csv, events)
                video_index += 1
                break
            if key == ord("p"):
                _save_current("auto-save")
                write_events(output_csv, events)
                video_index = max(0, video_index - 1)
                break
            if key == ord("s"):
                segment_start = current
                if segment_end is not None and segment_end < segment_start:
                    segment_end = segment_start
                draft_dirty = True
            elif key == ord("e"):
                segment_end = current
                if segment_start is not None and segment_start > segment_end:
                    segment_start = segment_end
                draft_dirty = True
            elif key == ord("m"):
                if not _save_current("save"):
                    print(f"[skip commit] set both start and end first: {info.path.name}")
                    playing = False
                continue
            elif key == ord("u"):
                before = len(events)
                events[:] = [event for event in events if Path(event.video) != info.path]
                print(f"[clear] {info.path.name} removed {before - len(events)} saved row(s)")
                write_events(output_csv, events)
                _new_blank()
            elif key == ord("z"):
                _new_blank()
            elif key == ord("1"):
                outcome = "supported"
                draft_dirty = True
            elif key == ord("2"):
                outcome = "contradicted"
                draft_dirty = True
            elif key == ord("3"):
                outcome = "unresolved"
                draft_dirty = True
            elif key == ord("c"):
                claim_id = _cycle(claim_id, CLAIMS)
                if step_id == "UNKNOWN":
                    step_id = _step_for_claim(claim_id)
                draft_dirty = True
            elif key == ord("t"):
                step_id = _cycle(step_id, STEPS)
                draft_dirty = True
            elif key == ord("g"):
                assembly_set = _cycle(assembly_set, ASSEMBLY_SETS)
                draft_dirty = True
            elif key == ord("a"):
                current = max(0, current - 10)
                cv2.setTrackbarPos("frame", window, current)
            elif key == ord("d"):
                current = min(info.total_frames - 1, current + 10)
                cv2.setTrackbarPos("frame", window, current)
            elif key == ord("j"):
                current = max(0, current - int(round(info.fps)))
                cv2.setTrackbarPos("frame", window, current)
            elif key == ord("l"):
                current = min(info.total_frames - 1, current + int(round(info.fps)))
                cv2.setTrackbarPos("frame", window, current)

    write_events(output_csv, events)
    cv2.destroyWindow(window)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory containing assistant/view/error videos.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output procedural claim event CSV.")
    parser.add_argument("--step-anno-dir", type=Path, default=None, help="Optional directory with *_anno.jsonl step segments.")
    parser.add_argument("--trim-seconds", type=float, default=0.75, help="Trim this many seconds from both ends for filename-rule events.")
    parser.add_argument("--include-od", action="store_true", help="Also convert OD_* videos into object_presence claims.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate output CSV even if it already exists.")
    parser.add_argument("--review", action="store_true", help="Open a lightweight OpenCV review UI after generating/loading CSV.")
    parser.add_argument("--review-all", action="store_true", help="Review all events, not only rows with needs_review=1.")
    parser.add_argument("--timeline", action="store_true", help="Annotate full videos with a draggable frame trackbar and manual segment commits.")
    parser.add_argument("--scale", type=float, default=0.8, help="Review display scale.")
    args = parser.parse_args()

    if args.timeline:
        events = [] if args.overwrite else read_events(args.output_csv)
        timeline_annotate(
            _find_videos(args.video_dir),
            args.output_csv,
            events,
            step_anno_dir=args.step_anno_dir,
            trim_seconds=args.trim_seconds,
            scale=args.scale,
        )
        return

    if args.output_csv.exists() and not args.overwrite:
        events = read_events(args.output_csv)
        print(f"[load] {len(events)} events from {args.output_csv}")
    else:
        events = build_events(
            args.video_dir,
            step_anno_dir=args.step_anno_dir,
            trim_seconds=args.trim_seconds,
            include_od=args.include_od,
        )
        write_events(args.output_csv, events)
        print(f"[write] {len(events)} events -> {args.output_csv}")

    if args.review:
        if args.review_all:
            review_events(events, args.output_csv, scale=args.scale)
        else:
            review_subset = [event for event in events if event.needs_review and not event.skip]
            if review_subset:
                review_events(review_subset, args.output_csv, all_events=events, scale=args.scale)
            else:
                print("[review] no rows marked needs_review=1; use --review-all to inspect all rows.")


if __name__ == "__main__":
    main()
