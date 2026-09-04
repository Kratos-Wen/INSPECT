"""Train the INSPECT assistance-calibrated evidence scorer.

Training uses only online Assistant run logs plus sparse user feedback. In the
current simulation, GT rows stand in for the user's online accept/reject
feedback. Robot six-view images and robot view utility labels are not used.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspect_system.evidence_scorer import extract_relation_features, normalize_claim, PrototypeEvidenceScorer


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def video_key(path: object) -> str:
    return Path(str(path or "")).name


def group_timeline(rows: Iterable[Mapping[str, str]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("skip", "")).strip() in {"1", "true", "yes"}:
            continue
        try:
            start = int(float(row.get("start_frame") or 0))
            end = int(float(row.get("end_frame") or start))
        except ValueError:
            continue
        item = dict(row)
        item["start_frame_i"] = start
        item["end_frame_i"] = end
        grouped.setdefault(video_key(row.get("video")), []).append(item)
    for values in grouped.values():
        values.sort(key=lambda item: (int(item["start_frame_i"]), int(item["end_frame_i"])))
    return grouped


def timeline_at(timeline: List[Dict[str, Any]], frame: int) -> Optional[Dict[str, Any]]:
    for row in timeline:
        if int(row["start_frame_i"]) <= frame <= int(row["end_frame_i"]):
            return row
    return None


def state_id(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"STEP1", "STEP2", "STEP3", "STEP4"}:
        return "S" + text[-1]
    return text


def label_from_feedback(feedback_row: Mapping[str, Any], timeline_row: Mapping[str, Any]) -> str:
    outcome = str(timeline_row.get("outcome", "")).strip().lower()
    feedback = dict(feedback_row.get("feedback", {}))
    accepted = bool(feedback.get("accepted", False))
    if outcome == "supported" and accepted:
        return "support"
    if outcome == "contradicted":
        return "contradiction"
    return ""


def features_from_iteration(record: Mapping[str, Any], timeline_row: Mapping[str, Any]) -> Dict[str, float]:
    detections = list(record.get("fused_detections") or record.get("raw_detections") or [])
    return extract_relation_features(
        detections,
        claim_id=timeline_row.get("claim_id", ""),
        step_id=timeline_row.get("step_id", ""),
        product=timeline_row.get("assembly_set", ""),
        image_shape=(1080, 1920),
    )


def build_samples(
    *,
    summary_csv: Path,
    timeline_csv: Path,
    window: int,
    min_conf: float,
) -> List[Dict[str, Any]]:
    summary_rows = read_csv(summary_csv)
    timelines = group_timeline(read_csv(timeline_csv))
    samples: List[Dict[str, Any]] = []
    for row in summary_rows:
        run_dir_text = str(row.get("run_dir", "")).strip()
        if not run_dir_text:
            continue
        run_dir = Path(run_dir_text)
        iterations = list(iter_jsonl(run_dir / "iterations.jsonl"))
        feedback_rows = list(iter_jsonl(run_dir / "feedback.jsonl"))
        if not iterations or not feedback_rows:
            continue
        by_frame = {int(item.get("frame_index", item.get("frame", -1))): item for item in iterations}
        sorted_frames = sorted(by_frame)
        timeline = timelines.get(video_key(row.get("video")), [])
        for feedback in feedback_rows:
            frame = int(feedback.get("frame_index", -1))
            trow = timeline_at(timeline, frame)
            if not trow:
                continue
            label = label_from_feedback(feedback, trow)
            if label not in {"support", "contradiction"}:
                continue
            claim_id = normalize_claim(trow.get("claim_id"), trow.get("step_id"))
            window_frames = [value for value in sorted_frames if value <= frame][-max(1, int(window)) :]
            for offset, sample_frame in enumerate(window_frames):
                record = by_frame[sample_frame]
                fused_conf = float(record.get("fused_conf", 0.0) or 0.0)
                # Keep low-confidence frames too, but down-weight them through metadata.
                feats = features_from_iteration(record, trow)
                if max(feats.get("target_conf", 0.0), feats.get("housing_role_conf", 0.0), feats.get("top_conf", 0.0)) < min_conf:
                    continue
                recency = (offset + 1) / max(1, len(window_frames))
                samples.append(
                    {
                        "video": row.get("video", ""),
                        "run_dir": str(run_dir),
                        "feedback_frame": frame,
                        "sample_frame": sample_frame,
                        "claim_id": claim_id,
                        "step_id": state_id(trow.get("step_id")),
                        "assembly_set": trow.get("assembly_set", ""),
                        "outcome": trow.get("outcome", ""),
                        "label": label,
                        "features": feats,
                        "sample_weight": max(0.25, recency) * max(0.5, min(1.0, fused_conf + 0.5)),
                        "source": "online_assistance_feedback_window",
                    }
                )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-summary-csv", type=Path, required=True)
    parser.add_argument("--timeline-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-jsonl", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--min-conf", type=float, default=0.02)
    args = parser.parse_args()

    samples = build_samples(
        summary_csv=args.online_summary_csv,
        timeline_csv=args.timeline_csv,
        window=args.window,
        min_conf=args.min_conf,
    )
    model = PrototypeEvidenceScorer.train(samples)
    model.metadata.update(
        {
            "online_summary_csv": str(args.online_summary_csv),
            "timeline_csv": str(args.timeline_csv),
            "window": int(args.window),
            "min_conf": float(args.min_conf),
            "training_source": "online_assistance_feedback_only",
            "uses_robot_view_training": False,
        }
    )
    model.save(args.output)
    if args.samples_jsonl:
        write_jsonl(args.samples_jsonl, samples)
    by_claim_label: Dict[str, Dict[str, int]] = {}
    for sample in samples:
        claim = str(sample.get("claim_id", ""))
        label = str(sample.get("label", ""))
        by_claim_label.setdefault(claim, {}).setdefault(label, 0)
        by_claim_label[claim][label] += 1
    report = {
        "output": str(args.output),
        "samples": len(samples),
        "by_claim_label": by_claim_label,
        "model_counts": model.counts,
        "metadata": model.metadata,
    }
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
