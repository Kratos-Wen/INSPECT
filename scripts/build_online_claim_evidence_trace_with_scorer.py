"""Build online claim/evidence traces with the assistance evidence scorer.

This is the postcondition-aware variant of
``build_online_claim_evidence_trace_from_run.py``. It does not create
before/after labels from interval endpoints. It emits supported/contradicted
samples only at online feedback frames; other frames remain unresolved.
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

from inspect_system.active_view.ontology import infer_role_from_fields
from inspect_system.evidence_scorer import (
    PrototypeEvidenceScorer,
    extract_relation_features,
    normalize_claim,
)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield item


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def video_key(value: object) -> str:
    return Path(str(value or "")).name


def group_timeline(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in read_csv(path):
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
    for rows in grouped.values():
        rows.sort(key=lambda item: (int(item["start_frame_i"]), int(item["end_frame_i"])))
    return grouped


def timeline_at(rows: List[Dict[str, Any]], frame: int) -> Optional[Dict[str, Any]]:
    for row in rows:
        if int(row["start_frame_i"]) <= frame <= int(row["end_frame_i"]):
            return row
    return None


def feedback_by_frame(path: Path) -> Dict[int, Dict[str, Any]]:
    indexed: Dict[int, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        try:
            frame = int(row.get("frame_index", -1))
        except Exception:
            continue
        indexed[frame] = row
    return indexed


def state_id(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"STEP1", "STEP2", "STEP3", "STEP4"}:
        return "S" + text[-1]
    return text


def claim_for_step(step: object) -> str:
    step_norm = state_id(step)
    return {
        "S1": "object_presence",
        "S2": "small_gear_inserted",
        "S3": "big_gear_inserted",
        "S4": "cover_fully_seated",
    }.get(step_norm, "state_validity" if step_norm else "")


def evidence_role(claim_id: str) -> str:
    return infer_role_from_fields(claim_id=claim_id)


def make_sample(
    *,
    record: Mapping[str, Any],
    feedback: Optional[Mapping[str, Any]],
    timeline_row: Optional[Mapping[str, Any]],
    scorer: PrototypeEvidenceScorer,
    video: str,
) -> Dict[str, Any]:
    frame = int(record.get("frame_index", 0) or 0)
    fused_step = state_id(record.get("fused_step"))
    step = state_id(timeline_row.get("step_id")) if timeline_row else fused_step
    claim = normalize_claim(timeline_row.get("claim_id") if timeline_row else claim_for_step(step), step)
    product = str(timeline_row.get("assembly_set", "")) if timeline_row else ""
    outcome = "unresolved"
    source = "auto_unresolved"
    if feedback and timeline_row:
        timeline_outcome = str(timeline_row.get("outcome", "")).strip().lower()
        if timeline_outcome in {"supported", "contradicted"}:
            outcome = timeline_outcome
            source = "feedback_anchor"

    detections = list(record.get("fused_detections") or record.get("raw_detections") or [])
    features = extract_relation_features(
        detections,
        claim_id=claim,
        step_id=step,
        product=product,
        image_shape=(1080, 1920),
    )
    scores = scorer.score_features(features, claim_id=claim, step_id=step)
    role = evidence_role(claim)
    if outcome == "contradicted":
        role_score = float(scores.get("contradiction_score", 0.0))
    elif outcome == "supported":
        role_score = float(scores.get("support_score", 0.0))
    else:
        role_score = max(float(scores.get("support_score", 0.0)), float(scores.get("visibility_score", 0.0)) * 0.5)
    role_scores = {
        role: role_score,
        "claim_disambiguation_view": max(role_score, float(scores.get("visibility_score", 0.0)) * 0.5),
    }
    return {
        "sample_id": f"online_scorer_f{frame:06d}_{claim}_{outcome}",
        "event_id": f"online_scorer_frame_{frame:06d}",
        "video": video,
        "frame": frame,
        "step_id": step,
        "claim_id": claim,
        "world_outcome": outcome,
        "source": "online_assistant_run_frame_with_assistance_scorer",
        "note": source,
        "role_scores": role_scores,
        "evidence_scores": dict(role_scores),
        "claim_evidence_score": role_score,
        "online_trace": {
            "fused_step": fused_step,
            "fused_conf": float(record.get("fused_conf", 0.0) or 0.0),
            "ensemble_margin": float(record.get("ensemble_margin", 0.0) or 0.0),
            "stable": bool(record.get("stable", False)),
            "has_visual_evidence": bool(record.get("has_visual_evidence", False)),
            "review_action": record.get("review_action", ""),
            "review_reason": record.get("review_reason", ""),
        },
        "metadata": {
            "uses_human_segment_boundary": False,
            "uses_robot_view_training": False,
            "online_feedback_anchor": bool(feedback),
            "not_relative_action": True,
            "not_view_transition": True,
            "assistance_support_score": scores.get("support_score", 0.0),
            "assistance_contradiction_score": scores.get("contradiction_score", 0.0),
            "assistance_visibility_score": scores.get("visibility_score", 0.0),
            "assembly_set": product,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timeline-csv", type=Path, required=True)
    parser.add_argument("--evidence-scorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, default=None)
    args = parser.parse_args()

    scorer = PrototypeEvidenceScorer.load(args.evidence_scorer)
    meta = read_json(args.run_dir / "meta.json")
    video = str(meta.get("video_path", "")) or str(args.run_dir)
    timeline = group_timeline(args.timeline_csv).get(video_key(video), [])
    feedback_index = feedback_by_frame(args.run_dir / "feedback.jsonl")
    samples: List[Dict[str, Any]] = []
    for record in iter_jsonl(args.run_dir / "iterations.jsonl"):
        frame = int(record.get("frame_index", 0) or 0)
        sample = make_sample(
            record=record,
            feedback=feedback_index.get(frame),
            timeline_row=timeline_at(timeline, frame),
            scorer=scorer,
            video=video,
        )
        samples.append(sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "run_dir": str(args.run_dir),
        "video": video,
        "samples": len(samples),
        "feedback_frames": len(feedback_index),
        "output": str(args.output),
        "by_world_outcome": {},
        "by_claim_id": {},
    }
    for sample in samples:
        for key, field in (("by_world_outcome", "world_outcome"), ("by_claim_id", "claim_id")):
            value = str(sample.get(field, "")) or "UNKNOWN"
            report[key][value] = int(report[key].get(value, 0)) + 1
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
