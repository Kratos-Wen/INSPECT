"""Evaluate a rule proposal expert from frozen online replay evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspect_assist.__main__ import _load_runtime_package

_load_runtime_package()

from inspect_runtime.components.kb import KnowledgeBase
from inspect_runtime.components.rules import RuleBasedStepExpert
from inspect_runtime.components.track_evidence import TrackEvidenceAggregator
from inspect_runtime.core_types import Detection


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _detection(item: dict[str, Any]) -> Detection:
    return Detection(
        name=str(item["name"]),
        xyxy=tuple(float(value) for value in item["xyxy"]),
        confidence=float(item["confidence"]),
        meta=dict(item.get("meta") or {}),
    )


def _topk(scores: dict[str, float], k: int) -> list[str]:
    return sorted(scores, key=lambda key: (-float(scores[key]), key))[:k]


def _metrics(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    total = len(records)
    top1 = sum(row[field]["top1"] == row["target"] for row in records)
    top2 = sum(row["target"] in row[field]["top2"] for row in records)
    return {
        "frames": total,
        "top1": top1 / max(1, total),
        "top2": top2 / max(1, total),
        "by_step": {
            step: _metrics(group, field)
            for step, group in _groups(records, "target").items()
            if len(group) < total
        },
        "by_outcome": {
            outcome: _metrics(group, field)
            for outcome, group in _groups(records, "outcome").items()
            if len(group) < total
        },
    }


def _groups(records: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        output[str(row[key])].append(row)
    return dict(output)


def _latest_runs(root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for run_dir in root.iterdir():
        if not run_dir.is_dir() or not (run_dir / "iterations.jsonl").exists():
            continue
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        video = Path(str(meta.get("video_path", ""))).name
        if not video:
            continue
        previous = output.get(video)
        if previous is None or run_dir.stat().st_mtime > previous.stat().st_mtime:
            output[video] = run_dir
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--timeline-csv", type=Path, required=True)
    parser.add_argument("--kb", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    timelines: dict[str, dict[str, Any]] = {}
    with args.timeline_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("skip", "0")).strip() == "1":
                continue
            timelines[Path(str(row["video"])).name] = row

    runs = _latest_runs(args.replay_root)
    expert = RuleBasedStepExpert(
        KnowledgeBase.from_path(str(args.kb)),
        ["S1", "S2", "S3", "S4"],
    )
    stable_filter = TrackEvidenceAggregator()
    evaluated: list[dict[str, Any]] = []
    missing_runs = []

    for video, timeline in sorted(timelines.items()):
        if str(timeline["step_id"]).upper() not in {"S1", "S2", "S3", "S4"}:
            continue
        run_dir = runs.get(video)
        if run_dir is None:
            missing_runs.append(video)
            continue
        start = int(float(timeline["start_frame"]))
        end = int(float(timeline["end_frame"]))
        for row in _rows(run_dir / "iterations.jsonl"):
            frame = int(row["frame_index"])
            if frame < start or frame > end:
                continue
            fused = [_detection(item) for item in row.get("fused_detections", [])]
            proposal_detections = stable_filter.stable_detections(fused) or fused
            relations = (
                row.get("evidence_token", {}).get("relation_facts")
                or row.get("scene_graph_relations")
                or []
            )
            prediction = expert.predict(
                {"detections": proposal_detections, "relations": relations}
            )
            old_scores = {
                str(key): float(value)
                for key, value in (row.get("state_scores") or {}).items()
            }
            target = str(timeline["step_id"]).upper()
            evaluated.append(
                {
                    "video": video,
                    "frame": frame,
                    "target": target,
                    "outcome": str(timeline["outcome"]).lower(),
                    "old": {
                        "top1": str(row.get("state_step", "")),
                        "top2": _topk(old_scores, 2),
                    },
                    "current": {
                        "top1": prediction.step_id,
                        "top2": _topk(prediction.scores, 2),
                    },
                }
            )

    report = {
        "protocol": {
            "source": "frozen online replay evidence",
            "timeline_used_for": "scoring only",
            "future_frames_used": False,
            "feedback_used": False,
            "ground_truth_boxes_used": False,
            "videos": len({row["video"] for row in evaluated}),
            "frames": len(evaluated),
            "outcomes": dict(Counter(row["outcome"] for row in evaluated)),
            "missing_runs": missing_runs,
        },
        "old_replay_rule": _metrics(evaluated, "old"),
        "current_rule": _metrics(evaluated, "current"),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
