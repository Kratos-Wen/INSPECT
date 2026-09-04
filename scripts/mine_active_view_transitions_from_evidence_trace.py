"""Mine view-learning transitions from frame-level evidence traces.

Input records should come from ``build_claim_frame_samples_from_timeline.py``
after a detector/verifier has added per-frame evidence scores. This script does
not use interval start/end as a before/after pair. It searches within each
video/claim/outcome stream for ordered frame pairs where claim-relevant evidence
changes from weak to decisive while the annotated world outcome is fixed.

Expected input fields:
  - video, frame, claim_id, world_outcome
  - one of: role_scores, evidence_scores, evidence, z

Output records are assistant-event-like JSONL records that can be passed to
``mine_inspect_active_view_transitions.py`` for relative-action inference and
Causal Transfer Gate filtering.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def normalize_key(value: object) -> str:
    text = str(value or "").lower().strip().replace("-", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text).strip("_")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def default_role_for_claim(claim_id: str) -> str:
    key = normalize_key(claim_id)
    if "cover" in key:
        return "gap_visibility_view"
    if "identity" in key:
        return "identity_disambiguation_view"
    if "insert" in key or "gear" in key:
        return "insertion_verification_view"
    if "presence" in key:
        return "object_presence_view"
    return "claim_disambiguation_view"


def role_candidates(records: Sequence[Mapping[str, Any]], explicit_roles: Sequence[str]) -> List[str]:
    if explicit_roles:
        return [str(role) for role in explicit_roles]
    roles = set()
    for record in records:
        for field in ("role_scores", "evidence_scores", "evidence", "z"):
            value = record.get(field)
            if isinstance(value, Mapping):
                roles.update(str(key) for key in value.keys())
        role = record.get("evidence_view_type") or record.get("evidence_role")
        if role:
            roles.add(str(role))
    if roles:
        return sorted(roles)
    return sorted({default_role_for_claim(str(record.get("claim_id", ""))) for record in records})


def score_for_role(record: Mapping[str, Any], role: str) -> float | None:
    for field in ("role_scores", "evidence_scores", "evidence", "z"):
        value = record.get(field)
        if isinstance(value, Mapping) and role in value:
            try:
                return float(value[role])
            except Exception:
                return None
    scalar_fields = (
        f"{role}_score",
        "claim_evidence_score",
        "evidence_score",
        "support_score",
        "score",
    )
    for field in scalar_fields:
        if record.get(field) not in (None, ""):
            try:
                return float(record[field])
            except Exception:
                return None
    return None


def group_records(records: Iterable[Dict[str, Any]]) -> Dict[tuple[str, str, str], List[Dict[str, Any]]]:
    grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        outcome = normalize_key(record.get("world_outcome") or record.get("outcome"))
        if outcome not in {"supported", "contradicted"}:
            continue
        key = (
            str(record.get("video", "")),
            str(record.get("claim_id", "")),
            outcome,
        )
        grouped[key].append(record)
    for items in grouped.values():
        items.sort(key=lambda item: int(item.get("frame", 0)))
    return grouped


def make_event(
    group_key: tuple[str, str, str],
    role: str,
    low: Mapping[str, Any],
    high: Mapping[str, Any],
    low_score: float,
    high_score: float,
    event_index: int,
) -> Dict[str, Any]:
    video, claim_id, outcome = group_key
    event_id = f"{Path(video).stem}_{normalize_key(claim_id)}_{role}_{event_index:05d}"
    is_supported = outcome == "supported"
    return {
        "event_id": event_id,
        "video": video,
        "claim_id": claim_id,
        "claim_type": claim_id,
        "before_frame": int(low["frame"]),
        "after_frame": int(high["frame"]),
        "reason": "auto_low_to_high_evidence",
        "feedback": "accepted" if is_supported else "rejected",
        "outcome": "supported" if is_supported else "contradicted",
        "missing_evidence": [role],
        "revealed_evidence": [role] if is_supported else [],
        "contradictory_evidence": [] if is_supported else [role],
        "evidence_view_type": role,
        "metadata": {
            "source": "frame_evidence_trace_low_to_high",
            "not_from_interval_endpoints": True,
            "uses_human_segment_boundary": True,
            "uses_robot_view_training": False,
            "world_outcome": outcome,
            "before_score": low_score,
            "after_score": high_score,
            "score_gain": high_score - low_score,
            "before_sample_id": low.get("sample_id", ""),
            "after_sample_id": high.get("sample_id", ""),
        },
    }


def mine_transitions(
    records: List[Dict[str, Any]],
    roles: Sequence[str],
    low_threshold: float,
    high_threshold: float,
    min_gain: float,
    min_frame_gap: int,
    max_frame_gap: int,
    max_pairs_per_group: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "groups": 0,
        "mined_events": 0,
        "skipped_no_scores": 0,
        "by_role": {},
        "by_outcome": {},
    }
    event_index = 0
    grouped = group_records(records)
    report["groups"] = len(grouped)
    for group_key, items in grouped.items():
        group_roles = role_candidates(items, roles)
        group_pairs = 0
        for role in group_roles:
            scored = []
            for item in items:
                score = score_for_role(item, role)
                if score is not None:
                    scored.append((item, score))
            if len(scored) < 2:
                report["skipped_no_scores"] += 1
                continue
            for low_index, (low, low_score) in enumerate(scored):
                if low_score > low_threshold:
                    continue
                best_high = None
                for high, high_score in scored[low_index + 1 :]:
                    gap = int(high["frame"]) - int(low["frame"])
                    if gap < min_frame_gap:
                        continue
                    if max_frame_gap > 0 and gap > max_frame_gap:
                        break
                    gain = high_score - low_score
                    if high_score >= high_threshold and gain >= min_gain:
                        candidate = (high, high_score, gain)
                        if best_high is None or candidate[2] > best_high[2]:
                            best_high = candidate
                if best_high is None:
                    continue
                event_index += 1
                high, high_score, _ = best_high
                events.append(make_event(group_key, role, low, high, low_score, high_score, event_index))
                report["by_role"][role] = int(report["by_role"].get(role, 0)) + 1
                outcome = group_key[2]
                report["by_outcome"][outcome] = int(report["by_outcome"].get(outcome, 0)) + 1
                group_pairs += 1
                if max_pairs_per_group > 0 and group_pairs >= max_pairs_per_group:
                    break
            if max_pairs_per_group > 0 and group_pairs >= max_pairs_per_group:
                break
    report["mined_events"] = len(events)
    return events, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--role", dest="roles", action="append", default=[], help="Evidence role to mine; repeatable.")
    parser.add_argument("--low-threshold", type=float, default=0.35)
    parser.add_argument("--high-threshold", type=float, default=0.70)
    parser.add_argument("--min-gain", type=float, default=0.30)
    parser.add_argument("--min-frame-gap", type=int, default=3)
    parser.add_argument("--max-frame-gap", type=int, default=0, help="0 means no maximum.")
    parser.add_argument("--max-pairs-per-group", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.frame_samples)
    events, report = mine_transitions(
        records=records,
        roles=args.roles,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        min_gain=args.min_gain,
        min_frame_gap=args.min_frame_gap,
        max_frame_gap=args.max_frame_gap,
        max_pairs_per_group=args.max_pairs_per_group,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    report["output"] = str(args.output)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
