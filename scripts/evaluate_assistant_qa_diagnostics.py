"""Evaluate INSPECT assistant single-turn and multi-turn grounded QA.

The script is intentionally diagnostic rather than a training entry point. It
compares lightweight assistant variants on the same recorded snapshots:

- step_only: step/KB context without visual evidence or memory.
- no_memory: visual evidence and KB, but no procedural/history/feedback memory.
- full: the current INSPECT assistant.
- stateless/dialogue/full-feedback variants for multi-turn episodes.

Optional direct LLM baselines can be added later without changing the output
schema; the default run is fast and deterministic.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspect_assist import load_runtime_package

load_runtime_package()

from inspect_runtime.assistant import AssistantSnapshot, ContextualAssistant, GroundedLLMResponder
from inspect_runtime.assistant.types import FeedbackTimelineEvent, MemoryMatchSummary, StepTimelineEvent
from inspect_runtime.components.kb import KnowledgeBase


def default_qwen3_model_path() -> str:
    return os.environ.get("QWEN3_MODEL_PATH", str(ROOT / "models" / "qwen3-4b"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_gt_text(text: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for item in str(text or "").split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip().lower()
        value = value.strip().lower()
        if key:
            parsed[key] = value
    return parsed


def load_gt_rows(gt_dir: Path, video_path: str) -> List[Dict[str, Any]]:
    if not gt_dir:
        return []
    gt_path = gt_dir / f"{Path(video_path).stem}_gt.csv"
    if not gt_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for row in read_csv(gt_path):
        meta = parse_gt_text(row.get("text", ""))
        rows.append(
            {
                **row,
                "_start": int(float(row.get("start_frame") or 0)),
                "_end": int(float(row.get("end_frame") or 0)),
                "_state": str(row.get("state") or meta.get("step") or "").strip().upper(),
                "_claim": meta.get("claim", ""),
                "_outcome": meta.get("outcome", ""),
            }
        )
    return rows


def attach_gt_metadata(records: List[Dict[str, Any]], gt_dir: Path) -> None:
    cache: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        video = str(record.get("_video") or "")
        if not video:
            continue
        if video not in cache:
            cache[video] = load_gt_rows(gt_dir, video)
        frame = int(record.get("frame_index", 0) or 0)
        match = next((row for row in cache[video] if row["_start"] <= frame <= row["_end"]), None)
        if not match:
            continue
        record["_gt_step"] = match.get("_state", "")
        record["_gt_claim"] = match.get("_claim", "")
        record["_gt_outcome"] = match.get("_outcome", "")
        record["_gt_interval"] = f"{match.get('_start','')}-{match.get('_end','')}"


def video_family(video_path: str) -> str:
    stem = Path(video_path).stem.lower()
    if "view_a" in stem:
        return "A"
    if "view_b" in stem:
        return "B"
    return ""


def family_parts(family: str) -> Dict[str, str]:
    if family == "B":
        return {
            "housing": "type_5_gearbox_housing",
            "cover": "type_5_gearbox_cover",
            "small_gear": "type_3_gear",
            "big_gear": "type_8_gear",
        }
    return {
        "housing": "type_6_gearbox_housing",
        "cover": "type_6_gearbox_cover",
        "small_gear": "type_7_gear",
        "big_gear": "type_2_gear",
    }


def gt_target_components(record: Dict[str, Any]) -> List[str]:
    video = str(record.get("_video") or "")
    stem = Path(video).stem.lower()
    parts = family_parts(video_family(video))
    claim = str(record.get("_gt_claim") or "").lower()
    targets: List[str] = []
    if "cover" in claim or "cover_seated" in stem:
        targets.extend([parts["cover"], parts["housing"]])
    elif "small_gear" in claim or "smallgear" in stem:
        targets.extend([parts["small_gear"], parts["housing"]])
    elif "big_gear" in claim or "biggear" in stem:
        targets.extend([parts["big_gear"], parts["housing"]])
    elif "identity" in stem:
        for name in ["type_2_gear", "type_3_gear", "type_7_gear", "type_8_gear"]:
            if name.replace("_gear", "") in stem or name in stem:
                targets.append(name)
        if "type2" in stem:
            targets.append("type_2_gear")
        if "type3" in stem:
            targets.append("type_3_gear")
        if "type7" in stem:
            targets.append("type_7_gear")
        if "type8" in stem:
            targets.append("type_8_gear")
    seen: List[str] = []
    for target in targets:
        if target and target not in seen:
            seen.append(target)
    return seen


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
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


def choose_records(summary_csv: Path, video_hints: List[str], snapshots_per_video: int) -> List[Dict[str, Any]]:
    runs = read_csv(summary_csv)
    selected: List[Dict[str, Any]] = []
    for hint in video_hints:
        hint_l = hint.lower()
        match = next((row for row in runs if hint_l in str(row.get("video", "")).lower()), None)
        if not match:
            continue
        records = list(iter_jsonl(Path(match["run_dir"]) / "iterations.jsonl"))
        if not records:
            continue
        if snapshots_per_video <= 1:
            indices = [len(records) // 2]
        else:
            indices = [
                round(k * (len(records) - 1) / max(1, snapshots_per_video - 1))
                for k in range(snapshots_per_video)
            ]
        for idx in indices:
            item = dict(records[int(idx)])
            item["_video"] = match.get("video", "")
            item["_run_dir"] = match.get("run_dir", "")
            selected.append(item)
    return selected


def choose_records_all(summary_csv: Path, snapshots_per_video: int) -> List[Dict[str, Any]]:
    runs = read_csv(summary_csv)
    selected: List[Dict[str, Any]] = []
    for match in runs:
        if str(match.get("returncode", "")).strip() not in {"", "0"}:
            continue
        run_dir = str(match.get("run_dir") or "").strip()
        if not run_dir:
            continue
        records = list(iter_jsonl(Path(run_dir) / "iterations.jsonl"))
        if not records:
            continue
        if snapshots_per_video <= 1:
            indices = [len(records) // 2]
        else:
            indices = [
                round(k * (len(records) - 1) / max(1, snapshots_per_video - 1))
                for k in range(snapshots_per_video)
            ]
        for idx in indices:
            item = dict(records[int(idx)])
            item["_video"] = match.get("video", "")
            item["_run_dir"] = run_dir
            selected.append(item)
    return selected


def relation_visible(record: Dict[str, Any]) -> bool:
    rels = record.get("scene_relations") or []
    objects = record.get("scene_objects") or record.get("fused_detections") or []
    return len(objects) >= 2 and bool(rels)


def object_counts(names: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name in names:
        key = str(name or "").strip().lower()
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts


def relation_fact(rel: Dict[str, Any]) -> Dict[str, str]:
    return {
        "subject": str(rel.get("subject", rel.get("a", rel.get("source", "")))).strip().lower(),
        "predicate": str(rel.get("predicate", rel.get("relation", rel.get("type", "")))).strip().lower(),
        "object": str(rel.get("object", rel.get("b", rel.get("target", "")))).strip().lower(),
    }


def snapshot_from_record(record: Dict[str, Any]) -> AssistantSnapshot:
    visible = [
        str(obj.get("name", "")).strip().lower()
        for obj in (record.get("scene_objects") or record.get("fused_detections") or [])
        if str(obj.get("name", "")).strip()
    ]
    relevant = [
        str(obj.get("name", "")).strip().lower()
        for obj in (record.get("relevant_detections") or [])
        if str(obj.get("name", "")).strip()
    ]
    rels = [relation_fact(rel) for rel in (record.get("scene_relations") or [])]
    rels = [rel for rel in rels if rel.get("subject") and rel.get("predicate") and rel.get("object")]
    step = str(record.get("fused_step") or record.get("decision_step") or record.get("state_step") or "").strip().upper()
    conf = float(record.get("fused_conf") or record.get("ensemble_conf") or record.get("state_conf") or 0.0)
    return AssistantSnapshot(
        frame_index=int(record.get("frame_index", 0)),
        step_id=step,
        step_confidence=conf,
        runner_up=str(record.get("fused_runner_up") or ""),
        visible_objects=visible,
        relevant_objects=relevant or visible[:3],
        object_counts=object_counts(visible),
        scene_relations=[f"{rel['subject']} {rel['predicate']} {rel['object']}" for rel in rels],
        relation_facts=rels,
        memory_step=str(record.get("memory_step") or "").strip().upper(),
        memory_confidence=float(record.get("memory_conf") or 0.0),
        memory_reason=str(record.get("memory_reason") or ""),
        memory_recalled=bool(record.get("memory_recalled", False)),
        review_action=str(record.get("review_action") or ""),
        review_reason=str(record.get("review_reason") or ""),
        recent_steps=[
            StepTimelineEvent(
                frame_index=int(record.get("frame_index", 0)),
                step_id=step,
                confidence=conf,
                source="online_replay",
                stable=bool(record.get("stable", False)),
            )
        ],
        memory_matches=[
            MemoryMatchSummary(
                step_id=str(record.get("memory_step") or step),
                store="online_memory",
                source="online_replay",
                score=float(record.get("memory_conf") or 0.0),
            )
        ]
        if bool(record.get("memory_recalled", False))
        else [],
    )


def variant_snapshot(snapshot: AssistantSnapshot, variant: str, feedback: List[FeedbackTimelineEvent] | None = None) -> AssistantSnapshot:
    feedback = list(feedback or [])
    if variant == "step_only":
        return replace(
            snapshot,
            visible_objects=[],
            relevant_objects=[],
            object_counts={},
            scene_relations=[],
            relation_facts=[],
            memory_step="",
            memory_confidence=0.0,
            memory_reason="",
            memory_recalled=False,
            memory_matches=[],
            recent_steps=[],
            recent_feedback=[],
            review_action="",
            review_reason="",
        )
    if variant == "no_visual":
        return replace(
            snapshot,
            visible_objects=[],
            relevant_objects=[],
            object_counts={},
            scene_relations=[],
            relation_facts=[],
            review_action="",
            review_reason="",
        )
    if variant == "no_relations":
        return replace(
            snapshot,
            scene_relations=[],
            relation_facts=[],
        )
    if variant in {"no_memory", "dialogue_only"}:
        return replace(
            snapshot,
            memory_step="",
            memory_confidence=0.0,
            memory_reason="",
            memory_recalled=False,
            memory_matches=[],
            recent_steps=[],
            recent_feedback=[],
            review_action="",
            review_reason="",
        )
    if variant in {"full", "full_feedback", "dialogue_feedback"}:
        return replace(snapshot, recent_feedback=feedback or snapshot.recent_feedback)
    return snapshot


def keyword_hit(answer: str, expected_keywords: str) -> str:
    expected_keywords = str(expected_keywords or "").strip()
    if not expected_keywords:
        return ""
    lowered = str(answer or "").lower()
    groups = [group.strip() for group in expected_keywords.split(";") if group.strip()]
    if not groups:
        return ""
    for group in groups:
        options = [token.strip().lower() for token in group.split("|") if token.strip()]
        if options and not any(option in lowered for option in options):
            return "False"
    return "True"


def contains_any(text: str, options: Iterable[str]) -> bool:
    lowered = str(text or "").lower()
    return any(str(option or "").lower() in lowered for option in options if str(option or "").strip())


EVIDENCE_ROLE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "object_presence": (
        "visible",
        "visibility",
        "see",
        "cannot see",
        "not visible",
        "missing part",
        "required part",
        "confirm the required parts",
        "cannot confirm",
        "part",
        "gear",
        "housing",
        "cover",
    ),
    "identity_disambiguation": (
        "identity",
        "correct gear",
        "wrong gear",
        "gear type",
        "type 2",
        "type_2",
        "type 3",
        "type_3",
        "type 7",
        "type_7",
        "type 8",
        "type_8",
        "hard-pair",
        "hard pair",
        "confusable",
    ),
    "slot_relation": (
        "slot",
        "inserted",
        "insertion",
        "inside",
        "containment",
        "contained",
        "relation",
        "housing slot",
        "gear in",
        "seated in",
    ),
    "seating_boundary": (
        "seated",
        "seating",
        "fully seated",
        "cover",
        "close",
        "closed",
        "gap",
        "boundary",
        "edge",
        "flush",
        "cover-housing",
        "cover housing",
    ),
    "geometry_alignment": (
        "align",
        "aligned",
        "alignment",
        "orientation",
        "oriented",
        "pose",
        "angle",
        "rotated",
        "wrong orientation",
    ),
    "occlusion_recovery": (
        "occluded",
        "occlusion",
        "hand",
        "object blocks",
        "blocked by",
        "blocking the",
        "clear view",
        "line of sight",
        "angle",
    ),
    "no_missing_evidence": (
        "sufficient",
        "enough evidence",
        "verified",
        "supported",
        "can proceed",
        "no missing",
        "not missing",
        "already visible",
    ),
}


def expected_evidence_roles(record: Dict[str, Any], category: str) -> List[str]:
    """Infer fine-grained evidence-role GT from video/task metadata.

    This is an evaluation-only mapping. It avoids requiring the assistant to
    emit internal ontology identifiers, while still checking whether answers
    name the type of evidence needed for the active claim.
    """

    if category != "EvidenceBoundary":
        return []
    stem = Path(str(record.get("_video") or "")).stem.lower()
    claim = str(record.get("_gt_claim") or "").lower()
    outcome = str(record.get("_gt_outcome") or "").lower()
    roles: List[str] = []

    def add(role: str) -> None:
        if role and role not in roles:
            roles.append(role)

    if outcome == "supported":
        add("no_missing_evidence")
    if "identity" in stem or "identity" in claim:
        add("identity_disambiguation")
    if any(token in stem or token in claim for token in ["cover_seated", "cover seated", "cover"]):
        add("seating_boundary")
    if any(token in stem or token in claim for token in ["gear_inserted", "gear inserted", "smallgear", "biggear", "inserted"]):
        add("slot_relation")
    if "wrong_orientation" in stem or "orientation" in claim:
        add("geometry_alignment")
    if stem.startswith("occ_") or "occlusion" in stem:
        add("occlusion_recovery")
        if "cover_edge" in stem or "cover" in stem:
            add("seating_boundary")
        if "slot" in stem or "gear" in stem:
            add("slot_relation")
    if not roles:
        targets = gt_target_components(record)
        if targets:
            add("object_presence")
    return roles


def evidence_role_correct(answer: str, roles: Sequence[str]) -> str:
    roles = [str(role or "").strip() for role in roles if str(role or "").strip()]
    if not roles:
        return ""
    for role in roles:
        aliases = EVIDENCE_ROLE_ALIASES.get(role, ())
        if aliases and contains_any(answer, aliases):
            return "True"
    return "False"


def display_tokens(component: str) -> List[str]:
    comp = str(component or "").strip().lower()
    if not comp:
        return []
    tokens = {comp, comp.replace("_", " "), comp.replace("gearbox_", "gearbox ")}
    if comp.startswith("type_"):
        parts = comp.split("_")
        if len(parts) >= 2:
            tokens.add(f"type {parts[1]}")
        if comp.endswith("_gear"):
            tokens.add("gear")
        if "housing" in comp:
            tokens.add("housing")
        if "cover" in comp:
            tokens.add("cover")
    return [token for token in tokens if token]


def outcome_from_answer(reply: Any) -> str:
    status = str(getattr(reply, "status", "") or "").lower()
    answer = str(getattr(reply, "answer", "") or "").lower()
    if status in {"insufficient", "abstain", "clarify"}:
        return "unresolved"
    if contains_any(answer, ["insufficient", "missing", "not enough", "cannot verify", "can't verify", "need more"]):
        return "unresolved"
    if contains_any(answer, ["contradict", "wrong", "not supported", "invalid", "cannot proceed", "blocked"]):
        return "contradicted"
    if contains_any(answer, ["supported", "verified", "can proceed", "looks correct", "is correct", "sufficient"]):
        return "supported"
    return "answer"


def grounded_scores(reply: Any, expected: Dict[str, Any], snapshot: AssistantSnapshot | None, record: Dict[str, Any] | None) -> Dict[str, Any]:
    if snapshot is None or record is None:
        return {
            "gt_correct": "",
            "step_correct": "",
            "object_evidence_correct": "",
            "claim_state_correct": "",
            "expected_evidence_roles": "",
            "evidence_role_correct": "",
        }
    category = str(expected.get("category", "") or expected.get("episode_id", "") or "").strip()
    answer = str(getattr(reply, "answer", "") or "").lower()
    gt_step = str(record.get("_gt_step") or "").strip().upper()
    gt_outcome = str(record.get("_gt_outcome") or "").strip().lower()
    targets = gt_target_components(record)
    visible = {str(name).strip().lower() for name in snapshot.visible_objects}

    step_correct = ""
    if gt_step and category in {"Step", "Guidance", "EvidenceBoundary"}:
        step_correct = str(snapshot.step_id == gt_step and gt_step.lower() in answer)

    object_correct = ""
    if targets and category in {"ObjectEvidence", "ClaimVerification"}:
        expected_visible = any(target in visible for target in targets)
        answer_mentions_target = any(contains_any(answer, display_tokens(target)) for target in targets)
        if gt_outcome == "supported":
            object_correct = str(expected_visible and answer_mentions_target)
        elif expected_visible:
            object_correct = str(answer_mentions_target)
        else:
            object_correct = str(contains_any(answer, ["not see", "do not see", "cannot verify", "missing", "not visible"]))

    claim_correct = ""
    if gt_outcome and category in {"Guidance", "EvidenceBoundary", "ClaimVerification"}:
        pred_outcome = outcome_from_answer(reply)
        if gt_outcome == "supported":
            claim_correct = str(pred_outcome == "supported" or (category == "Guidance" and not contains_any(answer, ["missing", "not enough", "cannot", "blocked"])))
        elif gt_outcome == "contradicted":
            claim_correct = str(pred_outcome == "contradicted" or contains_any(answer, ["wrong", "cannot proceed", "blocked"]))
        elif gt_outcome in {"unresolved", "uncertain"}:
            claim_correct = str(pred_outcome == "unresolved")

    gt_votes = [
        value
        for value in [step_correct, object_correct, claim_correct]
        if value in {"True", "False"}
    ]
    roles = expected_evidence_roles(record, category)
    role_correct = evidence_role_correct(str(getattr(reply, "answer", "") or ""), roles)
    gt_correct = "" if not gt_votes else str(all(value == "True" for value in gt_votes))
    return {
        "gt_step": gt_step,
        "gt_claim": record.get("_gt_claim", ""),
        "gt_outcome": gt_outcome,
        "gt_targets": ";".join(targets),
        "expected_evidence_roles": ";".join(roles),
        "evidence_role_correct": role_correct,
        "response_consistent": response_consistent_score(reply, gt_outcome),
        "gt_correct": gt_correct,
        "step_correct": step_correct,
        "object_evidence_correct": object_correct,
        "claim_state_correct": claim_correct,
    }


def response_consistent_score(reply: Any, gt_outcome: str) -> str:
    """Whether the surface answer avoids contradicting the known claim outcome.

    This is evaluated only when a video-level claim outcome is available. It is
    intentionally weaker than GT correctness: a response can be consistent
    while still lacking the exact step/object details.
    """

    outcome = str(gt_outcome or "").strip().lower()
    if not outcome:
        return ""
    pred = outcome_from_answer(reply)
    if outcome == "supported":
        return str(pred != "contradicted")
    if outcome == "contradicted":
        return str(pred != "supported")
    if outcome in {"unresolved", "uncertain"}:
        return str(pred != "supported")
    return ""


def score_row(
    reply: Any,
    expected: Dict[str, Any],
    snapshot: AssistantSnapshot | None = None,
    record: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    evidence = dict(reply.evidence or {})
    expected_route = str(expected.get("expected_route", "") or "").strip()
    expected_status = str(expected.get("expected_status", "") or "").strip()
    expected_target = str(expected.get("expected_target", "") or "").strip().lower()
    target = str(evidence.get("target_component", "") or "").strip().lower()
    status = str(reply.status or "")
    answer = str(reply.answer or "").lower()
    unsafe_answer = ""
    if expected_status in {"insufficient", "clarify", "abstain"}:
        unsafe_answer = str(status == "answer")
    if "do not have any recent feedback" in answer or "do not have enough" in answer:
        negative_memory_answer = "True"
    else:
        negative_memory_answer = "False"
    scores = {
        "route_match": "" if not expected_route else str(reply.route == expected_route),
        "status_match": "" if not expected_status else str(reply.status == expected_status),
        "target_match": "" if not expected_target else str(target == expected_target),
        "keyword_hit": keyword_hit(reply.answer, str(expected.get("expected_keywords", "") or "")),
        "used_history": str(bool(evidence.get("used_history", False))),
        "unsafe_answer": unsafe_answer,
        "negative_memory_answer": negative_memory_answer,
        "target_component": target,
    }
    scores.update(grounded_scores(reply, expected, snapshot, record))
    return scores


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def llm_diagnostics(reply: Any) -> Dict[str, Any]:
    evidence = dict(reply.evidence or {})
    profile = dict(evidence.get("llm_profile", {}) or {})
    invoked = bool(profile)
    return {
        "llm_invoked": str(invoked) if invoked else "",
        "llm_guard_rejected": str(bool(profile.get("guard_rejected", False))) if invoked else "",
        "llm_guard_reason": str(profile.get("guard_reason", "") or ""),
        "llm_generation_ms": f"{float(profile.get('generation_ms', 0.0) or 0.0):.4f}" if invoked else "",
        "llm_total_ms": f"{float(profile.get('total_ms', 0.0) or 0.0):.4f}" if invoked else "",
    }


def aggregate(rows: List[Dict[str, Any]], group_keys: List[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in group_keys)
        groups.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    metric_names = [
        "route_match",
        "status_match",
        "target_match",
        "keyword_hit",
        "response_consistent",
        "gt_correct",
        "step_correct",
        "object_evidence_correct",
        "claim_state_correct",
        "evidence_role_correct",
        "context_carry",
        "feedback_recall",
        "unsafe_answer",
        "llm_guard_rejected",
    ]
    for key, items in sorted(groups.items()):
        result = {group_keys[i]: key[i] for i in range(len(group_keys))}
        result["n"] = len(items)
        for metric in metric_names:
            vals = [bool_value(item.get(metric, "")) for item in items if str(item.get(metric, "")) != ""]
            result[metric] = "" if not vals else f"{sum(vals) / len(vals):.3f}"
        latencies = [float(item.get("latency_sec", 0.0) or 0.0) for item in items]
        result["mean_latency_sec"] = f"{sum(latencies) / max(1, len(latencies)):.4f}"
        ordered_latencies = sorted(latencies)
        if ordered_latencies:
            result["p50_latency_sec"] = f"{ordered_latencies[len(ordered_latencies) // 2]:.4f}"
            p90_index = min(len(ordered_latencies) - 1, int(0.90 * len(ordered_latencies)))
            result["p90_latency_sec"] = f"{ordered_latencies[p90_index]:.4f}"
        else:
            result["p50_latency_sec"] = ""
            result["p90_latency_sec"] = ""
        out.append(result)
    return out


def run_single_turn(
    records: List[Dict[str, Any]],
    kb: KnowledgeBase,
    questions: List[Dict[str, str]],
    llm: GroundedLLMResponder | None = None,
    llm_intents: List[str] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    variants: List[Tuple[str, str, bool]] = [
        ("step_only", "step_only", False),
        ("no_visual", "no_visual", False),
        ("no_relations", "no_relations", False),
        ("no_memory", "no_memory", False),
        ("full", "full", False),
    ]
    if llm is not None:
        variants.append(("full_qwen3", "full", True))
    for snap_idx, record in enumerate(records):
        base_snapshot = snapshot_from_record(record)
        for variant, snapshot_variant, use_llm in variants:
            assistant = ContextualAssistant(
                kb=kb,
                llm=llm if use_llm else None,
                llm_intents=llm_intents if use_llm else None,
            )
            snapshot = variant_snapshot(base_snapshot, snapshot_variant)
            for q_idx, question in enumerate(questions, start=1):
                start = time.perf_counter()
                reply = assistant.answer(question["question"], snapshot)
                latency = time.perf_counter() - start
                scored = score_row(reply, question, snapshot=snapshot, record=record)
                rows.append(
                    {
                        "suite": "single_turn",
                        "variant": variant,
                        "snapshot_id": snap_idx,
                        "video": record.get("_video", ""),
                        "frame_index": snapshot.frame_index,
                        "category": question.get("category", ""),
                        "question_id": q_idx,
                        "question": question["question"],
                        "expected_route": question.get("expected_route", ""),
                        "expected_status": question.get("expected_status", ""),
                        "expected_target": question.get("expected_target", ""),
                        "route": reply.route,
                        "status": reply.status,
                        "answer": reply.answer,
                        "latency_sec": f"{latency:.4f}",
                        **llm_diagnostics(reply),
                        **scored,
                    }
                )
    return rows


def run_multiturn(
    records: List[Dict[str, Any]],
    kb: KnowledgeBase,
    episodes: List[Dict[str, Any]],
    llm: GroundedLLMResponder | None = None,
    include_feedback_variant: bool = False,
    llm_intents: List[str] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    variants: List[Tuple[str, bool]] = [
        ("stateless", False),
        ("dialogue_only", False),
    ]
    if include_feedback_variant:
        variants.append(("dialogue_feedback", False))
    if llm is not None:
        variants.append(("dialogue_feedback_qwen3", True) if include_feedback_variant else ("dialogue_qwen3", True))
    for snap_idx, record in enumerate(records):
        base_snapshot = snapshot_from_record(record)
        for episode in episodes:
            for variant, use_llm in variants:
                assistant = ContextualAssistant(
                    kb=kb,
                    llm=llm if use_llm else None,
                    llm_intents=llm_intents if use_llm else None,
                )
                feedback_events: List[FeedbackTimelineEvent] = []
                for turn_idx, turn in enumerate(episode.get("turns", []), start=1):
                    if "feedback" in turn:
                        fb = dict(turn["feedback"])
                        feedback_events.append(
                            FeedbackTimelineEvent(
                                frame_index=base_snapshot.frame_index,
                                label=str(fb.get("label", "")),
                                source="scripted_user_feedback",
                                accepted=bool(fb.get("accepted", False)),
                                note=str(fb.get("note", "")),
                            )
                        )
                        rows.append(
                            {
                                "suite": "multi_turn",
                                "variant": variant,
                                "snapshot_id": snap_idx,
                                "video": record.get("_video", ""),
                                "frame_index": base_snapshot.frame_index,
                                "episode_id": episode.get("episode_id", ""),
                                "turn_id": turn_idx,
                                "turn_type": "feedback",
                                "question": "",
                                "answer": str(fb.get("note", "")),
                            }
                        )
                        continue
                    if variant == "stateless":
                        assistant = ContextualAssistant(kb=kb)
                    uses_feedback_memory = variant in {"dialogue_feedback", "dialogue_feedback_qwen3"}
                    snap_variant = "full_feedback" if uses_feedback_memory else "dialogue_only"
                    snapshot = variant_snapshot(base_snapshot, snap_variant, feedback_events if uses_feedback_memory else [])
                    start = time.perf_counter()
                    reply = assistant.answer(str(turn.get("user", "")), snapshot)
                    latency = time.perf_counter() - start
                    turn_expected = dict(turn)
                    turn_expected.setdefault("category", str(episode.get("category", "")))
                    scored = score_row(reply, turn_expected, snapshot=snapshot, record=record)
                    context_carry = ""
                    if bool(turn.get("requires_dialogue_history", False)):
                        context_carry = str(bool_value(scored.get("target_match", "")) and bool_value(scored.get("used_history", "")))
                    feedback_recall = ""
                    if bool(turn.get("requires_feedback_memory", False)):
                        feedback_recall = str(
                            bool_value(scored.get("keyword_hit", ""))
                            and not bool_value(scored.get("negative_memory_answer", ""))
                        )
                    rows.append(
                        {
                            "suite": "multi_turn",
                            "variant": variant,
                            "snapshot_id": snap_idx,
                            "video": record.get("_video", ""),
                            "frame_index": snapshot.frame_index,
                            "episode_id": episode.get("episode_id", ""),
                            "turn_id": turn_idx,
                            "turn_type": "question",
                            "question": turn.get("user", ""),
                            "expected_route": turn.get("expected_route", ""),
                            "expected_status": turn.get("expected_status", ""),
                            "expected_target": turn.get("expected_target", ""),
                            "route": reply.route,
                            "status": reply.status,
                            "answer": reply.answer,
                            "latency_sec": f"{latency:.4f}",
                            **llm_diagnostics(reply),
                            "context_carry": context_carry,
                            "feedback_recall": feedback_recall,
                            **scored,
                        }
                    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--kb", type=Path, default=ROOT / "components.json")
    parser.add_argument("--single-questions", type=Path, default=ROOT / "data" / "assistant_qa" / "single_turn_questions.csv")
    parser.add_argument("--multiturn-episodes", type=Path, default=ROOT / "data" / "assistant_qa" / "multiturn_episodes.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "assistant_qa")
    parser.add_argument("--video-hint", action="append", default=None)
    parser.add_argument("--all-videos", action="store_true", help="Sample snapshots from every successful run in the summary CSV.")
    parser.add_argument("--snapshots-per-video", type=int, default=1)
    parser.add_argument(
        "--relation-visible-only",
        action="store_true",
        help="Keep only snapshots with at least two visible objects and at least one scene relation.",
    )
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap on selected snapshots after filtering.")
    parser.add_argument("--use-qwen3", action="store_true", help="Evaluate the full assistant with the Qwen3 grounded responder.")
    parser.add_argument("--qwen3-model-path", type=str, default=default_qwen3_model_path())
    parser.add_argument("--qwen3-model-id", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--qwen3-device-map", type=str, default="auto")
    parser.add_argument("--qwen3-torch-dtype", type=str, default="auto")
    parser.add_argument("--qwen3-timeout-sec", type=float, default=8.0)
    parser.add_argument("--qwen3-max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--llm-intent",
        action="append",
        default=None,
        help="Invoke Qwen only for these parsed intents. Repeat for multiple intents.",
    )
    parser.add_argument(
        "--qwen3-include-template-answer",
        action="store_true",
        help="Pass the deterministic fallback answer to Qwen3. Disabled by default for non-leaky QA evaluation.",
    )
    parser.add_argument(
        "--include-feedback-memory-variant",
        action="store_true",
        help="Include scripted feedback-memory variants. Use only for a memory unit diagnostic, not the main no-HITL assistant QA evaluation.",
    )
    args = parser.parse_args()

    if args.all_videos:
        records = choose_records_all(args.summary_csv, int(args.snapshots_per_video))
    else:
        video_hints = args.video_hint or [
            "VIEW_B_cover_seated_supported_01",
            "VIEW_B_smallgear_inserted_supported_01",
            "VIEW_gear_identity_type2_01",
        ]
        records = choose_records(args.summary_csv, list(video_hints), int(args.snapshots_per_video))
    if args.relation_visible_only:
        records = [record for record in records if relation_visible(record)]
    if int(args.max_records) > 0:
        records = records[: int(args.max_records)]
    attach_gt_metadata(records, args.gt_dir)
    kb = KnowledgeBase.from_path(args.kb)
    single_questions = read_csv(args.single_questions)
    episodes = json.loads(args.multiturn_episodes.read_text(encoding="utf-8"))
    llm = None
    if args.use_qwen3:
        llm = GroundedLLMResponder(
            enabled=True,
            provider="transformers",
            model_id=args.qwen3_model_id,
            model_path=args.qwen3_model_path,
            device_map=args.qwen3_device_map,
            torch_dtype=args.qwen3_torch_dtype,
            max_new_tokens=args.qwen3_max_new_tokens,
            temperature=0.0,
            top_p=0.9,
            timeout_sec=args.qwen3_timeout_sec,
            history_turns=4,
            answer_word_limit=55,
            streaming=False,
            load_on_start=True,
            fallback_to_template=True,
            include_fallback_in_prompt=bool(args.qwen3_include_template_answer),
            trust_remote_code=True,
        )

    llm_intents = [str(value).strip().lower() for value in (args.llm_intent or []) if str(value).strip()]
    single_rows = run_single_turn(
        records,
        kb,
        single_questions,
        llm=llm,
        llm_intents=llm_intents or None,
    )
    multi_rows = run_multiturn(
        records,
        kb,
        episodes,
        llm=llm,
        include_feedback_variant=bool(args.include_feedback_memory_variant),
        llm_intents=llm_intents or None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detail_fields = [
        "suite",
        "variant",
        "snapshot_id",
        "video",
        "frame_index",
        "category",
        "episode_id",
        "turn_id",
        "turn_type",
        "question_id",
        "question",
        "expected_route",
        "expected_status",
        "expected_target",
        "route",
        "status",
        "target_component",
        "gt_step",
        "gt_claim",
        "gt_outcome",
        "gt_targets",
        "expected_evidence_roles",
        "route_match",
        "status_match",
        "target_match",
        "keyword_hit",
        "response_consistent",
        "gt_correct",
        "step_correct",
        "object_evidence_correct",
        "claim_state_correct",
        "evidence_role_correct",
        "used_history",
        "context_carry",
        "feedback_recall",
        "unsafe_answer",
        "llm_invoked",
        "llm_guard_rejected",
        "llm_guard_reason",
        "llm_generation_ms",
        "llm_total_ms",
        "latency_sec",
        "answer",
    ]
    write_csv(args.output_dir / "single_turn_details.csv", single_rows, detail_fields)
    write_csv(args.output_dir / "multi_turn_details.csv", multi_rows, detail_fields)
    write_csv(
        args.output_dir / "single_turn_summary.csv",
        aggregate(single_rows, ["variant", "category"]),
        [
            "variant",
            "category",
            "n",
            "route_match",
            "status_match",
            "target_match",
            "keyword_hit",
            "response_consistent",
            "gt_correct",
            "step_correct",
            "object_evidence_correct",
            "claim_state_correct",
            "evidence_role_correct",
            "context_carry",
            "feedback_recall",
            "unsafe_answer",
            "llm_guard_rejected",
            "mean_latency_sec",
            "p50_latency_sec",
            "p90_latency_sec",
        ],
    )
    write_csv(
        args.output_dir / "multi_turn_summary.csv",
        aggregate([row for row in multi_rows if row.get("turn_type") != "feedback"], ["variant", "episode_id"]),
        [
            "variant",
            "episode_id",
            "n",
            "route_match",
            "status_match",
            "target_match",
            "keyword_hit",
            "response_consistent",
            "gt_correct",
            "step_correct",
            "object_evidence_correct",
            "claim_state_correct",
            "evidence_role_correct",
            "context_carry",
            "feedback_recall",
            "unsafe_answer",
            "llm_guard_rejected",
            "mean_latency_sec",
            "p50_latency_sec",
            "p90_latency_sec",
        ],
    )
    payload = {
        "records": len(records),
        "gt_annotated_records": sum(1 for record in records if record.get("_gt_step")),
        "single_questions": len(single_questions),
        "episodes": len(episodes),
        "single_rows": len(single_rows),
        "multi_rows": len(multi_rows),
        "output_dir": str(args.output_dir.resolve()),
        "qwen3_enabled": bool(args.use_qwen3),
        "qwen3_model_path": str(args.qwen3_model_path) if args.use_qwen3 else "",
        "qwen3_include_template_answer": bool(args.qwen3_include_template_answer) if args.use_qwen3 else False,
        "llm_intents": llm_intents,
        "include_feedback_memory_variant": bool(args.include_feedback_memory_variant),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
