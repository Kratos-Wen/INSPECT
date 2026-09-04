"""Export online-corrected assistance logs as INSPECT supervision traces."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .state_spec import decompose_evidence, load_state_specs, state_spec_for
from .types import StateSpec, VerifiedTraceEvent


_SLUG_RE = re.compile(r"[^a-zA-Z0-9_]+")


def _slug(text: object) -> str:
    value = str(text or "").strip().lower().replace("-", "_").replace(" ", "_")
    value = _SLUG_RE.sub("_", value)
    return value.strip("_")


def _state(text: object) -> str:
    return str(text or "").strip().upper()


def _usable_evidence_key(key: object) -> str:
    text = str(key or "").strip()
    if not text:
        return ""
    parts = text.split(":")
    if len(parts) >= 4 and parts[0] in {"relation", "scene"}:
        if parts[0] == "relation" and _slug(parts[1]) == _slug(parts[3]):
            return ""
        if len(parts) >= 5 and parts[0] == "scene" and parts[1] in {"relation_started", "relation_ended"}:
            if _slug(parts[2]) == _slug(parts[4]):
                return ""
    return text


def _float(payload: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def write_trace_jsonl(events: Iterable[VerifiedTraceEvent], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")


def load_trace_jsonl(path: Path) -> List[VerifiedTraceEvent]:
    return [VerifiedTraceEvent.from_dict(record) for record in _read_jsonl(path)]


def _feedback_by_frame(records: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    indexed: Dict[int, Dict[str, Any]] = {}
    for record in records:
        try:
            frame_index = int(record.get("frame_index", -1))
        except (TypeError, ValueError):
            continue
        indexed[frame_index] = record
    return indexed


def _counts_from_record(record: Dict[str, Any], key: str) -> Dict[str, int]:
    token = dict(record.get("evidence_token") or {})
    source = token.get(key) if key in token else record.get(key)
    if not isinstance(source, dict) and key == "visible_counts":
        source = _counts_from_detections(record.get("fused_detections", []) or record.get("raw_detections", []))
    if not isinstance(source, dict) and key == "relevant_counts":
        source = _counts_from_detections(record.get("relevant_detections", []))
    if not isinstance(source, dict):
        return {}
    counts: Dict[str, int] = {}
    for name, value in source.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            counts[str(name)] = count
    return counts


def _counts_from_detections(detections: object) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not isinstance(detections, list):
        return counts
    for item in detections:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts


def _relation_facts_from_record(record: Dict[str, Any]) -> List[List[str]]:
    facts: List[List[str]] = []
    token = dict(record.get("evidence_token") or {})
    for item in token.get("relation_facts", []) or []:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            facts.append([str(item[0]), str(item[1]), str(item[2])])
    for item in record.get("scene_graph_relations", []) or []:
        if isinstance(item, dict):
            subject = item.get("subject")
            predicate = item.get("predicate")
            obj = item.get("object")
            if subject and predicate and obj:
                facts.append([str(subject), str(predicate), str(obj)])
    deduped: List[List[str]] = []
    seen: set[Tuple[str, str, str]] = set()
    for subject, predicate, obj in facts:
        key = (_slug(subject), _slug(predicate), _slug(obj))
        if key in seen:
            continue
        seen.add(key)
        deduped.append([subject, predicate, obj])
    return deduped


def _observed_evidence(
    visible_counts: Dict[str, int],
    relevant_counts: Dict[str, int],
    relation_counts: Dict[str, int],
    relation_facts: List[List[str]],
    record: Dict[str, Any],
) -> List[str]:
    evidence: set[str] = set()
    if bool(record.get("has_visual_evidence", False)):
        evidence.add("visual:evidence_present")
    token = dict(record.get("evidence_token") or {})
    for name, count in visible_counts.items():
        if count > 0:
            evidence.add(f"object:{_slug(name)}")
    for name, count in dict(token.get("track_counts") or {}).items():
        try:
            count_value = int(count)
        except (TypeError, ValueError):
            count_value = 0
        if count_value > 0:
            evidence.add(f"track:object:{_slug(name)}")
            evidence.add(f"object:{_slug(name)}")
    for name, count in dict(token.get("stable_track_counts") or {}).items():
        try:
            count_value = int(count)
        except (TypeError, ValueError):
            count_value = 0
        if count_value > 0:
            evidence.add(f"track:stable_object:{_slug(name)}")
    for key in token.get("track_evidence_keys", []) or []:
        usable = _usable_evidence_key(key)
        if usable:
            evidence.add(usable)
    for key in token.get("scene_evidence_keys", []) or []:
        usable = _usable_evidence_key(key)
        if usable:
            evidence.add(usable)
    for key in token.get("scene_relation_keys", []) or []:
        usable = _usable_evidence_key(key)
        if usable:
            evidence.add(usable)
    for key in token.get("scene_relation_change_keys", []) or []:
        usable = _usable_evidence_key(key)
        if usable:
            evidence.add(usable)
    for key in token.get("scene_transition_keys", []) or []:
        usable = _usable_evidence_key(key)
        if usable:
            evidence.add(usable)
    for name, count in relevant_counts.items():
        if count > 0:
            evidence.add(f"focus_object:{_slug(name)}")
            evidence.add(f"object:{_slug(name)}")
    for key, count in relation_counts.items():
        if count > 0:
            evidence.add(f"relation_type:{_slug(key)}")
    for subject, predicate, obj in relation_facts:
        if _slug(subject) == _slug(obj):
            continue
        pred = _slug(predicate)
        evidence.add(f"relation:{_slug(subject)}:{pred}:{_slug(obj)}")
        evidence.add(f"relation_type:{pred}")
    memory_step = _state(record.get("memory_step"))
    if bool(record.get("memory_recalled", False)) and memory_step:
        evidence.add(f"memory_match:{memory_step.lower()}")
    for name, count in dict(token.get("contact_counts") or {}).items():
        try:
            count_value = int(count)
        except (TypeError, ValueError):
            count_value = 0
        if count_value > 0:
            evidence.add(f"interaction:active_object:{_slug(name)}")
    for item in token.get("contact_facts", []) or []:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            subject, predicate, obj = item[:3]
            evidence.add(f"contact:{_slug(subject)}:{_slug(predicate)}:{_slug(obj)}")
    active_object = str(token.get("active_object", "") or "")
    if active_object:
        evidence.add(f"interaction:active_object:{_slug(active_object)}")
    contact_phase = str(token.get("contact_phase", "") or "")
    if contact_phase and contact_phase != "none":
        evidence.add(f"interaction:contact_phase:{_slug(contact_phase)}")
    interaction_target = str(token.get("interaction_target", "") or "")
    if interaction_target:
        evidence.add(f"interaction_target:{_slug(interaction_target)}")
    return sorted(evidence)


def _missing_evidence(record: Dict[str, Any], min_auto_confidence: float) -> List[str]:
    missing: set[str] = set()
    predicted = _state(record.get("fused_step"))
    confidence = _float(record, "fused_conf")
    if not bool(record.get("has_visual_evidence", False)):
        missing.add("visual:evidence_present")
    if not bool(record.get("stable", False)):
        missing.add("temporal:stable_observation")
    token = dict(record.get("evidence_token") or {})
    if bool(record.get("has_visual_evidence", False)) and not dict(token.get("stable_track_counts") or {}):
        missing.add("track:stable_object")
    if predicted and confidence < min_auto_confidence:
        missing.add(f"confidence:{predicted.lower()}")
    review_action = str(record.get("review_action", "") or "")
    if review_action in {"hold", "request_human", "prefer_candidate"}:
        missing.add(f"review:{_slug(review_action)}")
    for trigger in record.get("review_triggers", []) or []:
        missing.add(f"review_trigger:{_slug(trigger)}")
    return sorted(item for item in missing if item)


def _failure_cues(record: Dict[str, Any], feedback: Optional[Dict[str, Any]]) -> List[str]:
    cues: set[str] = set()
    review_action = str(record.get("review_action", "") or "")
    review_reason = str(record.get("review_reason", "") or "")
    if review_action in {"hold", "request_human", "prefer_candidate"} and review_reason:
        cues.add(f"review_reason:{_slug(review_reason)[:80]}")
    for trigger in record.get("review_triggers", []) or []:
        cues.add(f"trigger:{_slug(trigger)}")
    memory_step = _state(record.get("memory_step"))
    fused_step = _state(record.get("fused_step"))
    if memory_step and fused_step and memory_step != fused_step and _float(record, "memory_conf") >= 0.45:
        cues.add(f"memory_conflict:{memory_step.lower()}_vs_{fused_step.lower()}")
    if feedback:
        fb = dict(feedback.get("feedback") or {})
        evidence_feedback = dict((fb.get("extras") or {}).get("evidence_feedback") or {})
        status = str(evidence_feedback.get("status", "") or "")
        reason = str(evidence_feedback.get("reason", "") or "")
        if status in {"rejected", "occluded"}:
            cues.add(f"evidence_{status}:{_slug(reason or status)}")
        note = str(fb.get("note", "") or "")
        if note:
            cues.add(f"human_note:{_slug(note)[:80]}")
        if not bool(fb.get("accepted", False)):
            label = _state(fb.get("label"))
            if label and fused_step and label != fused_step:
                cues.add(f"human_correction:{fused_step.lower()}_to_{label.lower()}")
    return sorted(item for item in cues if item)


def _feedback_evidence_overrides(feedback: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    if not feedback:
        return {"postcondition": [], "missing": [], "negative": []}
    fb = dict(feedback.get("feedback") or {})
    evidence_feedback = dict((fb.get("extras") or {}).get("evidence_feedback") or {})
    observed = [str(item) for item in evidence_feedback.get("observed_evidence", []) if str(item)]
    status = str(evidence_feedback.get("status", "") or "")
    reason = _slug(evidence_feedback.get("reason", "") or status)
    if status == "verified":
        return {"postcondition": observed, "missing": [], "negative": []}
    if status == "occluded":
        return {"postcondition": [], "missing": observed or [f"occluded:{reason}"], "negative": []}
    if status == "rejected":
        return {"postcondition": [], "missing": [], "negative": [f"failure:{reason}"] if reason else []}
    return {"postcondition": [], "missing": [], "negative": []}


def _feedback_evidence_status(feedback: Optional[Dict[str, Any]]) -> str:
    if not feedback:
        return ""
    fb = dict(feedback.get("feedback") or {})
    evidence_feedback = dict((fb.get("extras") or {}).get("evidence_feedback") or {})
    return str(evidence_feedback.get("status", "") or "").strip().lower()


def _verified_state(
    record: Dict[str, Any],
    feedback: Optional[Dict[str, Any]],
    min_auto_confidence: float,
) -> Tuple[str, str, bool, bool]:
    predicted = _state(record.get("fused_step"))
    if feedback:
        fb = dict(feedback.get("feedback") or {})
        label = _state(fb.get("label"))
        if label:
            accepted = bool(fb.get("accepted", False))
            source = "human_accept" if accepted else "human_correction"
            return label, source, True, accepted
    review_action = str(record.get("review_action", "") or "")
    review_label = _state(record.get("review_label"))
    if review_action == "prefer_candidate" and review_label:
        return review_label, "reviewer_correction", True, False
    if (
        predicted
        and bool(record.get("stable", False))
        and bool(record.get("has_visual_evidence", False))
        and _float(record, "fused_conf") >= min_auto_confidence
        and review_action not in {"hold", "request_human", "prefer_candidate"}
    ):
        return predicted, "stable_auto", True, False
    return "", "unverified", False, False


def _verification_level(source: str, record: Dict[str, Any]) -> str:
    if bool(record.get("multi_view_verified", False)):
        return "L4"
    if source == "human_correction":
        return "L3"
    if source in {"human_accept", "reviewer_correction"}:
        return "L2"
    if source == "stable_auto":
        return "L1"
    return "L0"


def _trust_weight(level: str, source: str) -> float:
    if level == "L4":
        return 0.95
    if source == "human_correction":
        return 1.0
    if source == "human_accept":
        return 0.90
    if source == "reviewer_correction":
        return 0.75
    if level == "L1":
        return 0.10
    return 0.0


def _can_create_strong_edge(level: str, trust_weight: float) -> bool:
    return bool(level in {"L2", "L3", "L4"} and float(trust_weight) >= 0.70)


def _next_step_admissible(
    verified: bool,
    level: str,
    decomposition: Dict[str, List[str]],
    spec: Optional[StateSpec],
) -> Optional[bool]:
    if not verified or level not in {"L2", "L3", "L4"}:
        return False
    if decomposition.get("negative_evidence"):
        return False
    observed_admissibility = set(decomposition.get("admissibility_evidence", []))
    required_admissibility = set(spec.admissibility_evidence if spec else [])
    if required_admissibility:
        observed = observed_admissibility.intersection(required_admissibility)
        inside_hits = [key for key in observed if ":inside:" in key]
        aligned_hits = [key for key in observed if ":aligned_with:" in key]
        seated_hits = [key for key in observed if ":contacting:" in key or ":overlapping:" in key]
        if inside_hits:
            return True
        if aligned_hits and seated_hits:
            return True
        if not any(":inside:" in key or ":aligned_with:" in key or ":contacting:" in key or ":overlapping:" in key for key in required_admissibility):
            return bool(observed)
        return False
    required_postconditions = set(spec.postcondition_evidence if spec else [])
    if required_postconditions:
        observed = set(decomposition.get("postcondition_evidence", [])).intersection(required_postconditions)
        relation_hits = [key for key in observed if key.startswith("relation:")]
        if relation_hits:
            return True
        return bool(observed and not any(key.startswith(("object:", "track:object:")) for key in observed))
    if decomposition.get("postcondition_evidence"):
        return True
    return None


def _expert_states(record: Dict[str, Any]) -> Dict[str, str]:
    return {
        "state": _state(record.get("state_step")),
        "temporal": _state(record.get("temporal_step")),
        "retrieval": _state(record.get("retrieval_step")),
        "memory": _state(record.get("memory_step")),
        "ensemble": _state(record.get("ensemble_step")),
    }


def _expert_confidences(record: Dict[str, Any]) -> Dict[str, float]:
    return {
        "state": _float(record, "state_conf"),
        "temporal": _float(record, "temporal_conf"),
        "retrieval": _float(record, "retrieval_conf"),
        "memory": _float(record, "memory_conf"),
        "ensemble": _float(record, "ensemble_conf"),
    }


def export_verified_traces(
    run_dir: Path,
    output_path: Optional[Path] = None,
    min_auto_confidence: float = 0.78,
    state_specs: Optional[Dict[str, StateSpec]] = None,
    state_specs_path: Optional[Path] = None,
) -> List[VerifiedTraceEvent]:
    """Convert an INSPECT trace run directory into verified trace events."""

    run_dir = Path(run_dir)
    meta = _read_json(run_dir / "meta.json")
    run_id = str(run_dir.name)
    source_uri = str(meta.get("video_path", ""))
    if state_specs is None and state_specs_path is not None:
        state_specs = load_state_specs(Path(state_specs_path))
    feedback_index = _feedback_by_frame(_read_jsonl(run_dir / "feedback.jsonl"))
    iterations = _read_jsonl(run_dir / "iterations.jsonl")
    traces: List[VerifiedTraceEvent] = []
    for record in iterations:
        try:
            frame_index = int(record.get("frame_index", 0))
        except (TypeError, ValueError):
            frame_index = 0
        feedback = feedback_index.get(frame_index)
        visible_counts = _counts_from_record(record, "visible_counts")
        relevant_counts = _counts_from_record(record, "relevant_counts")
        relation_counts = _counts_from_record(record, "relation_counts")
        relation_facts = _relation_facts_from_record(record)
        observed = _observed_evidence(visible_counts, relevant_counts, relation_counts, relation_facts, record)
        missing = _missing_evidence(record, min_auto_confidence=min_auto_confidence)
        verified_state, source, verified, accepted = _verified_state(
            record,
            feedback=feedback,
            min_auto_confidence=min_auto_confidence,
        )
        confidence = _float(record, "fused_conf")
        predicted = _state(record.get("fused_step"))
        spec = state_spec_for(state_specs, predicted_state=predicted, verified_state=verified_state)
        candidate_state = spec.state_id if spec is not None else (verified_state or predicted)
        level = _verification_level(source, record)
        trust = _trust_weight(level, source)
        can_create_strong_edge = _can_create_strong_edge(level, trust)
        decomposition = decompose_evidence(
            observed_evidence=observed,
            missing_evidence=missing,
            failure_cues=_failure_cues(record, feedback),
            spec=spec,
        )
        evidence_overrides = _feedback_evidence_overrides(feedback)
        evidence_status = _feedback_evidence_status(feedback)
        if evidence_overrides["postcondition"]:
            decomposition["postcondition_evidence"] = sorted(
                set(decomposition["postcondition_evidence"]) | set(evidence_overrides["postcondition"])
            )
        if evidence_overrides["missing"]:
            missing = sorted(set(missing) | set(evidence_overrides["missing"]))
        if evidence_overrides["negative"]:
            decomposition["negative_evidence"] = sorted(
                set(decomposition["negative_evidence"]) | set(evidence_overrides["negative"])
            )
        if evidence_status in {"rejected", "occluded", "unknown"}:
            decomposition["postcondition_evidence"] = []
            decomposition["admissibility_evidence"] = []
            can_create_strong_edge = False
        next_step_admissible = _next_step_admissible(
            verified=verified,
            level=level,
            decomposition=decomposition,
            spec=spec,
        )
        traces.append(
            VerifiedTraceEvent(
                run_id=run_id,
                frame_index=frame_index,
                iteration=int(record.get("iter", 0) or 0),
                predicted_state=predicted,
                verified_state=verified_state,
                candidate_state=candidate_state,
                verification_source=source,
                verification_level=level,
                trust_weight=trust,
                can_create_strong_edge=can_create_strong_edge,
                verified=verified,
                accepted=accepted,
                stable=bool(record.get("stable", False)),
                confidence=confidence,
                uncertainty=max(0.0, 1.0 - confidence),
                prev_state=_state((record.get("evidence_token") or {}).get("prev_step") or record.get("prev_state")) or None,
                review_action=str(record.get("review_action", "") or ""),
                review_reason=str(record.get("review_reason", "") or ""),
                review_triggers=[str(item) for item in record.get("review_triggers", []) or []],
                observed_evidence=observed,
                missing_evidence=missing,
                failure_cues=_failure_cues(record, feedback),
                precondition_evidence=decomposition["precondition_evidence"],
                interaction_evidence=decomposition["interaction_evidence"],
                transition_evidence=decomposition["transition_evidence"],
                postcondition_evidence=decomposition["postcondition_evidence"],
                negative_evidence=decomposition["negative_evidence"],
                admissibility_evidence=decomposition["admissibility_evidence"],
                next_step_admissible=next_step_admissible,
                visible_counts=visible_counts,
                relevant_counts=relevant_counts,
                relation_counts=relation_counts,
                relation_facts=relation_facts,
                expert_states=_expert_states(record),
                expert_confidences=_expert_confidences(record),
                metadata={
                    "source_uri": source_uri,
                    "has_visual_evidence": bool(record.get("has_visual_evidence", False)),
                    "memory_active": bool(record.get("memory_active", False)),
                    "memory_reason": str(record.get("memory_reason", "") or ""),
                    "memory_recalled": bool(record.get("memory_recalled", False)),
                    "track_counts": dict((record.get("evidence_token") or {}).get("track_counts") or {}),
                    "stable_track_counts": dict((record.get("evidence_token") or {}).get("stable_track_counts") or {}),
                    "track_evidence_keys": list((record.get("evidence_token") or {}).get("track_evidence_keys") or []),
                    "scene_evidence_keys": [
                        key
                        for key in (
                            _usable_evidence_key(item)
                            for item in (record.get("evidence_token") or {}).get("scene_evidence_keys", []) or []
                        )
                        if key
                    ],
                    "scene_visible_objects": list((record.get("evidence_token") or {}).get("scene_visible_objects") or []),
                    "scene_stable_objects": list((record.get("evidence_token") or {}).get("scene_stable_objects") or []),
                    "scene_active_objects": list((record.get("evidence_token") or {}).get("scene_active_objects") or []),
                    "scene_moving_objects": list((record.get("evidence_token") or {}).get("scene_moving_objects") or []),
                    "scene_relation_keys": [
                        key
                        for key in (
                            _usable_evidence_key(item)
                            for item in (record.get("evidence_token") or {}).get("scene_relation_keys", []) or []
                        )
                        if key
                    ],
                    "scene_relation_change_keys": [
                        key
                        for key in (
                            _usable_evidence_key(item)
                            for item in (record.get("evidence_token") or {}).get("scene_relation_change_keys", []) or []
                        )
                        if key
                    ],
                    "scene_transition_keys": [
                        key
                        for key in (
                            _usable_evidence_key(item)
                            for item in (record.get("evidence_token") or {}).get("scene_transition_keys", []) or []
                        )
                        if key
                    ],
                    "procedural_scene_evidence": dict(record.get("scene_evidence") or {}),
                    "evidence_status": evidence_status,
                    "ensemble_margin": _float(record, "ensemble_margin"),
                    "expert_disagreement": _float(record, "expert_disagreement"),
                    "gates": dict(record.get("gates") or {}),
                    "state_spec": spec.to_dict() if spec is not None else {},
                },
            )
        )
    if output_path is not None:
        write_trace_jsonl(traces, Path(output_path))
    return traces
