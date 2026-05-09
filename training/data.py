"""Dataset export utilities for temporal step training."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..components import KnowledgeBase, TemporalEvidenceVectorizer
from ..types import EvidenceToken


@dataclass(frozen=True)
class LabelDecision:
    """Reliable supervision chosen for one frame."""

    step_id: str
    source: str
    weight: float


@dataclass(frozen=True)
class RunFrame:
    """One exported run frame with reconstructed evidence and vectorized features."""

    index: int
    record: dict[str, object]
    token: EvidenceToken
    vector: np.ndarray
    signature_key: Tuple[Tuple[str, int], ...]


def build_temporal_dataset(
    runs_root: str | Path,
    kb_path: str | Path,
    output_path: str | Path,
    window_size: int = 12,
    min_stable_confidence: float = 0.72,
    min_auto_confidence: float = 0.80,
    min_visual_confidence: float = 0.28,
    min_memory_confidence: float = 0.45,
    min_segment_length: int = 3,
    min_segment_vote_ratio: float = 0.58,
    min_segment_margin: float = 0.10,
    max_runs: Optional[int] = None,
) -> dict[str, object]:
    """Export temporal training samples from existing run logs."""

    kb = KnowledgeBase.from_path(kb_path)
    steps = kb.workflow_steps(["S1", "S2", "S3", "S4"])
    component_names = kb.component_names()
    vectorizer = TemporalEvidenceVectorizer(steps=steps, component_names=component_names)

    vectors: List[np.ndarray] = []
    labels: List[int] = []
    weights: List[float] = []
    aux_masks: List[float] = []
    aux_next_bootstrap: List[np.ndarray] = []
    aux_next_relations: List[np.ndarray] = []
    aux_next_flags: List[np.ndarray] = []
    metadata: List[dict[str, object]] = []
    run_names: List[str] = []
    source_counts: Dict[str, int] = {}
    skipped_counts: Dict[str, int] = {}

    runs = [path for path in sorted(Path(runs_root).iterdir()) if path.is_dir()]
    if max_runs is not None:
        runs = runs[: max(0, int(max_runs))]

    for run_dir in runs:
        records = list(_iter_iteration_records(run_dir))
        if not records:
            skipped_counts["empty_run"] = skipped_counts.get("empty_run", 0) + 1
            continue
        feedback_map = _load_feedback_map(run_dir)
        frames: List[RunFrame] = []
        previous_step: Optional[str] = None
        for record in records:
            token = _extract_token(record, steps=steps, prev_step=previous_step)
            if token is None:
                skipped_counts["missing_token"] = skipped_counts.get("missing_token", 0) + 1
                continue
            previous_step = str(record.get("fused_step") or previous_step or "").strip().upper() or previous_step
            frames.append(
                RunFrame(
                    index=len(frames),
                    record=dict(record),
                    token=token,
                    vector=vectorizer.vectorize(token),
                    signature_key=_signature_key(record, token),
                )
            )

        if not frames:
            skipped_counts["no_token_frames"] = skipped_counts.get("no_token_frames", 0) + 1
            continue

        decisions = _mine_run_labels(
            frames,
            feedback_map=feedback_map,
            steps=steps,
            min_stable_confidence=min_stable_confidence,
            min_auto_confidence=min_auto_confidence,
            min_visual_confidence=min_visual_confidence,
            min_memory_confidence=min_memory_confidence,
            min_segment_length=min_segment_length,
            min_segment_vote_ratio=min_segment_vote_ratio,
            min_segment_margin=min_segment_margin,
        )

        frame_vectors = [frame.vector for frame in frames]
        for frame in frames:
            decision = decisions.get(frame.index)
            if decision is None:
                skipped_counts["unlabeled_frame"] = skipped_counts.get("unlabeled_frame", 0) + 1
                continue
            if not frame.token.has_visual_evidence:
                skipped_counts["no_visual_evidence"] = skipped_counts.get("no_visual_evidence", 0) + 1
                continue
            label_index = steps.index(decision.step_id) if decision.step_id in steps else -1
            if label_index < 0:
                skipped_counts["unknown_label"] = skipped_counts.get("unknown_label", 0) + 1
                continue

            sequence = _left_padded_window(frame_vectors, index=frame.index, window_size=window_size)
            record = frame.record
            next_frame = frames[frame.index + 1] if frame.index + 1 < len(frames) else None
            aux_mask = 1.0 if next_frame is not None else 0.0
            next_bootstrap = (
                vectorizer.bootstrap_array(next_frame.token)
                if next_frame is not None
                else np.zeros((len(steps),), dtype=np.float32)
            )
            next_relations = (
                vectorizer.relation_presence_vector(next_frame.token)
                if next_frame is not None
                else np.zeros((len(vectorizer.relation_types),), dtype=np.float32)
            )
            next_flags = (
                vectorizer.flag_vector(next_frame.token)
                if next_frame is not None
                else np.zeros((len(vectorizer.FLAG_NAMES),), dtype=np.float32)
            )
            vectors.append(sequence)
            labels.append(label_index)
            weights.append(float(decision.weight))
            aux_masks.append(float(aux_mask))
            aux_next_bootstrap.append(next_bootstrap)
            aux_next_relations.append(next_relations)
            aux_next_flags.append(next_flags)
            run_names.append(run_dir.name)
            source_counts[decision.source] = source_counts.get(decision.source, 0) + 1
            metadata.append(
                {
                    "run_name": run_dir.name,
                    "frame_index": int(record.get("frame_index", 0)),
                    "iter": int(record.get("iter", 0)),
                    "label": decision.step_id,
                    "label_source": decision.source,
                    "weight": float(decision.weight),
                    "fused_step": str(record.get("fused_step", "")),
                    "fused_conf": float(record.get("fused_conf", 0.0)),
                    "stable": bool(record.get("stable", False)),
                    "signature_key": list(frame.signature_key),
                }
            )

    if not vectors:
        raise RuntimeError("No temporal training samples could be exported from the provided runs.")

    tensor_x = torch.tensor(np.stack(vectors), dtype=torch.float32)
    tensor_y = torch.tensor(labels, dtype=torch.long)
    tensor_w = torch.tensor(weights, dtype=torch.float32)
    tensor_aux_mask = torch.tensor(aux_masks, dtype=torch.float32)
    tensor_aux_bootstrap = torch.tensor(np.stack(aux_next_bootstrap), dtype=torch.float32)
    tensor_aux_relations = torch.tensor(np.stack(aux_next_relations), dtype=torch.float32)
    tensor_aux_flags = torch.tensor(np.stack(aux_next_flags), dtype=torch.float32)
    label_counts = Counter(int(value) for value in labels)
    dataset = {
        "steps": steps,
        "component_names": component_names,
        "relation_types": list(vectorizer.relation_types),
        "flag_names": list(vectorizer.FLAG_NAMES),
        "window_size": int(window_size),
        "feature_dim": int(tensor_x.shape[-1]),
        "x": tensor_x,
        "y": tensor_y,
        "weights": tensor_w,
        "aux_mask": tensor_aux_mask,
        "aux_next_bootstrap": tensor_aux_bootstrap,
        "aux_next_relations": tensor_aux_relations,
        "aux_next_flags": tensor_aux_flags,
        "metadata": metadata,
        "run_names": run_names,
        "label_source_counts": source_counts,
        "label_counts": {steps[index]: int(count) for index, count in sorted(label_counts.items())},
        "skipped_counts": skipped_counts,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)
    return {
        "output_path": str(output_path),
        "num_samples": int(tensor_x.shape[0]),
        "window_size": int(window_size),
        "feature_dim": int(tensor_x.shape[-1]),
        "label_source_counts": source_counts,
        "label_counts": dataset["label_counts"],
        "skipped_counts": skipped_counts,
    }


def _iter_iteration_records(run_dir: Path) -> Iterator[dict[str, object]]:
    path = run_dir / "iterations.jsonl"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_feedback_map(run_dir: Path) -> Dict[int, LabelDecision]:
    path = run_dir / "feedback.jsonl"
    mapping: Dict[int, LabelDecision] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            frame_index = int(item.get("frame_index", 0) or 0)
            feedback = dict(item.get("feedback") or {})
            label = str(feedback.get("label", "")).strip().upper()
            if frame_index <= 0 or not label:
                continue
            accepted = bool(feedback.get("accepted", False))
            source = str(feedback.get("source", "feedback")).strip().lower()
            if accepted:
                weight = 0.95
                label_source = f"{source}_accept"
            else:
                weight = 1.00
                label_source = f"{source}_correct"
            mapping[frame_index] = LabelDecision(step_id=label, source=label_source, weight=weight)
    return mapping


def _mine_run_labels(
    frames: Sequence[RunFrame],
    feedback_map: Dict[int, LabelDecision],
    steps: Sequence[str],
    min_stable_confidence: float,
    min_auto_confidence: float,
    min_visual_confidence: float,
    min_memory_confidence: float,
    min_segment_length: int,
    min_segment_vote_ratio: float,
    min_segment_margin: float,
) -> Dict[int, LabelDecision]:
    decisions: Dict[int, LabelDecision] = {}
    blocked: set[int] = set()

    # First pass: per-frame strong supervision only.
    for frame in frames:
        decision, should_block = _choose_strong_frame_label(
            frame.record,
            feedback_map=feedback_map,
            min_stable_confidence=min_stable_confidence,
            min_auto_confidence=min_auto_confidence,
            min_memory_confidence=min_memory_confidence,
        )
        if should_block:
            blocked.add(frame.index)
        if decision is not None:
            decisions[frame.index] = decision

    # Second pass: segment-level consensus on contiguous visual signatures.
    segment_labels = _mine_segment_labels(
        frames,
        existing=decisions,
        blocked=blocked,
        steps=steps,
        min_segment_length=min_segment_length,
        min_segment_vote_ratio=min_segment_vote_ratio,
        min_segment_margin=min_segment_margin,
    )
    for index, decision in segment_labels.items():
        decisions.setdefault(index, decision)

    # Final pass: weak frame labels for recall-heavy runs with little supervision.
    for frame in frames:
        if frame.index in decisions or frame.index in blocked:
            continue
        decision = _choose_weak_frame_label(
            frame.record,
            min_visual_confidence=min_visual_confidence,
            min_memory_confidence=min_memory_confidence,
        )
        if decision is not None:
            decisions[frame.index] = decision
    return decisions


def _choose_strong_frame_label(
    record: dict[str, object],
    feedback_map: Dict[int, LabelDecision],
    min_stable_confidence: float,
    min_auto_confidence: float,
    min_memory_confidence: float,
) -> tuple[Optional[LabelDecision], bool]:
    frame_index = int(record.get("frame_index", 0) or 0)
    if frame_index in feedback_map:
        return feedback_map[frame_index], False

    review_action = str(record.get("review_action", "")).strip().lower()
    review_label = str(record.get("review_label", "")).strip().upper()
    fused_step = str(record.get("fused_step", "")).strip().upper()
    fused_conf = float(record.get("fused_conf", 0.0) or 0.0)
    has_visual_evidence = bool(record.get("has_visual_evidence", False))
    stable = bool(record.get("stable", False))

    if not has_visual_evidence:
        return None, False
    if review_action == "prefer_candidate" and review_label:
        return LabelDecision(step_id=review_label, source="review_prefer", weight=0.82), False
    if review_action in {"hold", "request_human"}:
        return None, True
    if stable and fused_step and fused_conf >= min_stable_confidence:
        source = "stable_auto" if fused_conf >= min_auto_confidence else "stable_soft"
        weight = 0.60 if source == "stable_auto" else 0.45
        return LabelDecision(step_id=fused_step, source=source, weight=weight), False

    memory_label = _memory_consensus_label(record, min_memory_confidence=min_memory_confidence, strong_only=True)
    if memory_label is not None:
        return memory_label, False
    return None, False


def _choose_weak_frame_label(
    record: dict[str, object],
    min_visual_confidence: float,
    min_memory_confidence: float,
) -> Optional[LabelDecision]:
    has_visual_evidence = bool(record.get("has_visual_evidence", False))
    fused_step = str(record.get("fused_step", "")).strip().upper()
    fused_conf = float(record.get("fused_conf", 0.0) or 0.0)

    if not has_visual_evidence:
        return None
    memory_label = _memory_consensus_label(record, min_memory_confidence=min_memory_confidence, strong_only=False)
    if memory_label is not None:
        return memory_label
    if fused_step and fused_conf >= min_visual_confidence:
        return LabelDecision(step_id=fused_step, source="visual_pseudo", weight=0.18)
    return None


def _memory_consensus_label(
    record: dict[str, object],
    min_memory_confidence: float,
    strong_only: bool,
) -> Optional[LabelDecision]:
    if not bool(record.get("memory_active", False)):
        return None
    memory_step = str(record.get("memory_step", "")).strip().upper()
    memory_conf = float(record.get("memory_conf", 0.0) or 0.0)
    if not memory_step or memory_conf < min_memory_confidence:
        return None

    fused_step = str(record.get("fused_step", "")).strip().upper()
    state_step = str(record.get("state_step", "")).strip().upper()
    retrieval_step = str(record.get("retrieval_step", "")).strip().upper()
    aligned_sources = sum(1 for step_id in (fused_step, state_step, retrieval_step) if step_id and step_id == memory_step)
    if strong_only and aligned_sources < 1:
        return None
    if not strong_only and aligned_sources < 1 and memory_conf < (min_memory_confidence + 0.08):
        return None
    weight = 0.58 if aligned_sources >= 2 else 0.52
    source = "memory_consensus" if strong_only else "memory_consensus_soft"
    return LabelDecision(step_id=memory_step, source=source, weight=weight)


def _mine_segment_labels(
    frames: Sequence[RunFrame],
    existing: Dict[int, LabelDecision],
    blocked: set[int],
    steps: Sequence[str],
    min_segment_length: int,
    min_segment_vote_ratio: float,
    min_segment_margin: float,
) -> Dict[int, LabelDecision]:
    decisions: Dict[int, LabelDecision] = {}
    previous_segment_step: Optional[str] = None
    for segment in _iter_signature_segments(frames):
        decision = _segment_decision(
            segment,
            existing=existing,
            steps=steps,
            previous_segment_step=previous_segment_step,
            min_segment_length=min_segment_length,
            min_segment_vote_ratio=min_segment_vote_ratio,
            min_segment_margin=min_segment_margin,
        )
        if decision is None:
            continue
        previous_segment_step = decision.step_id
        for frame in segment:
            if frame.index in existing or frame.index in blocked:
                continue
            decisions[frame.index] = decision
    return decisions


def _iter_signature_segments(frames: Sequence[RunFrame]) -> Iterator[List[RunFrame]]:
    segment: List[RunFrame] = []
    previous_signature: Optional[Tuple[Tuple[str, int], ...]] = None
    previous_frame_index: Optional[int] = None
    for frame in frames:
        frame_index = int(frame.record.get("frame_index", 0) or 0)
        if (
            not frame.token.has_visual_evidence
            or not frame.signature_key
            or (previous_signature is not None and frame.signature_key != previous_signature)
            or (
                previous_frame_index is not None
                and frame_index > previous_frame_index + 6
            )
        ):
            if segment:
                yield segment
                segment = []
            previous_signature = None
            previous_frame_index = None
            if not frame.token.has_visual_evidence or not frame.signature_key:
                continue
        if not segment:
            previous_signature = frame.signature_key
        segment.append(frame)
        previous_signature = frame.signature_key
        previous_frame_index = frame_index
    if segment:
        yield segment


def _segment_decision(
    segment: Sequence[RunFrame],
    existing: Dict[int, LabelDecision],
    steps: Sequence[str],
    previous_segment_step: Optional[str],
    min_segment_length: int,
    min_segment_vote_ratio: float,
    min_segment_margin: float,
) -> Optional[LabelDecision]:
    if len(segment) < max(2, int(min_segment_length)):
        return None

    vote_totals = {step_id: 0.0 for step_id in steps}
    step_frame_support = {step_id: 0 for step_id in steps}
    anchored_counts: Counter[str] = Counter()
    for frame in segment:
        if frame.index in existing:
            anchored_counts[existing[frame.index].step_id] += 1
        frame_votes = _frame_vote_scores(frame.record, steps=steps)
        best_step = max(frame_votes, key=frame_votes.get) if frame_votes else ""
        if best_step and float(frame_votes.get(best_step, 0.0)) > 0.0:
            step_frame_support[best_step] = int(step_frame_support.get(best_step, 0)) + 1
        for step_id, value in frame_votes.items():
            vote_totals[step_id] = float(vote_totals.get(step_id, 0.0) + float(value))

    if anchored_counts:
        for step_id, count in anchored_counts.items():
            vote_totals[step_id] = float(vote_totals.get(step_id, 0.0) + 0.90 * float(count))

    ranked = sorted(((step_id, float(score)) for step_id, score in vote_totals.items()), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] <= 0.0:
        return None
    best_step, best_score = ranked[0]
    second_score = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    total_score = float(sum(vote_totals.values()))
    if total_score <= 0.0:
        return None

    vote_ratio = float(best_score) / float(total_score)
    margin = float(best_score - second_score) / float(max(1e-6, best_score))
    support_ratio = float(step_frame_support.get(best_step, 0)) / float(max(1, len(segment)))
    if vote_ratio < min_segment_vote_ratio or margin < min_segment_margin or support_ratio < 0.50:
        return None

    anchored_majority = anchored_counts.most_common(1)[0][0] if anchored_counts else None
    if anchored_majority and anchored_majority != best_step:
        return None

    if previous_segment_step and previous_segment_step != best_step and _is_local_transition(previous_segment_step, best_step, steps):
        return LabelDecision(step_id=best_step, source="stable_transition", weight=0.68)
    return LabelDecision(step_id=best_step, source="segment_consensus", weight=0.56)


def _frame_vote_scores(record: dict[str, object], steps: Sequence[str]) -> Dict[str, float]:
    votes = {str(step_id).strip().upper(): 0.0 for step_id in steps}

    def add_vote(step_id: object, confidence: object, scale: float, minimum: float = 0.0) -> None:
        normalized = str(step_id or "").strip().upper()
        if normalized not in votes:
            return
        conf = float(confidence or 0.0)
        if conf <= 0.0 and minimum <= 0.0:
            return
        votes[normalized] = float(votes[normalized] + scale * max(conf, minimum))

    add_vote(record.get("state_step"), record.get("state_conf"), scale=1.00)
    add_vote(record.get("retrieval_step"), record.get("retrieval_conf"), scale=0.55)
    if bool(record.get("memory_active", False)):
        add_vote(record.get("memory_step"), record.get("memory_conf"), scale=0.70)
    add_vote(record.get("fused_step"), record.get("fused_conf"), scale=0.35)

    exact_steps = [
        str(record.get("state_step") or "").strip().upper(),
        str(record.get("retrieval_step") or "").strip().upper(),
        str(record.get("memory_step") or "").strip().upper() if bool(record.get("memory_active", False)) else "",
        str(record.get("fused_step") or "").strip().upper(),
    ]
    counts = Counter(step_id for step_id in exact_steps if step_id in votes)
    for step_id, count in counts.items():
        if count >= 2:
            votes[step_id] = float(votes.get(step_id, 0.0) + 0.08 * float(count - 1))
    return votes


def _is_local_transition(previous_step: str, current_step: str, steps: Sequence[str]) -> bool:
    normalized_prev = str(previous_step).strip().upper()
    normalized_current = str(current_step).strip().upper()
    if normalized_prev == normalized_current:
        return True
    if normalized_prev not in steps or normalized_current not in steps:
        return False
    return abs(steps.index(normalized_current) - steps.index(normalized_prev)) <= 1


def _extract_token(
    record: dict[str, object],
    steps: Sequence[str],
    prev_step: Optional[str],
) -> Optional[EvidenceToken]:
    token_payload = record.get("evidence_token")
    if isinstance(token_payload, dict):
        return _token_from_payload(token_payload, steps=steps, prev_step=prev_step)
    return _reconstruct_token(record, steps=steps, prev_step=prev_step)


def _signature_key(record: dict[str, object], token: EvidenceToken) -> Tuple[Tuple[str, int], ...]:
    signature = record.get("signature")
    if isinstance(signature, list):
        normalized: List[Tuple[str, int]] = []
        for item in signature:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                name, count = item
                name_text = str(name).strip().lower()
                if name_text:
                    normalized.append((name_text, int(count)))
        if normalized:
            return tuple(sorted(normalized))
    counts = token.relevant_counts or token.visible_counts
    return tuple(sorted((str(name).strip().lower(), int(count)) for name, count in counts.items() if str(name).strip()))


def _token_from_payload(
    payload: dict[str, object],
    steps: Sequence[str],
    prev_step: Optional[str],
) -> EvidenceToken:
    return EvidenceToken(
        frame_index=int(payload.get("frame_index", 0) or 0),
        prev_step=str(payload.get("prev_step") or prev_step or "").strip().upper() or prev_step,
        visible_counts={str(key).strip().lower(): int(value) for key, value in dict(payload.get("visible_counts") or {}).items()},
        relevant_counts={str(key).strip().lower(): int(value) for key, value in dict(payload.get("relevant_counts") or {}).items()},
        relation_counts={str(key).strip().lower(): int(value) for key, value in dict(payload.get("relation_counts") or {}).items()},
        relation_facts=[],
        state_scores=_dense_scores(payload.get("state_scores"), steps),
        retrieval_scores=_dense_scores(payload.get("retrieval_scores"), steps),
        memory_scores=_dense_scores(payload.get("memory_scores"), steps),
        state_confidence=float(payload.get("state_confidence", 0.0) or 0.0),
        retrieval_confidence=float(payload.get("retrieval_confidence", 0.0) or 0.0),
        memory_confidence=float(payload.get("memory_confidence", 0.0) or 0.0),
        memory_active=bool(payload.get("memory_active", False)),
        memory_reason=str(payload.get("memory_reason", "")).strip(),
        review_action=str(payload.get("review_action", "")).strip(),
        review_reason=str(payload.get("review_reason", "")).strip(),
        has_visual_evidence=bool(payload.get("has_visual_evidence", False)),
    )


def _reconstruct_token(
    record: dict[str, object],
    steps: Sequence[str],
    prev_step: Optional[str],
) -> Optional[EvidenceToken]:
    visible_counts = _count_from_detections(record.get("fused_detections"))
    relevant_counts = _count_from_detections(record.get("relevant_detections"))
    relation_counts = _relation_counts_from_record(record)
    has_visual_evidence = bool(
        record.get("has_visual_evidence", False)
        or visible_counts
        or relevant_counts
        or any(value > 0 for value in relation_counts.values())
        or int(record.get("num_fused", 0) or 0) > 0
        or int(record.get("num_relevant", 0) or 0) > 0
    )
    return EvidenceToken(
        frame_index=int(record.get("frame_index", 0) or 0),
        prev_step=str(prev_step or "").strip().upper() or None,
        visible_counts=visible_counts,
        relevant_counts=relevant_counts,
        relation_counts=relation_counts,
        relation_facts=[],
        state_scores=_dense_scores(record.get("state_scores"), steps, fallback_step=record.get("state_step"), fallback_conf=record.get("state_conf")),
        retrieval_scores=_dense_scores(record.get("retrieval_scores"), steps, fallback_step=record.get("retrieval_step"), fallback_conf=record.get("retrieval_conf")),
        memory_scores=_dense_scores(record.get("memory_scores"), steps, fallback_step=record.get("memory_step"), fallback_conf=record.get("memory_conf")),
        state_confidence=float(record.get("state_conf", 0.0) or 0.0),
        retrieval_confidence=float(record.get("retrieval_conf", 0.0) or 0.0),
        memory_confidence=float(record.get("memory_conf", 0.0) or 0.0),
        memory_active=bool(record.get("memory_active", False)),
        memory_reason=str(record.get("memory_reason", "")).strip(),
        review_action=str(record.get("review_action", "")).strip(),
        review_reason=str(record.get("review_reason", "")).strip(),
        has_visual_evidence=has_visual_evidence,
    )


def _dense_scores(
    payload: object,
    steps: Sequence[str],
    fallback_step: object = None,
    fallback_conf: object = None,
) -> Dict[str, float]:
    if isinstance(payload, dict) and payload:
        return {str(step).strip().upper(): float(payload.get(str(step).strip().upper(), 0.0) or 0.0) for step in steps}
    dense = {str(step).strip().upper(): 0.0 for step in steps}
    step_id = str(fallback_step or "").strip().upper()
    if step_id in dense:
        dense[step_id] = float(fallback_conf or 0.0)
    return dense


def _count_from_detections(payload: object) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not isinstance(payload, list):
        return counts
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip().lower()
        if not name:
            continue
        counts[name] = int(counts.get(name, 0)) + 1
    return counts


def _relation_counts_from_record(record: dict[str, object]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    stats = dict(record.get("scene_graph_stats") or {})
    for key, value in stats.items():
        if str(key).startswith("rel_"):
            counts[str(key)[4:].strip().lower()] = int(value)
    relations = record.get("scene_graph_relations")
    if isinstance(relations, list):
        for item in relations:
            if not isinstance(item, dict):
                continue
            predicate = str(item.get("predicate", "")).strip().lower()
            if not predicate:
                continue
            counts[predicate] = int(counts.get(predicate, 0)) + 1
    return counts


def _left_padded_window(tokens: List[np.ndarray], index: int, window_size: int) -> np.ndarray:
    current = tokens[index]
    feature_dim = int(current.shape[-1])
    window = np.zeros((window_size, feature_dim), dtype=np.float32)
    start = max(0, index - window_size + 1)
    slice_tokens = tokens[start : index + 1]
    window[-len(slice_tokens) :] = np.stack(slice_tokens)
    return window
