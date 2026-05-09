"""Evaluation helpers for INSPECT artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, Sequence

from .types import RobotObservation, VerificationResult, VerifiedTraceEvent


def summarize_traces(traces: Iterable[VerifiedTraceEvent]) -> Dict[str, object]:
    trace_list = list(traces)
    sources = Counter(event.verification_source for event in trace_list)
    levels = Counter(event.verification_level for event in trace_list)
    verified = [event for event in trace_list if event.verified]
    strong = [event for event in trace_list if event.verification_level in {"L2", "L3", "L4"}]
    evidence_counts = [len(event.observed_evidence) for event in trace_list]
    return {
        "num_events": len(trace_list),
        "num_verified": len(verified),
        "num_strong_verified": len(strong),
        "verified_fraction": (len(verified) / len(trace_list) if trace_list else 0.0),
        "verification_sources": dict(sources),
        "verification_levels": dict(levels),
        "mean_trust_weight": (sum(event.trust_weight for event in trace_list) / len(trace_list) if trace_list else 0.0),
        "mean_observed_evidence": (sum(evidence_counts) / len(evidence_counts) if evidence_counts else 0.0),
        "num_human_corrections": int(sources.get("human_correction", 0)),
        "num_human_accepts": int(sources.get("human_accept", 0)),
        "num_reviewer_corrections": int(sources.get("reviewer_correction", 0)),
        "num_stable_auto": int(sources.get("stable_auto", 0)),
    }


def _truth_state(observation: RobotObservation) -> str:
    meta = observation.metadata
    for key in ("ground_truth_state", "verified_state", "label", "state", "current_state"):
        value = meta.get(key)
        if value:
            return str(value).upper()
    return ""


def _truth_bool(observation: RobotObservation, keys: Sequence[str]) -> bool | None:
    for key in keys:
        if key not in observation.metadata:
            continue
        value = observation.metadata[key]
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "allowed", "valid"}:
            return True
        if text in {"0", "false", "no", "blocked", "invalid"}:
            return False
    return None


def evaluate_verification_results(
    results: Sequence[VerificationResult],
    observations: Sequence[RobotObservation],
) -> Dict[str, object]:
    """Compute paper-facing verification metrics when labels are available."""

    by_id = {observation.observation_id: observation for observation in observations}
    labeled = []
    for result in results:
        observation = by_id.get(result.observation_id)
        if observation is None:
            continue
        truth = _truth_state(observation)
        if truth:
            labeled.append((result, observation, truth))

    correct = sum(1 for result, _, truth in labeled if result.predicted_state == truth)
    state_accuracy = correct / len(labeled) if labeled else None

    labels = sorted({truth for _, _, truth in labeled} | {result.predicted_state for result, _, _ in labeled})
    f1_by_state: Dict[str, float] = {}
    for label in labels:
        tp = sum(1 for result, _, truth in labeled if result.predicted_state == label and truth == label)
        fp = sum(1 for result, _, truth in labeled if result.predicted_state == label and truth != label)
        fn = sum(1 for result, _, truth in labeled if result.predicted_state != label and truth == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_by_state[label] = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    macro_f1 = sum(f1_by_state.values()) / len(f1_by_state) if f1_by_state else None

    admissibility_total = 0
    admissibility_correct = 0
    invalid_continuations = 0
    unnecessary_pauses = 0
    anomaly_total = 0
    anomaly_correct = 0
    action_counts = Counter(result.recommended_action for result in results)
    missing_counts: Counter[str] = Counter()
    for result in results:
        observation = by_id.get(result.observation_id)
        if observation is None:
            continue
        for key in result.missing_evidence:
            missing_counts[key] += 1
        truth_allowed = _truth_bool(observation, ("next_step_admissible", "next_step_allowed", "valid_next_step"))
        if truth_allowed is not None:
            admissibility_total += 1
            if result.next_step_admissible == truth_allowed:
                admissibility_correct += 1
            if result.next_step_admissible and not truth_allowed:
                invalid_continuations += 1
            if (not result.next_step_admissible) and truth_allowed and result.recommended_action in {"pause", "ask_human"}:
                unnecessary_pauses += 1
        truth_anomaly = _truth_bool(observation, ("anomaly", "is_anomaly", "failure", "invalid_state"))
        if truth_anomaly is not None:
            anomaly_total += 1
            if result.anomaly == truth_anomaly:
                anomaly_correct += 1

    return {
        "num_results": len(results),
        "num_labeled_states": len(labeled),
        "state_accuracy": state_accuracy,
        "state_macro_f1": macro_f1,
        "state_f1": f1_by_state,
        "admissibility_accuracy": (admissibility_correct / admissibility_total if admissibility_total else None),
        "invalid_continuations": invalid_continuations,
        "unnecessary_pauses": unnecessary_pauses,
        "anomaly_accuracy": (anomaly_correct / anomaly_total if anomaly_total else None),
        "action_counts": dict(action_counts),
        "top_missing_evidence": dict(missing_counts.most_common(20)),
    }
