"""Procedural Evidence Graph induction for INSPECT."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .types import EvidenceEdge, EvidenceNode, ProceduralEvidenceGraph, VerifiedTraceEvent


def _state_node(state_id: str) -> str:
    return f"state:{state_id.upper()}"


def _evidence_node(evidence_key: str) -> str:
    return f"evidence:{evidence_key}"


def _failure_node(cue: str) -> str:
    return f"failure:{cue}"


def _source_trust(event: VerifiedTraceEvent) -> float:
    if event.trust_weight > 0:
        return float(event.trust_weight)
    if event.verification_source == "human_correction":
        return 1.0
    if event.verification_source == "human_accept":
        return 0.9
    if event.verification_source == "reviewer_correction":
        return 0.75
    if event.verification_source == "stable_auto":
        return 0.10
    return 0.0


def _strong_verification(event: VerifiedTraceEvent) -> bool:
    return bool(event.verified and event.verification_level in {"L2", "L3", "L4"} and _source_trust(event) >= 0.7)


def _evidence_priority(key: str) -> float:
    if key.startswith("relation:"):
        return 1.0
    if key.startswith("focus_object:"):
        return 0.85
    if key.startswith("relation_type:"):
        return 0.75
    if key.startswith("memory_match:"):
        return 0.55
    if key.startswith("object:"):
        return 0.5
    if key.startswith("visual:"):
        return 0.25
    return 0.4


def _state_priors(counts: Counter[str]) -> Dict[str, float]:
    total = float(sum(counts.values()))
    if total <= 0:
        return {}
    return {state: count / total for state, count in sorted(counts.items())}


def save_evidence_graph(graph: ProceduralEvidenceGraph, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_evidence_graph(path: Path) -> ProceduralEvidenceGraph:
    return ProceduralEvidenceGraph.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def induce_evidence_graph(
    traces: Iterable[VerifiedTraceEvent],
    graph_id: str = "inspect_procedural_evidence_graph",
    min_support: float = 0.15,
    topk_per_state: int = 24,
) -> ProceduralEvidenceGraph:
    """Induce a view-agnostic procedural evidence graph from trace events."""

    trace_list = list(traces)
    support: Dict[Tuple[str, str, str], float] = defaultdict(float)
    counts: Counter[Tuple[str, str, str]] = Counter()
    weak_support: Dict[Tuple[str, str], float] = defaultdict(float)
    weak_counts: Counter[Tuple[str, str]] = Counter()
    state_counts: Counter[str] = Counter()
    strong_state_counts: Counter[str] = Counter()
    transitions: Counter[Tuple[str, str]] = Counter()
    contradictions: Counter[Tuple[str, str]] = Counter()
    uncertain_missing: Counter[Tuple[str, str]] = Counter()

    for event in trace_list:
        predicted = event.predicted_state
        verified = event.candidate_state or event.verified_state
        if event.verified and verified:
            state_counts[verified] += 1
            if _strong_verification(event):
                strong_state_counts[verified] += 1
            if event.prev_state and event.prev_state != verified:
                transitions[(event.prev_state, verified)] += 1
            trust = _source_trust(event)
            conf = max(0.2, min(1.0, float(event.confidence)))
            if _strong_verification(event):
                evidence_groups = [
                    ("requires", event.precondition_evidence),
                    ("supported_by_interaction", event.interaction_evidence),
                    ("supported_by_transition", event.transition_evidence),
                    ("verified_by", event.postcondition_evidence),
                    ("admissible_by", event.admissibility_evidence),
                ]
                for predicate, keys in evidence_groups:
                    for key in keys:
                        weight = trust * conf * _evidence_priority(key)
                        support[(verified, predicate, key)] += weight
                        counts[(verified, predicate, key)] += 1
                for key in event.negative_evidence:
                    contradictions[(verified, key)] += 1
            else:
                for key in event.observed_evidence:
                    weight = max(0.01, trust * conf * _evidence_priority(key))
                    weak_support[(verified, key)] += weight
                    weak_counts[(verified, key)] += 1
            if predicted and predicted != verified:
                for cue in event.failure_cues + event.negative_evidence:
                    contradictions[(predicted, cue)] += 1
        else:
            if predicted:
                for key in event.missing_evidence:
                    uncertain_missing[(predicted, key)] += 1

    nodes: Dict[str, EvidenceNode] = {}
    edges: List[EvidenceEdge] = []
    state_ids = sorted(state_counts.keys())
    for state_id in state_ids:
        node_id = _state_node(state_id)
        nodes[node_id] = EvidenceNode(
            node_id=node_id,
            kind="state",
            label=state_id,
            support=float(state_counts[state_id]),
            weight=float(state_counts[state_id]),
        )

    by_state_predicate: Dict[Tuple[str, str], List[Tuple[str, float]]] = defaultdict(list)
    for (state_id, predicate, key), weight in support.items():
        if weight >= min_support:
            by_state_predicate[(state_id, predicate)].append((key, weight))

    for (state_id, predicate), items in by_state_predicate.items():
        items = sorted(items, key=lambda item: item[1], reverse=True)[:topk_per_state]
        max_weight = max((weight for _, weight in items), default=1.0)
        for key, raw_weight in items:
            node_id = _evidence_node(key)
            normalized = raw_weight / max_weight if max_weight > 0 else raw_weight
            if node_id not in nodes:
                nodes[node_id] = EvidenceNode(
                    node_id=node_id,
                    kind=key.split(":", 1)[0],
                    label=key,
                    support=float(raw_weight),
                    weight=float(normalized),
                )
            edges.append(
                EvidenceEdge(
                    source=_state_node(state_id),
                    predicate=predicate,
                    target=node_id,
                    weight=float(normalized),
                    count=int(counts[(state_id, predicate, key)]),
                    metadata={"raw_support": float(raw_weight), "trust_gate": "strong_only"},
                )
            )

    for (state_id, key), raw_weight in weak_support.items():
        if raw_weight < min_support * 0.25:
            continue
        source = _state_node(state_id)
        target = _evidence_node(key)
        if source not in nodes:
            nodes[source] = EvidenceNode(node_id=source, kind="state", label=state_id, support=0.0, weight=0.0)
        if target not in nodes:
            nodes[target] = EvidenceNode(
                node_id=target,
                kind=key.split(":", 1)[0],
                label=key,
                support=float(raw_weight),
                weight=0.0,
                metadata={"weak_prior_only": True},
            )
        edges.append(
            EvidenceEdge(
                source=source,
                predicate="suggested_by",
                target=target,
                weight=float(raw_weight),
                count=int(weak_counts[(state_id, key)]),
                metadata={"trust_gate": "weak_prior_only"},
            )
        )

    for (prev_state, next_state), count in transitions.items():
        source = _state_node(prev_state)
        target = _state_node(next_state)
        if source not in nodes:
            nodes[source] = EvidenceNode(node_id=source, kind="state", label=prev_state, support=float(count), weight=float(count))
        if target not in nodes:
            nodes[target] = EvidenceNode(node_id=target, kind="state", label=next_state, support=float(count), weight=float(count))
        edges.append(
            EvidenceEdge(
                source=source,
                predicate="enables",
                target=target,
                weight=float(count),
                count=int(count),
            )
        )

    for (state_id, cue), count in contradictions.items():
        source = _state_node(state_id)
        target = _failure_node(cue)
        if source not in nodes:
            nodes[source] = EvidenceNode(node_id=source, kind="state", label=state_id, support=0.0, weight=0.0)
        nodes[target] = EvidenceNode(node_id=target, kind="failure", label=cue, support=float(count), weight=float(count))
        edges.append(
            EvidenceEdge(
                source=source,
                predicate="contradicted_by",
                target=target,
                weight=float(count),
                count=int(count),
            )
        )

    for (state_id, key), count in uncertain_missing.items():
        source = _state_node(state_id)
        target = _evidence_node(key)
        if source not in nodes:
            nodes[source] = EvidenceNode(node_id=source, kind="state", label=state_id, support=0.0, weight=0.0)
        if target not in nodes:
            nodes[target] = EvidenceNode(node_id=target, kind=key.split(":", 1)[0], label=key, support=float(count), weight=0.0)
        edges.append(
            EvidenceEdge(
                source=source,
                predicate="missing_when_uncertain",
                target=target,
                weight=float(count),
                count=int(count),
            )
        )

    all_states = sorted({node.label for node in nodes.values() if node.kind == "state"})
    return ProceduralEvidenceGraph(
        graph_id=graph_id,
        state_ids=all_states,
        nodes=nodes,
        edges=edges,
        state_priors=_state_priors(state_counts),
        metadata={
            "num_trace_events": len(trace_list),
            "num_verified_events": sum(1 for event in trace_list if event.verified),
            "num_strong_verified_events": sum(1 for event in trace_list if _strong_verification(event)),
            "min_support": float(min_support),
            "topk_per_state": int(topk_per_state),
            "trust_gate": "L2/L3/L4 events create strong evidence edges; L0/L1 create weak suggested_by edges only.",
        },
    )


def summarize_evidence_graph(graph: ProceduralEvidenceGraph) -> Dict[str, object]:
    edge_counts = Counter(edge.predicate for edge in graph.edges)
    return {
        "graph_id": graph.graph_id,
        "states": len(graph.state_ids),
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "edge_counts": dict(edge_counts),
        "state_ids": list(graph.state_ids),
    }
