"""End-to-end INSPECT artifact builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from .active_observation import choose_views_for_results, load_view_candidates, write_active_observation_jsonl
from .evidence_graph import induce_evidence_graph, save_evidence_graph, summarize_evidence_graph
from .learned_verifier import CalibratedEvidenceVerifier
from .metrics import evaluate_verification_results, summarize_traces
from .trace_export import export_verified_traces
from .verifier import load_robot_observations, verify_observations, write_verification_jsonl


def build_inspect_artifacts(
    run_dir: Path,
    output_dir: Optional[Path] = None,
    robot_observations_path: Optional[Path] = None,
    view_candidates_path: Optional[Path] = None,
    state_specs_path: Optional[Path] = None,
    verifier_model_path: Optional[Path] = None,
    min_auto_confidence: float = 0.78,
    graph_min_support: float = 0.15,
    topk_per_state: int = 24,
    verification_threshold: float = 0.55,
    anomaly_threshold: float = 0.30,
) -> Dict[str, object]:
    """Build trace, evidence graph, optional verifier, and optional view decisions."""

    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir / "inspect_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_path = output_dir / "verified_traces.jsonl"
    graph_path = output_dir / "procedural_evidence_graph.json"
    summary_path = output_dir / "inspect_summary.json"

    traces = export_verified_traces(
        run_dir=run_dir,
        output_path=trace_path,
        min_auto_confidence=min_auto_confidence,
        state_specs_path=state_specs_path,
    )
    graph = induce_evidence_graph(
        traces,
        graph_id=f"{run_dir.name}_inspect_evidence_graph",
        min_support=graph_min_support,
        topk_per_state=topk_per_state,
    )
    save_evidence_graph(graph, graph_path)

    summary: Dict[str, object] = {
        "system": "INSPECT",
        "system_full_name": "Interactive Supervision for Procedural Evidence and Cross-view Task Verification",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "trace_path": str(trace_path),
        "graph_path": str(graph_path),
        "state_specs_path": str(state_specs_path) if state_specs_path is not None else "",
        "num_trace_events": len(traces),
        "num_verified_trace_events": sum(1 for event in traces if event.verified),
        "trace_metrics": summarize_traces(traces),
        "graph": summarize_evidence_graph(graph),
    }

    verification_results = []
    if robot_observations_path is not None:
        observations = load_robot_observations(Path(robot_observations_path))
        if verifier_model_path is not None:
            learned_verifier = CalibratedEvidenceVerifier.load(Path(verifier_model_path))
            verification_results = [learned_verifier.verify(observation) for observation in observations]
            summary["verifier_model_path"] = str(verifier_model_path)
        else:
            verification_results = verify_observations(
                graph=graph,
                observations=observations,
                verification_threshold=verification_threshold,
                anomaly_threshold=anomaly_threshold,
            )
        verification_path = output_dir / "robot_verification.jsonl"
        write_verification_jsonl(verification_results, verification_path)
        summary["robot_observations_path"] = str(robot_observations_path)
        summary["verification_path"] = str(verification_path)
        summary["num_robot_observations"] = len(observations)
        summary["verification_metrics"] = evaluate_verification_results(verification_results, observations)

    if view_candidates_path is not None and verification_results:
        candidates = load_view_candidates(Path(view_candidates_path))
        decisions = choose_views_for_results(verification_results, candidates, graph)
        active_path = output_dir / "active_observation.jsonl"
        write_active_observation_jsonl(decisions, active_path)
        summary["view_candidates_path"] = str(view_candidates_path)
        summary["active_observation_path"] = str(active_path)
        summary["num_view_candidates"] = len(candidates)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
