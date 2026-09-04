import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from inspect_runtime.config import EdgeRuntimeConfig
from inspect_runtime.runtime.edge_runtime import ComputeContext, EpistemicComputeRouter, StageProfiler


def test_stable_scene_uses_budgeted_perception_schedule() -> None:
    config = EdgeRuntimeConfig(
        detection_interval_stable=2,
        geometry_interval_stable=6,
        retrieval_interval_stable=3,
    )
    plan = EpistemicComputeRouter(config).plan(
        ComputeContext(
            frame_index=5,
            claim_state="supported",
            step_confidence=0.9,
            stable_tracks=True,
            visual_change=False,
        )
    )
    assert not plan.run_detection
    assert not plan.run_geometry
    assert not plan.run_retrieval
    assert plan.response_tier == "none"


def test_missing_relation_evidence_forces_geometry_and_small_llm() -> None:
    plan = EpistemicComputeRouter(EdgeRuntimeConfig()).plan(
        ComputeContext(
            frame_index=5,
            claim_state="insufficient",
            missing_roles=("slot_relation", "containment"),
            query_intent="object_relation",
            step_confidence=0.8,
            stable_tracks=True,
            visual_change=False,
        )
    )
    assert plan.run_geometry
    assert plan.run_segmentation
    assert plan.response_tier == "small_llm"
    assert "missing_geometric_role" in plan.reasons


def test_identity_evidence_routes_detection_and_retrieval() -> None:
    plan = EpistemicComputeRouter(
        EdgeRuntimeConfig(
            cold_start_frames=0,
            detection_interval_stable=100,
            geometry_interval_stable=100,
            retrieval_interval_stable=100,
        )
    ).plan(
        ComputeContext(
            frame_index=5,
            claim_state="insufficient",
            missing_roles=("identity_disambiguation",),
            missing_role_scores=(("identity_disambiguation", 0.8),),
            step_confidence=0.8,
            stable_tracks=True,
            visual_change=False,
        )
    )
    assert plan.run_detection
    assert plan.run_retrieval
    assert not plan.run_geometry
    assert plan.acquisition_mode == "internal_compute"


def test_persistent_missing_role_escalates_to_external_observation() -> None:
    router = EpistemicComputeRouter(
        EdgeRuntimeConfig(
            cold_start_frames=0,
            persistent_missing_patience=2,
            detection_interval_stable=100,
            geometry_interval_stable=100,
            retrieval_interval_stable=100,
            memory_embedding_interval_stable=100,
        )
    )
    context = dict(
        claim_state="insufficient",
        missing_roles=("slot_relation",),
        missing_role_scores=(("slot_relation", 0.9),),
        step_confidence=0.8,
        stable_tracks=True,
        visual_change=False,
    )
    first = router.plan(ComputeContext(frame_index=5, **context))
    second = router.plan(ComputeContext(frame_index=6, **context))
    third = router.plan(ComputeContext(frame_index=7, **context))
    assert first.run_geometry
    assert second.run_geometry
    assert not third.run_geometry
    assert third.external_observation_recommended
    assert third.acquisition_mode == "external_observation"


def test_no_epistemic_gain_escalates_even_while_egocentric_view_moves() -> None:
    router = EpistemicComputeRouter(
        EdgeRuntimeConfig(
            cold_start_frames=0,
            persistent_missing_patience=2,
            detection_interval_stable=100,
            geometry_interval_stable=100,
        )
    )
    context = dict(
        claim_state="insufficient",
        missing_roles=("slot_relation",),
        missing_role_scores=(("slot_relation", 0.9),),
        step_confidence=0.8,
        stable_tracks=False,
        visual_change=True,
        epistemic_gain=0.0,
    )

    router.plan(ComputeContext(frame_index=5, **context))
    router.plan(ComputeContext(frame_index=6, **context))
    third = router.plan(ComputeContext(frame_index=7, **context))

    assert third.external_observation_recommended
    assert not third.run_geometry


def test_resolved_contradiction_does_not_force_missing_role_compute() -> None:
    plan = EpistemicComputeRouter(
        EdgeRuntimeConfig(
            cold_start_frames=0,
            detection_interval_stable=100,
            geometry_interval_stable=100,
            retrieval_interval_stable=100,
            memory_embedding_interval_stable=100,
        )
    ).plan(
        ComputeContext(
            frame_index=5,
            claim_state="contradicted",
            missing_roles=("slot_relation", "identity_disambiguation"),
            stable_tracks=False,
            visual_change=False,
        )
    )

    assert not plan.run_geometry
    assert not plan.run_retrieval


def test_direct_state_query_uses_structured_response() -> None:
    plan = EpistemicComputeRouter(EdgeRuntimeConfig()).plan(
        ComputeContext(frame_index=4, query_intent="current_step")
    )
    assert plan.response_tier == "structured"


def test_disabled_profiler_has_no_payload() -> None:
    profiler = StageProfiler(enabled=False)
    with profiler.measure("stage"):
        pass
    assert profiler.finish() == {}
