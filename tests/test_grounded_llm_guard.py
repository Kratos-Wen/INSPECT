import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from inspect_runtime.assistant.grounded_llm import _grounding_violation
from inspect_runtime.assistant.types import AssistantEvidence, AssistantSnapshot, ResolvedAssistantQuery


def snapshot(step_id: str = "S4") -> AssistantSnapshot:
    return AssistantSnapshot(
        frame_index=10,
        step_id=step_id,
        step_confidence=0.9,
        runner_up="S3",
        visible_objects=["type_5_gearbox_cover", "type_5_gearbox_housing"],
        relevant_objects=["type_5_gearbox_cover"],
        object_counts={"type_5_gearbox_cover": 1, "type_5_gearbox_housing": 1},
        scene_relations=[],
        memory_step=step_id,
        memory_confidence=0.9,
    )


def query(intent: str) -> ResolvedAssistantQuery:
    return ResolvedAssistantQuery(text="test", intent=intent)


def test_current_step_generation_cannot_change_verified_step() -> None:
    evidence = AssistantEvidence(intent="current_step", facts={"step_id": "S4"})
    violation = _grounding_violation(
        "The current step is S1.",
        resolved=query("current_step"),
        evidence=evidence,
        snapshot=snapshot(),
    )
    assert violation == "unsupported_step_reference"


def test_next_step_generation_preserves_current_and_next_step() -> None:
    evidence = AssistantEvidence(
        intent="next_step",
        facts={"current_step": "S2", "next_step": "S3", "missing_for_next": ["type_8_gear"]},
    )
    violation = _grounding_violation(
        "After S2, proceed to S3 when Type 8 Gear is available.",
        resolved=query("next_step"),
        evidence=evidence,
        snapshot=snapshot("S2"),
    )
    assert violation == ""


def test_generation_cannot_invent_component_identifier() -> None:
    evidence = AssistantEvidence(
        intent="object_presence",
        facts={"target_component": "type_3_gear", "visible": True},
    )
    violation = _grounding_violation(
        "Type 7 Gear is visible.",
        resolved=query("object_presence"),
        evidence=evidence,
        snapshot=snapshot("S2"),
    )
    assert violation == "unsupported_component_reference"
