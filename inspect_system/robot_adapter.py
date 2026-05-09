"""Robot-side observation adapter and decision loop for INSPECT."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Iterable, List, Optional, Protocol, Sequence

from .active_observation import choose_view_for_missing_evidence
from .types import ProceduralEvidenceGraph, RobotObservation, VerificationResult, ViewCandidate
from .verifier import RobotProceduralStateVerifier


class RobotVerifier(Protocol):
    def verify(self, observation: RobotObservation) -> VerificationResult:
        """Return a procedural verification result for the robot observation."""


class RobotControlAdapter(Protocol):
    """Boundary that a ROS/MoveIt/robot-specific adapter should implement."""

    def observe(self, view_id: str = "") -> RobotObservation:
        """Capture and convert the current robot view into a RobotObservation."""

    def move_to_view(self, view_id: str) -> None:
        """Move camera/robot to a predefined safe observation viewpoint."""

    def continue_task(self, result: VerificationResult) -> None:
        """Allow the task planner to continue to the next skill/state."""

    def pause_task(self, result: VerificationResult) -> None:
        """Pause or hold the robot/task process."""

    def ask_human(self, result: VerificationResult) -> None:
        """Request human confirmation or intervention."""


def robot_observation_from_payload(payload: dict) -> RobotObservation:
    """Convert robot perception output into the INSPECT observation schema."""

    metadata = dict(payload.get("metadata", {}))
    for key in ("rgb_path", "depth_path", "camera_pose", "robot_state", "timestamp"):
        if key in payload and key not in metadata:
            metadata[key] = payload[key]
    return RobotObservation.from_dict({**payload, "metadata": metadata})


def robot_observation_from_evidence_token(
    observation_id: str,
    token: object,
    candidate_states: Optional[Iterable[str]] = None,
    view_id: str = "",
    metadata: Optional[dict] = None,
) -> RobotObservation:
    """Convert an EvidenceToken-like object from a robot perception stack."""

    return RobotObservation(
        observation_id=str(observation_id),
        frame_index=getattr(token, "frame_index", None),
        prev_state=getattr(token, "prev_step", None),
        view_id=str(view_id),
        visible_counts=dict(getattr(token, "visible_counts", {}) or {}),
        relevant_counts=dict(getattr(token, "relevant_counts", {}) or {}),
        relation_counts=dict(getattr(token, "relation_counts", {}) or {}),
        relation_facts=[list(item) for item in getattr(token, "relation_facts", []) or []],
        evidence_keys=[
            *(f"contact:{a}:{b}:{c}" for a, b, c in getattr(token, "contact_facts", []) or []),
            *(f"interaction:active_object:{name}" for name in dict(getattr(token, "contact_counts", {}) or {})),
        ],
        candidate_states=[str(item).upper() for item in (candidate_states or [])],
        metadata=dict(metadata or {}),
    )


class RobotProceduralDecisionLoop:
    """Reusable observe-verify-act loop for robot active observation."""

    def __init__(
        self,
        graph: ProceduralEvidenceGraph,
        robot: RobotControlAdapter,
        verifier: Optional[RobotVerifier] = None,
        view_candidates: Optional[Sequence[ViewCandidate]] = None,
        max_observation_moves: int = 2,
        on_result: Optional[Callable[[VerificationResult], None]] = None,
    ) -> None:
        self.graph = graph
        self.robot = robot
        self.verifier = verifier or RobotProceduralStateVerifier(graph)
        self.view_candidates = list(view_candidates or [])
        self.max_observation_moves = max(0, int(max_observation_moves))
        self.on_result = on_result

    def step(self, initial_view: str = "") -> VerificationResult:
        """Run one closed-loop procedural verification decision."""

        observation = self.robot.observe(initial_view)
        result = self.verifier.verify(observation)
        moves = 0
        visited: set[str] = set()
        while result.recommended_action == "observe" and self.view_candidates and moves < self.max_observation_moves:
            decision = choose_view_for_missing_evidence(result, self.view_candidates, self.graph)
            if not decision.selected_view or decision.selected_view in visited:
                break
            visited.add(decision.selected_view)
            self.robot.move_to_view(decision.selected_view)
            observation = self.robot.observe(decision.selected_view)
            result = self.verifier.verify(observation)
            result.metadata["active_observation_decision"] = decision.to_dict()
            moves += 1

        if self.on_result is not None:
            self.on_result(result)
        if result.recommended_action == "continue":
            self.robot.continue_task(result)
        elif result.recommended_action == "pause":
            self.robot.pause_task(result)
        elif result.recommended_action == "ask_human":
            self.robot.ask_human(result)
        else:
            self.robot.pause_task(result)
        return result


class RecordingRobotAdapter:
    """Test adapter for replaying precomputed robot observations."""

    def __init__(self, observations: Sequence[RobotObservation]) -> None:
        self.observations = list(observations)
        self.index = 0
        self.actions: List[dict] = []

    def observe(self, view_id: str = "") -> RobotObservation:
        if not self.observations:
            raise RuntimeError("RecordingRobotAdapter has no observations.")
        observation = self.observations[min(self.index, len(self.observations) - 1)]
        self.index += 1
        self.actions.append({"action": "observe", "view_id": view_id, "observation": observation.to_dict()})
        return observation

    def move_to_view(self, view_id: str) -> None:
        self.actions.append({"action": "move_to_view", "view_id": view_id})

    def continue_task(self, result: VerificationResult) -> None:
        self.actions.append({"action": "continue", "result": result.to_dict()})

    def pause_task(self, result: VerificationResult) -> None:
        self.actions.append({"action": "pause", "result": result.to_dict()})

    def ask_human(self, result: VerificationResult) -> None:
        self.actions.append({"action": "ask_human", "result": result.to_dict()})

    def action_log(self) -> List[dict]:
        return [dict(item) for item in self.actions]
