"""Live camera assistant with OpenCV UI and optional voice control."""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..assistant import AssistantSnapshot, ContextualAssistant, GroundedLLMResponder
from ..assistant.types import FeedbackTimelineEvent, MemoryMatchSummary, StepTimelineEvent
from ..components import AssistantSpeechService, JsonlCsvLogger, OpenCVRuntimeUI, VoiceCommandService
from ..core_types import Detection, FeedbackEvent, RuntimeAction
from .pipeline import StepPipeline, build_default_pipeline


def _camera_backend_candidates(name: str) -> list[tuple[str, Optional[int]]]:
    normalized = str(name or "auto").strip().lower()
    lookup = {
        "default": ("default", None),
        "auto": ("default", None),
        "dshow": ("dshow", getattr(cv2, "CAP_DSHOW", None)),
        "msmf": ("msmf", getattr(cv2, "CAP_MSMF", None)),
        "v4l2": ("v4l2", getattr(cv2, "CAP_V4L2", None)),
        "ffmpeg": ("ffmpeg", getattr(cv2, "CAP_FFMPEG", None)),
    }
    ordered: list[tuple[str, Optional[int]]]
    if normalized in {"auto", ""}:
        ordered = [
            lookup["default"],
            lookup["dshow"],
            lookup["msmf"],
            lookup["v4l2"],
            lookup["ffmpeg"],
        ]
    else:
        ordered = [lookup.get(normalized, (normalized, None)), lookup["default"]]
    deduped: list[tuple[str, Optional[int]]] = []
    seen: set[tuple[str, Optional[int]]] = set()
    for item in ordered:
        if item[1] is None and item[0] not in {"default", "auto"} and normalized != item[0]:
            continue
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _open_camera_capture(index: int, camera_config, event_bus) -> tuple[cv2.VideoCapture, str, int]:
    last_error = f"Failed to open camera: {index}"
    for backend_name, backend_id in _camera_backend_candidates(getattr(camera_config, "backend", "auto")):
        capture = cv2.VideoCapture(index) if backend_id is None else cv2.VideoCapture(index, backend_id)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(camera_config.width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(camera_config.height))
        capture.set(cv2.CAP_PROP_FPS, float(camera_config.fps))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, float(camera_config.buffer_size))
        if capture.isOpened():
            warmup_ok = 0
            warmup_frames = max(0, int(getattr(camera_config, "warmup_frames", 0)))
            warmup_delay_sec = max(0.0, float(getattr(camera_config, "warmup_delay_ms", 0)) / 1000.0)
            for _ in range(warmup_frames):
                ok, frame_bgr = capture.read()
                if ok and frame_bgr is not None and getattr(frame_bgr, "size", 0) > 0:
                    warmup_ok += 1
                if warmup_delay_sec > 0:
                    time.sleep(warmup_delay_sec)
            event_bus.emit(
                "camera.warmup",
                {
                    "camera_index": index,
                    "backend": backend_name,
                    "requested_frames": warmup_frames,
                    "ready_frames": warmup_ok,
                },
            )
            if warmup_frames == 0 or warmup_ok > 0:
                event_bus.emit(
                    "camera.opened",
                    {
                        "camera_index": index,
                        "backend": backend_name,
                        "width": camera_config.width,
                        "height": camera_config.height,
                        "fps": camera_config.fps,
                        "ready_frames": warmup_ok,
                    },
                )
                return capture, backend_name, warmup_ok
            event_bus.emit(
                "camera.backend_rejected",
                {
                    "camera_index": index,
                    "backend": backend_name,
                    "reason": "warmup_empty",
                },
            )
        capture.release()
        last_error = f"Failed to open camera: {index} with backend={backend_name}"
        event_bus.emit(
            "camera.open_failed",
            {"camera_index": index, "backend": backend_name},
        )
    raise RuntimeError(last_error)

def _draw_overlay(frame_bgr: np.ndarray, detections: list[Detection], step_text: str) -> np.ndarray:
    canvas = frame_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection.xyxy]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"{detection.name} {detection.confidence:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        step_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (50, 200, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _feedback_log_payload(
    processed_index: int,
    frame_index: int,
    feedback: FeedbackEvent,
    fusion_result,
    memory_step: str,
    memory_conf: float,
) -> dict[str, object]:
    return {
        "iter": processed_index,
        "frame_index": frame_index,
        "feedback": {
            "label": feedback.label,
            "accepted": feedback.accepted,
            "strength": feedback.strength,
            "source": feedback.source,
            "note": feedback.note,
        },
        "before": {
            "fused_step": fusion_result.step_id,
            "fused_conf": fusion_result.confidence,
            "gate_state": float(fusion_result.gates.get("state", 0.0)),
            "gate_temporal": float(fusion_result.gates.get("temporal", 0.0)),
            "gate_retrieval": float(fusion_result.gates.get("retrieval", 0.0)),
            "gate_memory": float(fusion_result.gates.get("memory", 0.0)),
            "memory_step": memory_step,
            "memory_conf": memory_conf,
        },
    }


def _focus_crop(frame_bgr: np.ndarray, detections: list[Detection]) -> Optional[np.ndarray]:
    if not detections:
        return None
    x1, y1, x2, y2 = [int(value) for value in detections[0].xyxy]
    h, w = frame_bgr.shape[:2]
    pad = 10
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2].copy()


def _summarize_memory_matches(memory_recall, limit: int = 3) -> list[MemoryMatchSummary]:
    summaries: list[MemoryMatchSummary] = []
    for match in list(getattr(memory_recall, "matches", []) or [])[: max(0, int(limit))]:
        summaries.append(
            MemoryMatchSummary(
                step_id=str(match.record.step_id).strip().upper(),
                store=str(match.store_name).strip().lower(),
                source=str(match.record.source).strip().lower(),
                score=float(match.total_score),
            )
        )
    return summaries


def _append_step_history(
    recent_steps,
    frame_index: int,
    step_id: str,
    confidence: float,
    source: str,
    stable: bool,
) -> None:
    normalized_step = str(step_id).strip().upper()
    normalized_source = str(source).strip().lower()
    if not normalized_step or not normalized_source:
        return
    if recent_steps:
        latest = recent_steps[-1]
        if latest.step_id == normalized_step and latest.source == normalized_source:
            return
    recent_steps.append(
        StepTimelineEvent(
            frame_index=int(frame_index),
            step_id=normalized_step,
            confidence=float(confidence),
            source=normalized_source,
            stable=bool(stable),
        )
    )


def _append_feedback_history(recent_feedback, frame_index: int, feedback: FeedbackEvent) -> None:
    recent_feedback.append(
        FeedbackTimelineEvent(
            frame_index=int(frame_index),
            label=str(feedback.label).strip().upper(),
            source=str(feedback.source).strip().lower(),
            accepted=bool(feedback.accepted),
            note=str(feedback.note).strip(),
        )
    )


class _RealtimeMetricsRecorder:
    """Small live-run profiler used only when explicit realtime evaluation is enabled."""

    def __init__(self, *, enabled: bool, run_dir: Path, window: int = 60, report_interval: int = 15) -> None:
        self.enabled = bool(enabled)
        self.window = max(2, int(window))
        self.report_interval = max(1, int(report_interval))
        self.raw_times = deque(maxlen=self.window)
        self.proc_times = deque(maxlen=self.window)
        self.read_latencies_ms: list[float] = []
        self.process_latencies_ms: list[float] = []
        self.loop_latencies_ms: list[float] = []
        self._latest: dict[str, float] = {}
        self._path = Path(run_dir) / "realtime_metrics.jsonl"
        self._frames_path = Path(run_dir) / "frame_timestamps.jsonl"
        self._handle = self._path.open("w", encoding="utf-8") if self.enabled else None
        self._frame_handle = self._frames_path.open("w", encoding="utf-8") if self.enabled else None

    def mark_raw_frame(self, frame_index: int, read_latency_ms: float, video_recorded: bool) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        self.raw_times.append(now)
        self.read_latencies_ms.append(float(read_latency_ms))
        if self._frame_handle is not None:
            self._frame_handle.write(
                json.dumps(
                    {
                        "frame_index": int(frame_index),
                        "perf_time": float(now),
                        "read_latency_ms": float(read_latency_ms),
                        "video_recorded": bool(video_recorded),
                    }
                )
                + "\n"
            )

    def record_processed(
        self,
        *,
        frame_index: int,
        processed_index: int,
        process_latency_ms: float,
        loop_latency_ms: float,
    ) -> dict[str, float]:
        if not self.enabled:
            return {}
        self.proc_times.append(time.perf_counter())
        self.process_latencies_ms.append(float(process_latency_ms))
        self.loop_latencies_ms.append(float(loop_latency_ms))
        record = {
            "frame_index": int(frame_index),
            "processed_index": int(processed_index),
            "raw_fps": self._fps(self.raw_times),
            "processed_fps": self._fps(self.proc_times),
            "read_latency_ms": self._last(self.read_latencies_ms),
            "process_latency_ms": float(process_latency_ms),
            "loop_latency_ms": float(loop_latency_ms),
            "mean_process_latency_ms": self._mean_tail(self.process_latencies_ms),
            "mean_loop_latency_ms": self._mean_tail(self.loop_latencies_ms),
        }
        self._latest = {key: float(value) for key, value in record.items() if isinstance(value, (int, float))}
        if self._handle is not None:
            self._handle.write(json.dumps(record) + "\n")
            if int(processed_index) % self.report_interval == 0:
                self._handle.flush()
        return self._latest

    def status_line(self) -> str:
        if not self.enabled or not self._latest:
            return ""
        return (
            f"perf raw_fps={self._latest.get('raw_fps', 0.0):.1f} "
            f"proc_fps={self._latest.get('processed_fps', 0.0):.1f} "
            f"lat={self._latest.get('loop_latency_ms', 0.0):.1f}ms "
            f"proc={self._latest.get('process_latency_ms', 0.0):.1f}ms"
        )

    def close(self) -> None:
        if not self.enabled:
            return
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None
        if self._frame_handle is not None:
            self._frame_handle.flush()
            self._frame_handle.close()
            self._frame_handle = None
        summary = {
            "raw_frames": len(self.read_latencies_ms),
            "processed_frames": len(self.process_latencies_ms),
            "raw_fps_window": self._fps(self.raw_times),
            "processed_fps_window": self._fps(self.proc_times),
            "read_latency_ms": self._summary(self.read_latencies_ms),
            "process_latency_ms": self._summary(self.process_latencies_ms),
            "loop_latency_ms": self._summary(self.loop_latencies_ms),
        }
        (self._path.parent / "realtime_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    def _mean_tail(self, values: list[float]) -> float:
        tail = values[-self.window :]
        return float(np.mean(tail)) if tail else 0.0

    @staticmethod
    def _last(values: list[float]) -> float:
        return float(values[-1]) if values else 0.0

    @staticmethod
    def _fps(times: deque[float]) -> float:
        if len(times) < 2:
            return 0.0
        elapsed = float(times[-1] - times[0])
        if elapsed <= 1e-6:
            return 0.0
        return float((len(times) - 1) / elapsed)

    @staticmethod
    def _summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        array = np.asarray(values, dtype=np.float32)
        return {
            "mean": float(np.mean(array)),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "max": float(np.max(array)),
        }


def _build_assistant_snapshot(
    frame_index: int,
    step_id: str,
    step_confidence: float,
    runner_up: Optional[str],
    fused_detections: list[Detection],
    relevant_detections: list[Detection],
    scene_graph,
    memory_step: str,
    memory_confidence: float,
    memory_recalled: bool = False,
    memory_reason: str = "",
    memory_matches: Optional[list[MemoryMatchSummary]] = None,
    recent_steps: Optional[list[StepTimelineEvent]] = None,
    recent_feedback: Optional[list[FeedbackTimelineEvent]] = None,
    review_action: str = "",
    review_reason: str = "",
    claim_decision=None,
    compute_plan: Optional[dict[str, object]] = None,
) -> AssistantSnapshot:
    object_counts: dict[str, int] = {}
    for detection in fused_detections:
        object_counts[detection.name] = int(object_counts.get(detection.name, 0)) + 1
    relations = [
        f"{relation.subject_name} {relation.predicate} {relation.object_name}"
        for relation in getattr(scene_graph, "relations", [])[:6]
    ]
    relation_facts = [
        {
            "subject": str(relation.subject_name).strip().lower(),
            "predicate": str(relation.predicate).strip().lower(),
            "object": str(relation.object_name).strip().lower(),
        }
        for relation in getattr(scene_graph, "relations", [])[:6]
    ]
    return AssistantSnapshot(
        frame_index=int(frame_index),
        step_id=str(step_id),
        step_confidence=float(step_confidence),
        runner_up=runner_up,
        visible_objects=[str(item.name) for item in fused_detections],
        relevant_objects=[str(item.name) for item in relevant_detections],
        object_counts=object_counts,
        scene_relations=relations,
        relation_facts=relation_facts,
        memory_step=str(memory_step),
        memory_confidence=float(memory_confidence),
        recent_steps=list(recent_steps or []),
        recent_feedback=list(recent_feedback or []),
        memory_matches=list(memory_matches or []),
        memory_reason=str(memory_reason),
        memory_recalled=bool(memory_recalled),
        review_action=str(review_action),
        review_reason=str(review_reason),
        proposed_step=str(getattr(claim_decision, "proposed_step", "")),
        active_claim=str(getattr(claim_decision, "claim_id", "")),
        claim_state=str(getattr(claim_decision, "state", "insufficient")),
        claim_support=float(getattr(claim_decision, "support_score", 0.0)),
        claim_contradiction=float(getattr(claim_decision, "contradiction_score", 0.0)),
        claim_margin=float(getattr(claim_decision, "counterfactual_margin", 0.0)),
        claim_admissible=bool(getattr(claim_decision, "admissible", True)),
        missing_evidence_roles=list(getattr(claim_decision, "missing_roles", ()) or ()),
        product_family=str(getattr(claim_decision, "product_family", "")),
        acquisition_mode=str((compute_plan or {}).get("acquisition_mode", "monitor")),
        external_observation_recommended=bool(
            (compute_plan or {}).get("external_observation_recommended", False)
        ),
    )


class LiveAssistantRunner:
    """Run the modular step-recognition stack on a live camera feed."""

    def __init__(
        self,
        pipeline: StepPipeline,
        ui: OpenCVRuntimeUI,
        voice: VoiceCommandService,
        speech: AssistantSpeechService,
        assistant: ContextualAssistant,
        camera_index: int = 0,
    ) -> None:
        self.pipeline = pipeline
        self.ui = ui
        self.voice = voice
        self.speech = speech
        self.assistant = assistant
        self.camera_index = int(camera_index)

    def run(self, camera_index: Optional[int] = None, eval_realtime: Optional[bool] = None) -> Path:
        """Run the live assistant on the requested camera index."""

        index = self.camera_index if camera_index is None else int(camera_index)
        source_path = Path(f"camera_{index}_live.mp4")
        logger = JsonlCsvLogger(self.pipeline.runlog_root, source_path)
        self.pipeline.prepare_run(logger, source_uri=f"camera://{index}")
        metrics = _RealtimeMetricsRecorder(
            enabled=bool(self.pipeline.config.runlog.eval_realtime if eval_realtime is None else eval_realtime),
            run_dir=logger.run_dir,
            window=self.pipeline.config.runlog.eval_window,
            report_interval=self.pipeline.config.runlog.eval_report_interval,
        )
        self.pipeline.ops_tracker.update(
            status="Doing",
            frame_index=0,
            step_id="",
            reason="camera_started",
            force=True,
        )

        try:
            capture, backend_name, warmup_ok = _open_camera_capture(
                index, self.pipeline.config.camera, self.pipeline.event_bus
            )
        except Exception:
            self.pipeline.memory_manager.close()
            self.pipeline.event_bus.close()
            self.pipeline.ops_tracker.close()
            metrics.close()
            logger.close()
            raise

        self.ui.open()
        self.voice.start()
        self.speech.start()
        self.pipeline.event_bus.emit(
            "voice.backend_ready",
            {
                "backend": self.voice.backend_name(),
                "status": self.voice.status_text(),
                "mode": self.pipeline.config.voice.mode,
            },
        )
        self.pipeline.event_bus.emit(
            "speech.backend_ready",
            {
                "backend": self.speech.backend_name(),
                "status": self.speech.status_text(),
            },
        )

        frame_index = 0
        processed_index = 0
        prev_step_for_transition: Optional[str] = None
        pending_feedback_actions: list[RuntimeAction] = []
        pending_query_actions: list[RuntimeAction] = []
        paused = False
        force_feedback = False
        running = True
        last_canvas: Optional[np.ndarray] = None
        last_focus: Optional[np.ndarray] = None
        last_status_lines: list[str] = []
        last_assistant_answer = ""
        recent_steps = deque(maxlen=6)
        recent_feedback = deque(maxlen=6)
        read_failures = 0
        eval_video_writer: Optional[cv2.VideoWriter] = None
        eval_video_path = logger.run_dir / "live_capture.mp4"

        try:
            while running:
                running, paused, force_feedback = self._drain_actions(
                    pending_feedback_actions=pending_feedback_actions,
                    pending_query_actions=pending_query_actions,
                    paused=paused,
                    force_feedback=force_feedback,
                    frame_index=frame_index,
                )
                if not running:
                    break
                if paused:
                    if last_canvas is not None:
                        self.ui.render(last_canvas, status_lines=last_status_lines, focus_crop=last_focus)
                    time.sleep(0.05)
                    continue

                read_start = time.perf_counter()
                ok, frame_bgr = capture.read()
                read_end = time.perf_counter()
                if not ok or frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
                    read_failures += 1
                    self.pipeline.event_bus.emit(
                        "camera.read_retry",
                        {
                            "camera_index": index,
                            "backend": backend_name,
                            "consecutive_failures": read_failures,
                            "retry_limit": int(self.pipeline.config.camera.read_retry_limit),
                        },
                        frame_index=frame_index or None,
                    )
                    if read_failures >= max(1, int(self.pipeline.config.camera.read_retry_limit)):
                        self.pipeline.ops_tracker.update(
                            status="Blocked",
                            frame_index=frame_index,
                            step_id=prev_step_for_transition or "",
                            reason="camera_read_exhausted",
                            force=True,
                        )
                        self.pipeline.event_bus.emit(
                            "camera.read_exhausted",
                            {
                                "camera_index": index,
                                "backend": backend_name,
                                "consecutive_failures": read_failures,
                            },
                            frame_index=frame_index or None,
                        )
                        break
                    time.sleep(max(0.0, float(self.pipeline.config.camera.read_retry_delay_ms) / 1000.0))
                    continue
                if read_failures:
                    self.pipeline.event_bus.emit(
                        "camera.read_recovered",
                        {
                            "camera_index": index,
                            "backend": backend_name,
                            "consecutive_failures": read_failures,
                        },
                        frame_index=frame_index or None,
                    )
                    read_failures = 0
                frame_index += 1
                video_recorded = False
                if metrics.enabled:
                    if eval_video_writer is None:
                        height, width = frame_bgr.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        eval_video_writer = cv2.VideoWriter(
                            str(eval_video_path),
                            fourcc,
                            float(self.pipeline.config.camera.fps),
                            (int(width), int(height)),
                        )
                        self.pipeline.event_bus.emit(
                            "runtime.eval_video_started",
                            {"path": str(eval_video_path), "width": int(width), "height": int(height)},
                            frame_index=frame_index,
                        )
                    if eval_video_writer is not None and eval_video_writer.isOpened():
                        eval_video_writer.write(frame_bgr)
                        video_recorded = True
                metrics.mark_raw_frame(frame_index, (read_end - read_start) * 1000.0, video_recorded)
                if frame_index % self.pipeline.config.video.stride != 0:
                    if last_canvas is not None:
                        self.ui.render(last_canvas, status_lines=last_status_lines, focus_crop=last_focus)
                    continue

                processed_index += 1
                process_start = time.perf_counter()
                processed = self.pipeline.process_frame(
                    frame_bgr=frame_bgr,
                    frame_index=frame_index,
                    prev_step_for_transition=prev_step_for_transition,
                    source_kind="camera",
                )
                process_end = time.perf_counter()
                logger.log_iteration(self.pipeline.build_iteration_payload(processed_index, processed, source_kind="camera"))

                refined_reply = self.assistant.poll_refinement()
                if refined_reply is not None:
                    last_assistant_answer = refined_reply.answer
                    self.pipeline.event_bus.emit(
                        "assistant.refinement",
                        {
                            "route": refined_reply.route,
                            "answer": refined_reply.answer,
                            "llm_route": refined_reply.evidence.get("llm_route", "async_refined"),
                        },
                        frame_index=frame_index,
                    )
                    print(f"[Assistant refinement] {refined_reply.answer}")

                fused_detections = processed.fused_detections
                relevant_detections = processed.relevant_detections
                scene_graph = processed.scene_graph
                memory_observation = processed.memory_observation
                memory_recall = processed.memory_recall
                expert_predictions = processed.expert_predictions
                fusion_result = processed.fusion_result
                review_decision = processed.review_decision
                stable = processed.stable

                if not memory_observation.has_visual_evidence:
                    display_step = self.pipeline.display_step(processed, prev_step_for_transition)
                    self.pipeline.ops_tracker.update(
                        status="Doing",
                        frame_index=frame_index,
                        step_id=prev_step_for_transition or "",
                        reason="no_visual_evidence",
                    )
                    snapshot = _build_assistant_snapshot(
                        frame_index=frame_index,
                        step_id=display_step,
                        step_confidence=fusion_result.confidence,
                        runner_up=fusion_result.runner_up,
                        fused_detections=fused_detections,
                        relevant_detections=relevant_detections,
                        scene_graph=scene_graph,
                        memory_step=memory_recall.prediction.step_id,
                        memory_confidence=memory_recall.prediction.confidence,
                        memory_recalled=memory_recall.recalled,
                        memory_reason=memory_recall.reason,
                        memory_matches=_summarize_memory_matches(memory_recall),
                        recent_steps=list(recent_steps),
                        recent_feedback=list(recent_feedback),
                        claim_decision=processed.claim_decision,
                        compute_plan=processed.compute_plan,
                    )
                    while pending_query_actions:
                        query_action = pending_query_actions.pop(0)
                        self.pipeline.event_bus.emit(
                            "assistant.query",
                            {"source": query_action.source, "text": query_action.text},
                            frame_index=frame_index,
                        )
                        reply = self.assistant.answer(query_action.text, snapshot)
                        last_assistant_answer = reply.answer
                        self.pipeline.event_bus.emit(
                            "assistant.answer",
                            {"route": reply.route, "answer": reply.answer},
                            frame_index=frame_index,
                        )
                        self.speech.speak(reply.answer)
                        print(f"[Assistant] {reply.answer}")
                    last_canvas = _draw_overlay(frame_bgr, fused_detections, f"STEP: {display_step} ({fusion_result.confidence:.2f})")
                    last_focus = _focus_crop(frame_bgr, relevant_detections or fused_detections)
                    last_status_lines = [
                        f"camera={index} backend={backend_name} frame={frame_index} stable=0 warmup={warmup_ok}",
                        self.voice.status_text(),
                        "review=none",
                        "evidence=missing",
                    ]
                    metrics.record_processed(
                        frame_index=frame_index,
                        processed_index=processed_index,
                        process_latency_ms=(process_end - process_start) * 1000.0,
                        loop_latency_ms=(time.perf_counter() - read_start) * 1000.0,
                    )
                    perf_line = metrics.status_line()
                    if perf_line:
                        last_status_lines.append(perf_line)
                    last_transcript = self.voice.last_transcript()
                    if last_transcript:
                        last_status_lines.append(f'voice=\"{last_transcript[:72]}\"')
                    last_status_lines.append(self.speech.status_text())
                    if last_assistant_answer:
                        last_status_lines.append(f'assistant=\"{last_assistant_answer[: self.assistant.overlay_chars]}\"')
                    self.ui.render(last_canvas, status_lines=last_status_lines, focus_crop=last_focus)
                    continue

                feedback, should_skip_feedback = self._resolve_pending_feedback(
                    pending_actions=pending_feedback_actions,
                    fused_step=fusion_result.step_id,
                )
                display_step = self.pipeline.display_step(processed, prev_step_for_transition)
                claim_supported = processed.claim_decision is None or processed.claim_decision.state == "supported"
                allow_auto_capture = bool(claim_supported)
                transition_step: Optional[str] = None
                pending_reason = review_decision.reason if review_decision is not None else "stable_accept"
                step_history_source = ""

                if feedback is not None:
                    self.pipeline.fusion_head.apply_feedback(
                        feedback=feedback,
                        expert_predictions=expert_predictions,
                        fusion_result=fusion_result,
                    )
                    self.pipeline.memory_manager.record_feedback(memory_observation, feedback)
                    self.pipeline.review_manager.record_feedback(frame_index, feedback)
                    self.pipeline.ops_tracker.record_feedback(feedback)
                    _append_feedback_history(recent_feedback, frame_index, feedback)
                    self.pipeline.event_bus.emit(
                        "feedback.applied",
                        {
                            "source": feedback.source,
                            "label": feedback.label,
                            "accepted": feedback.accepted,
                            "strength": feedback.strength,
                            "note": feedback.note,
                        },
                        frame_index=frame_index,
                    )
                    logger.log_feedback(
                        _feedback_log_payload(
                            processed_index=processed_index,
                            frame_index=frame_index,
                            feedback=feedback,
                            fusion_result=fusion_result,
                            memory_step=memory_recall.prediction.step_id,
                            memory_conf=memory_recall.prediction.confidence,
                        )
                    )
                    display_step = feedback.label
                    prev_step_for_transition = feedback.label
                    if self.pipeline.claim_verifier is not None:
                        self.pipeline.claim_verifier.confirm_step(feedback.label)
                    step_history_source = "feedback_accept" if feedback.accepted else "feedback_correct"
                    self.pipeline.ops_tracker.update(
                        status="Next",
                        frame_index=frame_index,
                        step_id=feedback.label,
                        reason="voice_accept" if feedback.accepted else "voice_feedback",
                        review_action=review_decision.action if review_decision is not None else "",
                    )
                else:
                    prompt_console = (
                        stable
                        and self.pipeline.feedback_provider is not None
                        and not should_skip_feedback
                        and (force_feedback or bool(review_decision and review_decision.should_prompt_human))
                    )
                    force_feedback = False

                    if prompt_console:
                        operator_feedback = self.pipeline.feedback_provider.request(
                            fusion_result=fusion_result,
                            expert_predictions=expert_predictions,
                            evidence_token=processed.evidence_token,
                            review_decision=review_decision,
                        )
                        if operator_feedback is not None:
                            self.pipeline.fusion_head.apply_feedback(
                                feedback=operator_feedback,
                                expert_predictions=expert_predictions,
                                fusion_result=fusion_result,
                            )
                            self.pipeline.memory_manager.record_feedback(memory_observation, operator_feedback)
                            self.pipeline.review_manager.record_feedback(frame_index, operator_feedback)
                            self.pipeline.ops_tracker.record_feedback(operator_feedback)
                            _append_feedback_history(recent_feedback, frame_index, operator_feedback)
                            self.pipeline.event_bus.emit(
                                "feedback.applied",
                                {
                                    "source": operator_feedback.source,
                                    "label": operator_feedback.label,
                                    "accepted": operator_feedback.accepted,
                                    "strength": operator_feedback.strength,
                                    "note": operator_feedback.note,
                                },
                                frame_index=frame_index,
                            )
                            logger.log_feedback(
                                _feedback_log_payload(
                                    processed_index=processed_index,
                                    frame_index=frame_index,
                                    feedback=operator_feedback,
                                    fusion_result=fusion_result,
                                    memory_step=memory_recall.prediction.step_id,
                                    memory_conf=memory_recall.prediction.confidence,
                                )
                            )
                            display_step = operator_feedback.label
                            prev_step_for_transition = operator_feedback.label
                            if self.pipeline.claim_verifier is not None:
                                self.pipeline.claim_verifier.confirm_step(operator_feedback.label)
                            step_history_source = "feedback_accept" if operator_feedback.accepted else "feedback_correct"
                        else:
                            allow_auto_capture = bool(
                                claim_supported
                                and (review_decision is None or review_decision.action == "approve")
                            )
                    elif (
                        review_decision is not None
                        and review_decision.action == "prefer_candidate"
                        and review_decision.weak_feedback is not None
                    ):
                        reviewer_feedback = review_decision.weak_feedback
                        self.pipeline.fusion_head.apply_feedback(
                            feedback=reviewer_feedback,
                            expert_predictions=expert_predictions,
                            fusion_result=fusion_result,
                        )
                        self.pipeline.memory_manager.record_feedback(memory_observation, reviewer_feedback)
                        self.pipeline.review_manager.record_feedback(frame_index, reviewer_feedback)
                        self.pipeline.ops_tracker.record_feedback(reviewer_feedback)
                        _append_feedback_history(recent_feedback, frame_index, reviewer_feedback)
                        self.pipeline.event_bus.emit(
                            "feedback.applied",
                            {
                                "source": reviewer_feedback.source,
                                "label": reviewer_feedback.label,
                                "accepted": reviewer_feedback.accepted,
                                "strength": reviewer_feedback.strength,
                                "note": reviewer_feedback.note,
                            },
                            frame_index=frame_index,
                        )
                        logger.log_feedback(
                            _feedback_log_payload(
                                processed_index=processed_index,
                                frame_index=frame_index,
                                feedback=reviewer_feedback,
                                fusion_result=fusion_result,
                                memory_step=memory_recall.prediction.step_id,
                                memory_conf=memory_recall.prediction.confidence,
                            )
                        )
                        display_step = reviewer_feedback.label
                        transition_step = reviewer_feedback.label if claim_supported else None
                        allow_auto_capture = False
                        step_history_source = "review_feedback"
                    elif review_decision is not None and review_decision.action in {"hold", "request_human"}:
                        allow_auto_capture = False

                    if feedback is None:
                        if allow_auto_capture and stable:
                            self.pipeline.memory_manager.record_auto(
                                memory_observation,
                                fusion_result=fusion_result,
                                stable=stable,
                            )
                        if transition_step is not None:
                            prev_step_for_transition = transition_step
                            self.pipeline.ops_tracker.update(
                                status="Next",
                                frame_index=frame_index,
                                step_id=transition_step,
                                reason=pending_reason,
                                review_action=review_decision.action if review_decision is not None else "",
                            )
                        elif allow_auto_capture and stable:
                            prev_step_for_transition = fusion_result.step_id
                            step_history_source = "stable_auto"
                            self.pipeline.ops_tracker.update(
                                status="Next",
                                frame_index=frame_index,
                                step_id=fusion_result.step_id,
                                reason="stable_auto",
                                review_action=review_decision.action if review_decision is not None else "",
                            )
                        elif allow_auto_capture:
                            self.pipeline.ops_tracker.update(
                                status="Doing",
                                frame_index=frame_index,
                                step_id=fusion_result.step_id,
                                reason="tracking_candidate",
                                review_action=review_decision.action if review_decision is not None else "",
                            )
                        else:
                            self.pipeline.ops_tracker.update(
                                status="Blocked" if review_decision and review_decision.should_prompt_human else "Review",
                                frame_index=frame_index,
                                step_id=fusion_result.step_id,
                                reason=pending_reason,
                                review_action=review_decision.action if review_decision is not None else "",
                            )

                if step_history_source:
                    _append_step_history(
                        recent_steps,
                        frame_index=frame_index,
                        step_id=display_step,
                        confidence=fusion_result.confidence,
                        source=step_history_source,
                        stable=stable,
                    )

                snapshot = _build_assistant_snapshot(
                    frame_index=frame_index,
                    step_id=display_step,
                    step_confidence=fusion_result.confidence,
                    runner_up=fusion_result.runner_up,
                    fused_detections=fused_detections,
                    relevant_detections=relevant_detections,
                    scene_graph=scene_graph,
                    memory_step=memory_recall.prediction.step_id,
                    memory_confidence=memory_recall.prediction.confidence,
                    memory_recalled=memory_recall.recalled,
                    memory_reason=memory_recall.reason,
                    memory_matches=_summarize_memory_matches(memory_recall),
                    recent_steps=list(recent_steps),
                    recent_feedback=list(recent_feedback),
                    review_action=review_decision.action if review_decision is not None else "",
                    review_reason=review_decision.reason if review_decision is not None else "",
                    claim_decision=processed.claim_decision,
                    compute_plan=processed.compute_plan,
                )
                while pending_query_actions:
                    query_action = pending_query_actions.pop(0)
                    self.pipeline.event_bus.emit(
                        "assistant.query",
                        {"source": query_action.source, "text": query_action.text},
                        frame_index=frame_index,
                    )
                    reply = self.assistant.answer(query_action.text, snapshot)
                    last_assistant_answer = reply.answer
                    self.pipeline.event_bus.emit(
                        "assistant.answer",
                        {"route": reply.route, "answer": reply.answer},
                        frame_index=frame_index,
                    )
                    self.speech.speak(reply.answer)
                    print(f"[Assistant] {reply.answer}")

                last_canvas = _draw_overlay(frame_bgr, fused_detections, f"STEP: {display_step} ({fusion_result.confidence:.2f})")
                last_focus = _focus_crop(frame_bgr, relevant_detections or fused_detections)
                last_status_lines = [
                    f"camera={index} backend={backend_name} frame={frame_index} stable={int(stable)} warmup={warmup_ok}",
                    self.voice.status_text(),
                    f"review={review_decision.action if review_decision is not None else 'none'}",
                ]
                metrics.record_processed(
                    frame_index=frame_index,
                    processed_index=processed_index,
                    process_latency_ms=(process_end - process_start) * 1000.0,
                    loop_latency_ms=(time.perf_counter() - read_start) * 1000.0,
                )
                perf_line = metrics.status_line()
                if perf_line:
                    last_status_lines.append(perf_line)
                if read_failures:
                    last_status_lines.append(f"camera_retry={read_failures}")
                last_transcript = self.voice.last_transcript()
                if last_transcript:
                    last_status_lines.append(f'voice="{last_transcript[:72]}"')
                last_status_lines.append(self.speech.status_text())
                if last_assistant_answer:
                    last_status_lines.append(f'assistant="{last_assistant_answer[: self.assistant.overlay_chars]}"')
                self.ui.render(last_canvas, status_lines=last_status_lines, focus_crop=last_focus)
        finally:
            capture.release()
            if eval_video_writer is not None:
                eval_video_writer.release()
            self.voice.stop()
            self.speech.stop()
            self.ui.close()
            self.pipeline.finalize_run(logger, source_uri=f"camera://{index}")
            metrics.close()

        return logger.run_dir

    def _drain_actions(
        self,
        pending_feedback_actions: list[RuntimeAction],
        pending_query_actions: list[RuntimeAction],
        paused: bool,
        force_feedback: bool,
        frame_index: int,
    ) -> tuple[bool, bool, bool]:
        running = True
        actions = self.voice.poll_actions()
        ui_action = self.ui.poll_action(delay_ms=1)
        if ui_action is not None:
            actions.append(ui_action)
        for action in actions:
            self.pipeline.event_bus.emit(
                "runtime.action",
                {
                    "action": action.action,
                    "source": action.source,
                    "label": action.label,
                    "text": action.text,
                },
                frame_index=frame_index or None,
            )
            if action.action == "quit":
                running = False
            elif action.action in {"toggle_pause", "pause"}:
                paused = not paused if action.action == "toggle_pause" else True
            elif action.action == "resume":
                paused = False
            elif action.action == "toggle_help":
                self.ui.toggle_help()
            elif action.action == "voice_capture":
                if self.pipeline.config.speech.interrupt_on_voice_input:
                    self.assistant.interrupt()
                    self.speech.interrupt()
                self.voice.trigger_manual_capture()
            elif action.action == "toggle_voice_mute":
                self.voice.toggle_mute()
            elif action.action == "mute_voice":
                self.voice.set_muted(True)
            elif action.action == "unmute_voice":
                self.voice.set_muted(False)
            elif action.action == "force_feedback":
                force_feedback = True
            elif action.action in {"feedback_accept", "feedback_correct", "feedback_skip"}:
                pending_feedback_actions.append(action)
            elif action.action == "transcript":
                if self.pipeline.config.speech.interrupt_on_voice_input:
                    self.assistant.interrupt()
                    self.speech.interrupt()
                pending_query_actions.append(action)
        return running, paused, force_feedback

    @staticmethod
    def _resolve_pending_feedback(
        pending_actions: list[RuntimeAction],
        fused_step: str,
    ) -> tuple[Optional[FeedbackEvent], bool]:
        if not pending_actions:
            return None, False
        action = pending_actions.pop(0)
        if action.action == "feedback_skip":
            return None, True
        if action.action == "feedback_accept":
            return FeedbackEvent(label=fused_step, accepted=True, source=action.source, note=action.text), False
        if action.action == "feedback_correct" and action.label:
            return FeedbackEvent(label=action.label, accepted=False, source=action.source, note=action.text), False
        return None, False


def build_live_assistant(
    config,
    kb_path: str,
    yolo_weights: str,
    device: str,
    state_path: Optional[Path],
    interactive: bool,
    feedback_provider_override=None,
    camera_index: Optional[int] = None,
) -> LiveAssistantRunner:
    """Build a live camera assistant from the shared modular components."""

    cache_dir = config.voice.cache_dir or str(Path(__file__).resolve().parents[1] / ".cache" / "stt")
    pipeline = build_default_pipeline(
        config=config,
        kb_path=kb_path,
        yolo_weights=yolo_weights,
        device=device,
        state_path=state_path,
        interactive=interactive,
        feedback_provider_override=feedback_provider_override,
    )
    ui = OpenCVRuntimeUI(
        enabled=config.ui.enabled,
        window_name=config.ui.window_name,
        focus_window_name=config.ui.focus_window_name,
        show_focus=config.ui.show_focus,
        show_help=config.ui.show_help,
        key_quit=config.ui.key_quit,
        key_pause=config.ui.key_pause,
        key_voice=config.ui.key_voice,
        key_mute=config.ui.key_mute,
        key_feedback=config.ui.key_feedback,
        key_help=config.ui.key_help,
    )
    steps = pipeline.kb.workflow_steps(config.experts.steps)
    voice = VoiceCommandService(
        steps=steps,
        enabled=config.voice.enabled,
        mode=config.voice.mode,
        backend=config.voice.backend,
        model_name=config.voice.model_name,
        cache_dir=cache_dir,
        language=config.voice.language,
        wake_words=config.voice.wake_words,
        require_wake_word_in_always_on=config.voice.require_wake_word_in_always_on,
        command_max_tokens=config.voice.command_max_tokens,
        compute_type=config.voice.compute_type,
        cpu_threads=config.voice.cpu_threads,
        sample_rate=config.voice.sample_rate,
        manual_duration_sec=config.voice.manual_duration_sec,
        chunk_duration_sec=config.voice.chunk_duration_sec,
        cooldown_sec=config.voice.cooldown_sec,
        min_rms=config.voice.min_rms,
        device=config.voice.device,
        mute_by_default=config.voice.mute_by_default,
    )
    speech = AssistantSpeechService(
        enabled=config.speech.enabled,
        backend=config.speech.backend,
        rate=config.speech.rate,
        volume=config.speech.volume,
        voice_name=config.speech.voice_name,
        kokoro_model_path=config.speech.kokoro_model_path,
        kokoro_voices_path=config.speech.kokoro_voices_path,
        kokoro_voice=config.speech.kokoro_voice,
        kokoro_language=config.speech.kokoro_language,
        kokoro_speed=config.speech.kokoro_speed,
        kokoro_python_path=config.speech.kokoro_python_path,
        dedupe_window_sec=config.speech.dedupe_window_sec,
        drop_pending_on_new=config.speech.drop_pending_on_new,
    )
    assistant = ContextualAssistant(
        kb=pipeline.kb,
        enabled=config.assistant.enabled,
        history_limit=config.assistant.history_limit,
        relation_limit=config.assistant.relation_limit,
        overlay_chars=config.assistant.overlay_chars,
        llm=GroundedLLMResponder(
            enabled=config.assistant.llm_enabled,
            provider=config.assistant.llm_provider,
            model_id=config.assistant.llm_model_id,
            model_path=config.assistant.llm_model_path,
            device_map=config.assistant.llm_device_map,
            torch_dtype=config.assistant.llm_torch_dtype,
            max_new_tokens=config.assistant.llm_max_new_tokens,
            temperature=config.assistant.llm_temperature,
            top_p=config.assistant.llm_top_p,
            timeout_sec=config.assistant.llm_timeout_sec,
            history_turns=config.assistant.llm_history_turns,
            max_context_chars=config.assistant.llm_max_context_chars,
            answer_word_limit=config.assistant.llm_answer_word_limit,
            streaming=config.assistant.llm_streaming,
            load_on_start=config.assistant.llm_load_on_start,
            fallback_to_template=config.assistant.llm_fallback_to_template,
            grounding_guard_enabled=config.assistant.llm_grounding_guard_enabled,
            trust_remote_code=config.assistant.llm_trust_remote_code,
        ),
        llm_async_refine=config.assistant.llm_async_refine,
        llm_intents=(
            config.edge_runtime.small_llm_intents
            if config.edge_runtime.enabled
            else None
        ),
    )
    return LiveAssistantRunner(
        pipeline=pipeline,
        ui=ui,
        voice=voice,
        speech=speech,
        assistant=assistant,
        camera_index=config.camera.index if camera_index is None else camera_index,
    )
