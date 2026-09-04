"""Configuration dataclasses and YAML loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _default_qwen3_model_path() -> str:
    env_path = os.environ.get("INSPECT_QWEN3_MODEL_PATH", "").strip()
    if env_path:
        return env_path
    return "models/qwen3-0.6b"


@dataclass
class VideoConfig:
    stride: int = 3
    write_annotated: bool = False
    output_fps: Optional[float] = None


@dataclass
class CameraConfig:
    index: int = 0
    backend: str = "auto"
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    buffer_size: int = 1
    warmup_frames: int = 8
    warmup_delay_ms: int = 40
    read_retry_limit: int = 60
    read_retry_delay_ms: int = 50
    detection_conf: Optional[float] = 0.30
    detection_tta: Optional[bool] = False


@dataclass
class DetectionConfig:
    conf: float = 0.25
    identity_commit_conf: float = 0.50
    identity_commit_track_margin: float = 0.12
    tta: bool = True
    nms_iou: float = 0.4
    end2end: Optional[bool] = None
    use_builtin_tta: bool = False
    track_class_smoothing_alpha: float = 0.82
    track_class_switch_margin: float = 0.12
    min_box_area_ratio: float = 0.0002
    max_box_area_ratio: float = 0.60
    max_aspect_ratio: float = 6.0
    reject_multi_border_boxes: bool = True
    border_margin_px: int = 4
    max_per_class: int = 2
    dedupe_iou: float = 0.65
    role_bridge_enabled: bool = False
    role_bridge_max_gap: int = 120
    role_bridge_confidence_decay: float = 0.995
    role_bridge_min_confidence: float = 0.08
    role_bridge_max_width: int = 960
    role_bridge_min_points: int = 6
    role_bridge_fb_error: float = 2.5


@dataclass
class GeometryConfig:
    backend: str = "moge"
    moge_model: str = "Ruicheng/moge-2-vits-normal"
    use_fp16: bool = True
    resolution_level: int = 9
    apply_mask: bool = True


@dataclass
class TemporalConfig:
    backend: str = "ultralytics_botsort"
    tracker_config: str = "botsort.yaml"
    window: int = 3
    iou_thr: float = 0.35
    track_high_thresh: float = 0.32
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.40
    match_iou_thr: float = 0.25
    lost_buffer: int = 8
    min_confirmed_hits: int = 2
    smooth_alpha: float = 0.70
    identity_groups: List[List[str]] = field(default_factory=list)
    cross_identity_match_penalty: float = 0.04
    preserve_current_detections: bool = False


@dataclass
class TrackEvidenceConfig:
    enabled: bool = True
    stable_hits: int = 3
    stable_confidence: float = 0.35
    role_track_min_quality: float = 0.50
    motion_px_threshold: float = 4.0
    max_tracks: int = 24
    prefer_stable_tracks_for_rules: bool = True
    prefer_stable_tracks_for_stability: bool = True
    fallback_to_confirmed_tracks: bool = True


@dataclass
class SceneEvidenceConfig:
    enabled: bool = True
    relation_change_memory: int = 24
    max_objects: int = 32
    max_relations: int = 48
    max_changes: int = 32


@dataclass
class DepthContextConfig:
    tau_p: float = 80.0
    tau_d: float = 0.12
    # Step proposal describes scene-level progress; nearest-object context is
    # retained for interaction focus and claim-level evidence queries.
    use_relevant_for_rules: bool = False


@dataclass
class StabilityConfig:
    stable_n: int = 5
    require_same_step: int = 2


@dataclass
class SceneGraphConfig:
    enabled: bool = True
    visibility_calibration_enabled: bool = True
    depth_margin: float = 0.08
    depth_inner_ratio: float = 0.12
    min_depth_valid_fraction: float = 0.35
    max_relative_depth_mad: float = 0.20
    contact_pixel_gap: float = 24.0
    contact_depth_gap: float = 0.12
    support_vertical_gap: float = 20.0
    support_overlap_ratio: float = 0.25
    hard_pair_dedupe_iou: float = 0.85
    min_relation_score: float = 0.08
    max_relations: int = 24


@dataclass
class InteractionConfig:
    enabled: bool = True
    hand_names: List[str] = field(default_factory=lambda: ["hand", "left_hand", "right_hand", "glove"])
    tool_names: List[str] = field(default_factory=lambda: ["screwdriver", "wrench", "pliers", "tool"])
    exclude_object_names: List[str] = field(default_factory=lambda: ["person", "arm", "hand", "left_hand", "right_hand", "glove"])
    contact_margin_px: float = 18.0
    near_margin_px: float = 48.0
    depth_contact_gap: float = 0.12
    min_contact_score: float = 0.35
    history: int = 6


@dataclass
class SegmentationConfig:
    backend: str = "none"  # "sam3.1" for best masks, "none" for bbox/geometry-only fallback
    model_name: str = "facebook/sam3.1"
    device: str = "auto"
    prompts: List[str] = field(default_factory=list)
    score_threshold: float = 0.45
    max_masks: int = 32
    fail_on_unavailable: bool = True


@dataclass
class ExpertsConfig:
    steps: List[str] = field(default_factory=lambda: ["S1", "S2", "S3", "S4"])
    gallery_root: str = ""
    gallery_exts: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp"])
    gallery_embed: str = "hybrid-4"
    retrieval_topk: int = 5
    strict_gallery: bool = False


@dataclass
class MemoryConfig:
    preset: str = "custom"
    enabled: bool = True
    session_enabled: bool = True
    long_term_enabled: bool = True
    auto_capture_enabled: bool = True
    long_term_path: str = ""
    topk: int = 6
    max_per_source: int = 3
    recall_margin: float = 0.18
    recall_disagreement: float = 0.30
    recall_cooldown: int = 2
    recall_on_transition: bool = True
    vector_weight: float = 0.72
    token_weight: float = 0.20
    prev_step_bonus: float = 0.08
    recency_half_life_sec: float = 1800.0
    session_source_weight: float = 1.0
    long_term_source_weight: float = 0.85
    dedup_similarity: float = 0.97
    min_auto_capture_confidence: float = 0.78
    accepted_long_term_margin: float = 0.22
    accepted_long_term_disagreement: float = 0.30
    corrected_source_gain: float = 1.15
    accepted_source_gain: float = 1.0
    auto_source_gain: float = 0.85


@dataclass
class ReviewConfig:
    enabled: bool = True
    low_confidence: float = 0.62
    margin_threshold: float = 0.12
    disagreement_threshold: float = 0.34
    memory_confidence_threshold: float = 0.55
    correction_window: int = 48
    correction_streak: int = 2
    cooldown_frames: int = 6
    prefer_vote_count: int = 2
    prefer_confidence: float = 0.58
    prefer_margin: float = 0.10
    hold_confidence: float = 0.48
    request_human_confidence: float = 0.42
    reviewer_feedback_strength: float = 0.35
    evidence_prompt_enabled: bool = True


@dataclass
class EventsConfig:
    enabled: bool = True


@dataclass
class OpsConfig:
    enabled: bool = True
    persist_history: bool = True


@dataclass
class UIConfig:
    enabled: bool = True
    window_name: str = "INSPECT Trace Engine"
    focus_window_name: str = "INSPECT Focus"
    show_focus: bool = True
    show_help: bool = True
    key_quit: str = "q"
    key_pause: str = "p"
    key_voice: str = "v"
    key_mute: str = "m"
    key_feedback: str = "f"
    key_help: str = "h"


@dataclass
class VoiceConfig:
    enabled: bool = True
    mode: str = "manual"
    backend: str = "auto"
    model_name: str = "distil-small.en"
    cache_dir: str = ""
    language: str = "en"
    wake_words: List[str] = field(default_factory=lambda: ["inspect", "assistant"])
    require_wake_word_in_always_on: bool = True
    command_max_tokens: int = 6
    compute_type: str = "auto"
    cpu_threads: int = 0
    sample_rate: int = 16000
    manual_duration_sec: float = 4.0
    chunk_duration_sec: float = 2.5
    cooldown_sec: float = 1.2
    min_rms: float = 0.008
    device: Optional[int] = None
    mute_by_default: bool = False


@dataclass
class SpeechConfig:
    enabled: bool = True
    backend: str = "kokoro"
    rate: int = 1
    volume: float = 1.0
    voice_name: str = ""
    kokoro_model_path: str = "models/kokoro-82m"
    kokoro_voices_path: str = ""
    kokoro_voice: str = "af_sarah"
    kokoro_language: str = "en-us"
    kokoro_speed: float = 1.0
    kokoro_python_path: str = ""
    dedupe_window_sec: float = 1.5
    drop_pending_on_new: bool = True
    interrupt_on_voice_input: bool = True


@dataclass
class AssistantConfig:
    enabled: bool = True
    history_limit: int = 12
    relation_limit: int = 3
    overlay_chars: int = 120
    llm_enabled: bool = True
    llm_provider: str = "transformers"
    llm_model_id: str = "Qwen/Qwen3-0.6B"
    llm_model_path: str = field(default_factory=_default_qwen3_model_path)
    llm_device_map: str = "auto"
    llm_torch_dtype: str = "auto"
    llm_max_new_tokens: int = 48
    llm_temperature: float = 0.0
    llm_top_p: float = 0.9
    llm_timeout_sec: float = 2.0
    llm_history_turns: int = 3
    llm_max_context_chars: int = 2400
    llm_answer_word_limit: int = 40
    llm_streaming: bool = True
    llm_async_refine: bool = True
    llm_load_on_start: bool = False
    llm_fallback_to_template: bool = True
    llm_grounding_guard_enabled: bool = True
    llm_trust_remote_code: bool = True


@dataclass
class ClaimVerifierConfig:
    """Frozen claim-verification operating point used by replay and live assistance."""

    enabled: bool = False
    evidence_scorer_path: str = ""
    support_threshold: float = 0.35
    contradiction_threshold: float = 0.35
    counterfactual_margin: float = 0.0
    admissibility_gate_enabled: bool = True
    memory_gate_enabled: bool = True
    specialized_counterfactual_enabled: bool = True
    prerequisite_bootstrap_enabled: bool = True
    prerequisite_confirmation_frames: int = 2
    ema_decay: float = 0.55
    require_step_match_for_support: bool = True
    product_family: str = ""
    family_min_confidence: float = 0.08
    family_margin: float = 0.03
    family_confirmation_frames: int = 2


@dataclass
class CausalEvidenceBankConfig:
    """Sparse assistant evidence shared by proposal and claim triage."""

    enabled: bool = False
    proposal_model_path: str = ""
    triage_model_path: str = ""
    dinov2_repo: str = "~/.cache/torch/hub/facebookresearch_dinov2_main"
    device: str = "cuda:0"
    load_encoder_on_start: bool = False


@dataclass
class EdgeRuntimeConfig:
    """Latency and compute-routing controls for edge deployment."""

    enabled: bool = False
    profiling_enabled: bool = True
    synchronize_cuda_for_timing: bool = False
    target_perception_ms: float = 100.0
    target_answer_ms: float = 350.0
    detection_interval_stable: int = 2
    geometry_interval_stable: int = 6
    retrieval_interval_stable: int = 3
    memory_embedding_interval_stable: int = 6
    visual_probe_width: int = 160
    visual_change_threshold: float = 0.012
    max_geometry_age: int = 12
    max_memory_embedding_age: int = 12
    cold_start_frames: int = 2
    persistent_missing_patience: int = 2
    min_epistemic_gain: float = 0.02
    min_stage_evidence_value: float = 0.25
    force_detection_roles: List[str] = field(
        default_factory=lambda: [
            "identity_disambiguation",
            "object_presence",
            "occlusion_recovery",
        ]
    )
    force_retrieval_roles: List[str] = field(
        default_factory=lambda: [
            "identity_disambiguation",
            "object_presence",
        ]
    )
    force_geometry_roles: List[str] = field(
        default_factory=lambda: [
            "alignment",
            "boundary_visibility",
            "contact_verification",
            "containment",
            "gap_visibility",
            "insertion",
            "slot_relation",
        ]
    )
    force_segmentation_roles: List[str] = field(
        default_factory=lambda: [
            "boundary_visibility",
            "contact_verification",
            "containment",
            "gap_visibility",
            "insertion",
            "slot_relation",
        ]
    )
    structured_response_intents: List[str] = field(
        default_factory=lambda: [
            "capability",
            "current_step",
            "history_feedback",
            "history_step",
            "memory_context",
            "next_step",
            "object_count",
            "object_presence",
            "why_not_progressing",
        ]
    )
    small_llm_intents: List[str] = field(
        default_factory=lambda: [
            "component_info",
            "object_relation",
            "safety",
            "troubleshooting",
        ]
    )


@dataclass
class TemporalStepConfig:
    enabled: bool = True
    backend: str = "ema_graph"
    history_size: int = 12
    ema_alpha: float = 0.55
    device: str = "cpu"
    checkpoint_path: str = ""
    gru_hidden_size: int = 0
    learned_token_aggregation: bool = False
    token_hidden_size: int = 24
    token_output_size: int = 48
    state_score_weight: float = 0.60
    retrieval_score_weight: float = 0.25
    memory_score_weight: float = 0.15
    graph_transition_penalty: float = 0.18
    graph_requirement_bonus: float = 0.03
    graph_requirement_penalty: float = 0.04
    graph_forbid_penalty: float = 0.16
    graph_relation_bonus: float = 0.12
    graph_relation_penalty: float = 0.10


@dataclass
class OnlineFusionConfig:
    state_gate: float = 0.5
    temporal_gate: float = 0.45
    retrieval_gate: float = 0.5
    memory_gate: float = 0.2
    leak_state: float = 0.05
    leak_temporal: float = 0.03
    leak_retrieval: float = 0.05
    leak_memory: float = 0.02
    clamp_lo: float = 0.05
    clamp_hi: float = 0.95
    floor_per_class: float = 0.02
    bias_cap: float = 0.5
    eta: float = 0.10
    gate_eta: float = 0.05
    margin: float = 0.20
    positive_margin: float = 0.05
    positive_scale: float = 0.35
    hit_gamma: float = 2.0
    error_gamma: float = 2.0
    freeze_confidence: float = 0.90
    exposure_rho: float = 0.5
    balance_window: int = 50
    balance_tau: float = 0.6
    lambda_transition: float = 2.0
    context_gate_enabled: bool = True
    context_gate_scale: float = 0.35
    context_gate_eta: float = 0.02
    context_gate_checkpoint: str = ""
    transitions: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class RunLogConfig:
    save_dir: str = "runs_modular"
    detail_level: str = "debug"
    live_detail_level: str = "sparse"
    persist_temporal_tokens: bool = True
    eval_realtime: bool = False
    eval_window: int = 60
    eval_report_interval: int = 15


@dataclass
class AppConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    track_evidence: TrackEvidenceConfig = field(default_factory=TrackEvidenceConfig)
    scene_evidence: SceneEvidenceConfig = field(default_factory=SceneEvidenceConfig)
    depth_context: DepthContextConfig = field(default_factory=DepthContextConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    scene_graph: SceneGraphConfig = field(default_factory=SceneGraphConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    experts: ExpertsConfig = field(default_factory=ExpertsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    ops: OpsConfig = field(default_factory=OpsConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    assistant: AssistantConfig = field(default_factory=AssistantConfig)
    claim_verifier: ClaimVerifierConfig = field(default_factory=ClaimVerifierConfig)
    causal_evidence_bank: CausalEvidenceBankConfig = field(
        default_factory=CausalEvidenceBankConfig
    )
    edge_runtime: EdgeRuntimeConfig = field(default_factory=EdgeRuntimeConfig)
    temporal_step: TemporalStepConfig = field(default_factory=TemporalStepConfig)
    online_fusion: OnlineFusionConfig = field(default_factory=OnlineFusionConfig)
    runlog: RunLogConfig = field(default_factory=RunLogConfig)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AppConfig":
        """Construct a typed config from a raw dictionary."""

        voice_payload = dict(payload.get("voice", {}) or {})
        voice_payload.pop("whisper_model", None)
        return cls(
            video=VideoConfig(**(payload.get("video", {}) or {})),
            camera=CameraConfig(**(payload.get("camera", {}) or {})),
            detection=DetectionConfig(**(payload.get("detection", {}) or {})),
            geometry=GeometryConfig(**(payload.get("geometry", {}) or {})),
            temporal=TemporalConfig(**(payload.get("temporal", {}) or {})),
            track_evidence=TrackEvidenceConfig(**(payload.get("track_evidence", {}) or {})),
            scene_evidence=SceneEvidenceConfig(**(payload.get("scene_evidence", {}) or {})),
            depth_context=DepthContextConfig(**(payload.get("depth_context", {}) or {})),
            stability=StabilityConfig(**(payload.get("stability", {}) or {})),
            scene_graph=SceneGraphConfig(**(payload.get("scene_graph", {}) or {})),
            interaction=InteractionConfig(**(payload.get("interaction", {}) or {})),
            segmentation=SegmentationConfig(**(payload.get("segmentation", {}) or {})),
            experts=ExpertsConfig(**(payload.get("experts", {}) or {})),
            memory=MemoryConfig(**(payload.get("memory", {}) or {})),
            review=ReviewConfig(**(payload.get("review", {}) or {})),
            events=EventsConfig(**(payload.get("events", {}) or {})),
            ops=OpsConfig(**(payload.get("ops", {}) or {})),
            ui=UIConfig(**(payload.get("ui", {}) or {})),
            voice=VoiceConfig(**voice_payload),
            speech=SpeechConfig(**(payload.get("speech", {}) or {})),
            assistant=AssistantConfig(**(payload.get("assistant", {}) or {})),
            claim_verifier=ClaimVerifierConfig(**(payload.get("claim_verifier", {}) or {})),
            causal_evidence_bank=CausalEvidenceBankConfig(
                **(payload.get("causal_evidence_bank", {}) or {})
            ),
            edge_runtime=EdgeRuntimeConfig(**(payload.get("edge_runtime", {}) or {})),
            temporal_step=TemporalStepConfig(**(payload.get("temporal_step", {}) or {})),
            online_fusion=OnlineFusionConfig(**(payload.get("online_fusion", {}) or {})),
            runlog=RunLogConfig(**(payload.get("runlog", {}) or {})),
        )


def apply_memory_preset(config: AppConfig, preset: Optional[str]) -> AppConfig:
    """Apply a named memory preset to the loaded config."""

    normalized = str(preset or config.memory.preset or "").strip().lower()
    if not normalized or normalized == "custom":
        return config
    if normalized == "memory-off":
        config.memory.enabled = False
        return config
    if normalized == "session-only":
        config.memory.enabled = True
        config.memory.session_enabled = True
        config.memory.long_term_enabled = False
        return config
    if normalized == "long-term-only":
        config.memory.enabled = True
        config.memory.session_enabled = False
        config.memory.long_term_enabled = True
        return config
    if normalized == "no-auto-capture":
        config.memory.enabled = True
        config.memory.auto_capture_enabled = False
        return config
    raise ValueError(f"Unknown memory preset: {preset}")


def apply_ablation_preset(config: AppConfig, preset: Optional[str]) -> AppConfig:
    """Backward-compatible alias for older scripts."""

    return apply_memory_preset(config, preset)


def load_config(path: Optional[str]) -> AppConfig:
    """Load a YAML config file into an :class:`AppConfig`."""

    if path is None:
        path = str(Path(__file__).resolve().parent / "resources" / "config.example.yaml")
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return apply_memory_preset(AppConfig.from_dict(payload), None)
