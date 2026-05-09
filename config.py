"""Configuration dataclasses and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


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
    tta: bool = True
    nms_iou: float = 0.4
    use_builtin_tta: bool = False
    min_box_area_ratio: float = 0.0002
    max_box_area_ratio: float = 0.60
    max_aspect_ratio: float = 6.0
    reject_multi_border_boxes: bool = True
    border_margin_px: int = 4
    max_per_class: int = 2
    dedupe_iou: float = 0.65


@dataclass
class GeometryConfig:
    backend: str = "gradient"
    moge_model: str = "Ruicheng/moge-2-vits-normal"
    use_fp16: bool = True
    resolution_level: int = 9
    apply_mask: bool = True


@dataclass
class TemporalConfig:
    backend: str = "bytetrack_lite"
    window: int = 3
    iou_thr: float = 0.35
    track_high_thresh: float = 0.32
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.40
    match_iou_thr: float = 0.25
    lost_buffer: int = 8
    min_confirmed_hits: int = 2
    smooth_alpha: float = 0.70


@dataclass
class DepthContextConfig:
    tau_p: float = 80.0
    tau_d: float = 0.12
    use_relevant_for_rules: bool = True


@dataclass
class StabilityConfig:
    stable_n: int = 5
    require_same_step: int = 2


@dataclass
class SceneGraphConfig:
    enabled: bool = True
    depth_margin: float = 0.08
    contact_pixel_gap: float = 24.0
    contact_depth_gap: float = 0.12
    support_vertical_gap: float = 20.0
    support_overlap_ratio: float = 0.25
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
    window_name: str = "MICA Live"
    focus_window_name: str = "MICA Focus"
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
    wake_words: List[str] = field(default_factory=lambda: ["mica", "assistant"])
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
    backend: str = "auto"
    rate: int = 1
    volume: float = 1.0
    voice_name: str = ""
    kokoro_model_path: str = ""
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
    graph_requirement_bonus: float = 0.10
    graph_requirement_penalty: float = 0.12
    graph_forbid_penalty: float = 0.16
    graph_relation_bonus: float = 0.04
    graph_relation_penalty: float = 0.06


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
class CorlAblationConfig:
    gru_checkpoint: str = ""
    gru_aux_checkpoint: str = ""
    gru_agg_checkpoint: str = ""
    gru_agg_offline_gate_checkpoint: str = ""
    gru_agg_offline_gate_context_gate_checkpoint: str = ""
    full_online_adapt_checkpoint: str = ""
    full_online_adapt_context_gate_checkpoint: str = ""


@dataclass
class RunLogConfig:
    save_dir: str = "runs_modular"
    detail_level: str = "debug"
    live_detail_level: str = "sparse"
    persist_temporal_tokens: bool = True


@dataclass
class AppConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
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
    temporal_step: TemporalStepConfig = field(default_factory=TemporalStepConfig)
    online_fusion: OnlineFusionConfig = field(default_factory=OnlineFusionConfig)
    corl_ablation: CorlAblationConfig = field(default_factory=CorlAblationConfig)
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
            temporal_step=TemporalStepConfig(**(payload.get("temporal_step", {}) or {})),
            online_fusion=OnlineFusionConfig(**(payload.get("online_fusion", {}) or {})),
            corl_ablation=CorlAblationConfig(**(payload.get("corl_ablation", {}) or {})),
            runlog=RunLogConfig(**(payload.get("runlog", {}) or {})),
        )


def apply_ablation_preset(config: AppConfig, preset: Optional[str]) -> AppConfig:
    """Apply a named ablation preset to the loaded config."""

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
    if normalized in {"gru", "gru-aux", "gru-agg", "gru-agg-offline-gate", "full-online-adapt"}:
        return _apply_corl_ablation_preset(config, normalized)
    raise ValueError(f"Unknown ablation preset: {preset}")


def _apply_corl_ablation_preset(config: AppConfig, preset: str) -> AppConfig:
    config.temporal_step.enabled = True
    config.temporal_step.backend = "gru_stream"
    config.memory.enabled = False
    config.review.enabled = False
    config.online_fusion.context_gate_enabled = False
    config.online_fusion.context_gate_checkpoint = ""
    config.temporal_step.learned_token_aggregation = False

    def pick(*values: str) -> str:
        for value in values:
            candidate = str(value or "").strip()
            if candidate:
                return candidate
        return ""

    def require(value: str, *, name: str) -> str:
        candidate = str(value or "").strip()
        if candidate:
            return candidate
        raise ValueError(f"A checkpoint path is required for the '{preset}' preset: missing {name}.")

    if preset == "gru":
        config.temporal_step.checkpoint_path = require(
            pick(config.corl_ablation.gru_checkpoint, config.temporal_step.checkpoint_path),
            name="corl_ablation.gru_checkpoint",
        )
        return config

    if preset == "gru-aux":
        config.temporal_step.checkpoint_path = require(
            pick(
                config.corl_ablation.gru_aux_checkpoint,
                config.corl_ablation.gru_checkpoint,
                config.temporal_step.checkpoint_path,
            ),
            name="corl_ablation.gru_aux_checkpoint",
        )
        return config

    if preset == "gru-agg":
        config.temporal_step.learned_token_aggregation = True
        config.temporal_step.checkpoint_path = require(
            pick(
                config.corl_ablation.gru_agg_checkpoint,
                config.corl_ablation.gru_aux_checkpoint,
                config.temporal_step.checkpoint_path,
            ),
            name="corl_ablation.gru_agg_checkpoint",
        )
        return config

    if preset == "gru-agg-offline-gate":
        config.temporal_step.learned_token_aggregation = True
        config.temporal_step.checkpoint_path = require(
            pick(
                config.corl_ablation.gru_agg_offline_gate_checkpoint,
                config.corl_ablation.gru_agg_checkpoint,
                config.temporal_step.checkpoint_path,
            ),
            name="corl_ablation.gru_agg_offline_gate_checkpoint",
        )
        config.online_fusion.context_gate_enabled = True
        config.online_fusion.context_gate_checkpoint = require(
            pick(
                config.corl_ablation.gru_agg_offline_gate_context_gate_checkpoint,
                config.online_fusion.context_gate_checkpoint,
            ),
            name="corl_ablation.gru_agg_offline_gate_context_gate_checkpoint",
        )
        return config

    if preset == "full-online-adapt":
        config.temporal_step.learned_token_aggregation = True
        config.temporal_step.checkpoint_path = require(
            pick(
                config.corl_ablation.full_online_adapt_checkpoint,
                config.corl_ablation.gru_agg_offline_gate_checkpoint,
                config.corl_ablation.gru_agg_checkpoint,
                config.temporal_step.checkpoint_path,
            ),
            name="corl_ablation.full_online_adapt_checkpoint",
        )
        config.memory.enabled = True
        config.review.enabled = True
        config.online_fusion.context_gate_enabled = True
        config.online_fusion.context_gate_checkpoint = require(
            pick(
                config.corl_ablation.full_online_adapt_context_gate_checkpoint,
                config.corl_ablation.gru_agg_offline_gate_context_gate_checkpoint,
                config.online_fusion.context_gate_checkpoint,
            ),
            name="corl_ablation.full_online_adapt_context_gate_checkpoint",
        )
        return config

    raise ValueError(f"Unknown CoRL ablation preset: {preset}")


def load_config(path: Optional[str]) -> AppConfig:
    """Load a YAML config file into an :class:`AppConfig`."""

    if path is None:
        path = str(Path(__file__).resolve().parent / "resources" / "config.example.yaml")
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return apply_ablation_preset(AppConfig.from_dict(payload), None)
