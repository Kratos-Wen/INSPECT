# MICA Step Modular

`mica_step_modular` is a new, standalone step-recognition package for the ACVR workspace.
It does not modify the original `mica_glasses` scripts.

The package focuses on the perception-to-step pipeline:

1. detection
2. geometry inference
3. temporal fusion
4. geometry-guided context selection
5. geometry-aware scene graph construction
6. rule-based step expert
7. retrieval-based step expert with a shared visual encoder
8. episodic memory recall and capture
9. sparse review inspired by EDICT-style governance
10. online adaptive fusion with human feedback
11. live camera assistant with OpenCV UI and optional voice control
12. contextual scene-aware Q&A on top of the live assistant
13. structured logging, sparse events, and ops-state tracking
14. INSPECT artifacts for assistance-as-supervision robot state verification

The main design goal is modularity. Each file owns a single responsibility, so components
can be swapped for replacement experiments and ablation studies.

## INSPECT: assistance-as-supervision artifacts

This repository now includes an INSPECT layer:

```text
INSPECT = Interactive Supervision for Procedural Evidence and Cross-view Task Verification
```

INSPECT is the system layer for the paper direction:

```text
Learning What to Inspect:
Assistance-as-Supervision for Robot Procedural State Verification
```

It is not a rename of MICA and it is not another egocentric assistant. MICA-style runtime
outputs remain the data-collection interface. INSPECT consumes those outputs and builds
robot-learning artifacts:

```text
online-corrected assistance traces
  -> verified procedural evidence traces
  -> procedural evidence graph
  -> robot-view procedural state verifier
  -> trace-guided active observation decisions
```

The distinction from Pro^2Assist, Vinci, EgoLife, and Vid2Coach is the target of learning.
Those systems primarily use egocentric perception to help a person in the moment. INSPECT
uses the human's corrections, confirmations, sparse review events, memory recall, object
relations, and visual evidence as supervision for a robot to verify task state from its own
viewpoint. The robot is not asked to imitate the human camera trajectory or low-level action;
it learns which evidence validates the current procedural state and which missing evidence
should trigger a viewpoint change, pause, or human query.

After running the existing pipeline, build INSPECT artifacts from the run directory:

```bash
python -m mica_step_modular inspect ^
  --run-dir path/to/run_dir ^
  --output-dir path/to/inspect_artifacts ^
  --state-specs path/to/state_specs.json
```

This writes:

- `verified_traces.jsonl`: assistance-derived trace events with observed/missing evidence
- `procedural_evidence_graph.json`: state-object-relation-failure evidence graph
- `inspect_summary.json`: compact artifact summary, trace metrics, and optional robot metrics

INSPECT treats `stable_auto` as weak supervision only. Strong evidence-graph edges are gated by
verification level:

```text
L0 unverified: no supervision edge
L1 stable_auto / consensus: weak suggested_by edge only
L2 reviewed / accepted: strong evidence candidate
L3 human-corrected: strong evidence candidate
L4 multi-view / postcondition verified: strongest evidence candidate
```

Use `StateSpec` files to prevent a step label from being treated as a verified procedural state.
A spec decomposes one nominal step into preconditions, interaction evidence, transition evidence,
postconditions, negative evidence, and next-step admissibility:

```json
{
  "states": [
    {
      "state_id": "GEAR_INSERTED",
      "nominal_step": "STEP_3",
      "preconditions": ["object:gear", "object:slot"],
      "interaction_evidence": ["contact:hand:gear"],
      "transition_evidence": ["transition:gear_to_slot"],
      "postcondition_evidence": ["relation:gear:inside:slot"],
      "negative_evidence": ["failure:gear_tilted", "failure:slot_empty"],
      "admissibility_evidence": ["relation:gear:inside:slot"],
      "next_states": ["PLACE_COVER"]
    }
  ]
}
```

Hand-object contact belongs in `interaction_evidence` or `transition_evidence`. It should not be
used as a postcondition by itself; the robot verifier still needs state-validating evidence such
as object-object relation, pose/alignment, absence of gaps, or an explicit postcondition cue.

The runtime pipeline now estimates hand/tool-object contact evidence before building each
`EvidenceToken`. Configure the detector vocabulary and contact thresholds with:

```yaml
interaction:
  enabled: true
  hand_names: ["hand", "left_hand", "right_hand", "glove"]
  tool_names: ["screwdriver", "wrench", "pliers", "tool"]
  contact_margin_px: 18
  near_margin_px: 48
  depth_contact_gap: 0.12
  min_contact_score: 0.35
  history: 6
```

For the strongest mask-based contact evidence, enable SAM3.1 concept segmentation. SAM3.1 is
preferred over SAM3/SAM2 here because Meta released it as a drop-in SAM3 update with more efficient
multi-object video tracking. SAM3 Video still uses a SAM2-style tracker internally for temporal
consistency, but SAM3.1 is the current best Meta checkpoint family for concept segmentation.

```yaml
segmentation:
  backend: "sam3.1"
  model_name: "facebook/sam3.1"
  device: "auto"
  prompts: ["hand", "glove", "screwdriver", "gear", "slot"]
  score_threshold: 0.45
  max_masks: 32
  fail_on_unavailable: true
```

If SAM3.1 is not installed or the checkpoint is unavailable, keep `segmentation.backend: "none"` for
the bbox/geometry fallback, or set `fail_on_unavailable: false` to fall back automatically. Use
`backend: "sam3_video_hf"` only when you specifically want the Hugging Face SAM3 Video API; the
SAM3.1 checkpoint currently uses the official Meta SAM3 repository rather than Transformers
integration. The segmentation backend only changes how contact evidence is grounded; INSPECT's
trace, graph, verifier, and robot loop remain the same.

Interactive feedback is also evidence-oriented when `review.evidence_prompt_enabled` is true.
The console still accepts or corrects the step label, but then asks whether the current
postcondition evidence is verified, rejected, occluded, or unknown. That extra supervision is
stored in `feedback.jsonl` under `feedback.extras.evidence_feedback` and is consumed by INSPECT.

If robot-view observations have already been converted into the same lightweight evidence
vocabulary, run the verifier:

```bash
python -m mica_step_modular inspect ^
  --run-dir path/to/run_dir ^
  --robot-observations path/to/robot_observations.jsonl
```

Each robot observation may provide any mix of:

```json
{
  "observation_id": "robot_0001",
  "prev_state": "STEP_2",
  "view_id": "front",
  "visible_counts": {"gear": 1, "slot": 1},
  "relevant_counts": {"gear": 1},
  "relation_facts": [["gear", "inside", "slot"]],
  "candidate_states": ["STEP_3"],
  "ground_truth_state": "STEP_3",
  "next_step_admissible": true
}
```

Optional active observation uses finite robot view candidates:

```bash
python -m mica_step_modular inspect ^
  --run-dir path/to/run_dir ^
  --robot-observations path/to/robot_observations.jsonl ^
  --view-candidates path/to/view_candidates.jsonl
```

View candidates list which evidence each candidate viewpoint is expected to reveal:

```json
{"view_id": "side_close", "visible_evidence": ["relation:gear:inside:slot"], "motion_cost": 0.2}
```

Train a learned calibrated verifier from labeled robot observations:

```bash
python -m mica_step_modular inspect-train-verifier ^
  --graph path/to/procedural_evidence_graph.json ^
  --robot-observations path/to/labeled_robot_observations.jsonl ^
  --model-out path/to/calibrated_verifier.json
```

Then use it instead of the rule-based verifier:

```bash
python -m mica_step_modular inspect ^
  --run-dir path/to/run_dir ^
  --state-specs path/to/state_specs.json ^
  --robot-observations path/to/robot_observations.jsonl ^
  --verifier-model path/to/calibrated_verifier.json
```

For real robot integration, use `RobotProceduralDecisionLoop` with a robot-specific adapter that
implements `observe`, `move_to_view`, `continue_task`, `pause_task`, and `ask_human`. The loop is
controller-agnostic: a ROS/MoveIt implementation only needs to translate these callbacks into
safe robot or active-camera commands.

## What changed relative to the old monolithic pipeline

- The runtime pipeline is split into small components with typed interfaces.
- The online fusion module keeps the lightweight `W / b / g` structure, but updates it with
  a margin-based online rule instead of the previous branch-heavy heuristic.
- Human acceptance is treated as weak positive supervision; explicit correction remains a
  stronger update.
- Stability uses both detection-signature consistency and step consistency, which produces
  cleaner feedback events.
- The package now uses a `GeometryProvider` abstraction, so `MoGe-2`, `UniDepthV2`, or any
  future geometry backend can be plugged in without changing the runtime loop.
- The package now includes an explicit episodic memory layer with:
  recall-before-decision, capture-after-verified-outcome, session memory, long-term memory,
  source diversification, and a dense memory prior expert for fusion.
- The package now builds a lightweight scene graph and feeds relation/contact/support features
  into the memory encoder.
- The gallery retriever and the memory subsystem now share the same visual encoder, so visual
  retrieval and episodic recall live in one feature space instead of drifting apart.
- The package now includes a lightweight governance layer:
  sparse review, structured runtime events, and a compact `Doing / Review / Blocked / Next`
  ops state. This lives outside the per-frame hot path and only activates on stable but
  uncertain windows.

## Folder layout

```text
mica_step_modular/
  __init__.py
  __main__.py
  cli.py
  config.py
  interfaces.py
  types.py
  components/
    detection.py
    geometry.py
    temporal.py
    context.py
    scene_graph.py
    kb.py
    rules.py
    retrieval.py
    visual_embedding.py
    calibration.py
    online_fusion.py
    stability.py
    feedback.py
    logging.py
    ui.py
    voice.py
  events/
    types.py
    bus.py
  review/
    types.py
    policy.py
    agent.py
    manager.py
  ops/
    types.py
    state.py
  memory/
    types.py
    features.py
    store.py
    retrieval.py
    recall_policy.py
    capture.py
    prior.py
    manager.py
  runtime/
    __init__.py
    live_assistant.py
    pipeline.py
  resources/
    config.example.yaml
```

## Quick start

```bash
python -m mica_step_modular ^
  --video path/to/video.mp4 ^
  --kb path/to/components.json ^
  --config mica_step_modular/resources/config.example.yaml ^
  --device 0 ^
  --interactive
```

If [`best.pt`](F:/Hackathon/ACVR/mica_step_modular/best.pt) exists inside the package, the CLI uses it by default, so
`--yolo-weights` becomes optional.

The CLI now supports both offline video files and a live camera:

```bash
python -m mica_step_modular ^
  --camera 0 ^
  --kb path/to/components.json ^
  --device cpu
```

In camera mode, the live assistant keeps the perception stack unchanged and adds:

- OpenCV windows for the main view and optional focus crop
- keyboard control for quit / pause / voice capture / mute / feedback prompt
- manual voice capture by default, with optional always-on listening

Live hotkeys:

- `Q`: quit
- `P` or `Space`: pause / resume
- `V`: manual voice capture
- `M` or `S`: mute / unmute the voice listener
- `F`: request a console feedback prompt on the next stable step
- `H`: toggle the help overlay

In `always_on` voice mode, non-command utterances are treated as questions for the contextual
assistant. It answers from the current step, visible parts, scene relations, memory state, and
knowledge-base snippets without adding a heavy LLM loop to the hot path.
By default, continuous listening uses a wake prefix such as `Mica ...` or `Assistant ...` before
it turns speech into a command or assistant query. Ambient speech is ignored so it cannot trigger
accidental online feedback.

If `experts.gallery_root` is empty, the runtime auto-discovers gallery folders in this order:

1. `mica_step_modular/gallery`
2. `test/after_online`
3. `test/before_online`

To enable `MoGe-2`, set the geometry backend in the YAML config:

```yaml
geometry:
  backend: "moge"
  moge_model: "Ruicheng/moge-2-vits-normal"
  use_fp16: true
  resolution_level: 8
  apply_mask: true
```

The memory system is enabled by default. It stores:

- `session_memory.jsonl` inside each run directory
- `long_term_memory.jsonl` under `runlog.save_dir/memory_bank/` unless you override `memory.long_term_path`

Recall is triggered on uncertain, disagreeing, or transitional states. Captured memories are
structured events, not raw text, and the retrieved matches are converted into a dense memory
prior that is fused with the state and retrieval experts.

The governance layer is also enabled by default. It adds:

- a sparse reviewer that only runs on stable but suspicious windows
- `events.jsonl` for replayable control-plane events
- `ops_state.json` and `ops_state_history.jsonl` for dashboard-style runtime state

Reviewer-generated corrections are weak hints stored in session memory only. They do not go to
long-term memory.

Example memory config:

```yaml
memory:
  preset: "custom"
  enabled: true
  session_enabled: true
  long_term_enabled: true
  auto_capture_enabled: true
  topk: 6
  max_per_source: 3
  recall_margin: 0.18
  recall_disagreement: 0.30
  min_auto_capture_confidence: 0.78
```

Example review config:

```yaml
review:
  enabled: true
  low_confidence: 0.62
  margin_threshold: 0.12
  disagreement_threshold: 0.34
  correction_streak: 2
  prefer_vote_count: 2
  reviewer_feedback_strength: 0.35
```

Example live camera + voice config:

```yaml
camera:
  index: 0
  backend: "auto"    # tries default, then DShow/MSMF fallback on Windows
  width: 1280
  height: 720
  warmup_frames: 8
  warmup_delay_ms: 40
  read_retry_limit: 60
  read_retry_delay_ms: 50

ui:
  enabled: true
  show_focus: true

voice:
  enabled: true
  mode: "manual"      # or "always_on"
  backend: "auto"     # prefers faster_whisper, falls back to transformers if needed
  model_name: "distil-small.en"
  cache_dir: "F:/Hackathon/ACVR/mica_step_modular/.cache/stt"
  language: "en"
  wake_words: ["mica", "assistant"]
  require_wake_word_in_always_on: true
  command_max_tokens: 6
  compute_type: "auto"
  cpu_threads: 0
assistant:
  enabled: true
  history_limit: 12
  relation_limit: 3
```

Example spoken-answer config:

```yaml
speech:
  enabled: true
  backend: "auto"     # or "kokoro"
  rate: 1
  volume: 1.0
  voice_name: ""
  kokoro_model_path: "F:/Hackathon/ACVR/mica_step_modular/models/kokoro-82m"
  kokoro_voices_path: ""
  kokoro_voice: "af_sarah"
  kokoro_language: "en-us"
  kokoro_speed: 1.0
  kokoro_python_path: "F:/Hackathon/ACVR/mica_step_modular/.venv_kokoro/Scripts/python.exe"
  interrupt_on_voice_input: true
```

Scene graph and shared visual encoder:

```yaml
scene_graph:
  enabled: true
  contact_depth_gap: 0.12
  support_overlap_ratio: 0.25

experts:
  gallery_embed: "hybrid-4"  # or "clip" when open_clip_torch is available
```

Ablation presets are available from either YAML (`memory.preset`) or CLI:

- `memory-off`
- `session-only`
- `long-term-only`
- `no-auto-capture`

Example:

```bash
python -m mica_step_modular ^
  --video path/to/video.mp4 ^
  --kb path/to/components.json ^
  --ablation-preset session-only
```

To override gallery discovery:

```bash
python -m mica_step_modular ^
  --video path/to/video.mp4 ^
  --kb path/to/components.json ^
  --gallery-root F:/Hackathon/ACVR/test/after_online
```

To run a batch ablation suite and emit CSV/Markdown tables:

```bash
python -m mica_step_modular ^
  --video path/to/video.mp4 ^
  --kb path/to/components.json ^
  --run-suite ^
  --suite-presets custom,memory-off,session-only,long-term-only,no-auto-capture ^
  --suite-embeds hybrid-4,clip
```

Install `MoGe-2` with:

```bash
pip install git+https://github.com/microsoft/MoGe.git
```

For live voice control, the default English command stack is `distil-small.en` with
`faster-whisper` as first priority. On machines where `ctranslate2` is unstable, the runtime
falls back to `transformers` with the same Distil-Whisper family model instead of crashing.
If no supported backend is available, camera mode still works, but voice control stays disabled.
Spoken answers are also offline by default. `speech.backend: "kokoro"` selects a local Kokoro
ONNX export and can point to a dedicated Python environment via `speech.kokoro_python_path`,
so the Kokoro stack does not need to share the main runtime environment. If Kokoro is not
available, the runtime can keep using `pyttsx3` or offline Windows `PowerShell/System.Speech`.

Example `always_on` live assistant:

```bash
python -m mica_step_modular ^
  --camera 0 ^
  --kb F:/Hackathon/ACVR/components.json ^
  --voice-mode always_on
```

Supported live assistant question types:

- current step / current status
- next step
- visible parts / counts
- scene relations
