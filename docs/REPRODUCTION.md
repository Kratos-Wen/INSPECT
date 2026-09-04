# Reproduction guide

This repository contains the executable pipeline and evaluation code. It does
not contain participant videos, robot images, annotations, model checkpoints,
or hosted-model response caches.

## Data contracts

### Object evidence

`scripts/evaluate_yolo_gt_dataset.py` expects a YOLO-format dataset:

```text
<dataset-root>/
  classes.txt
  images/<split>/*
  labels/<split>/*.txt
```

Each label row is `class_id cx cy width height` in normalized coordinates.
Every detector in a comparison must use the same split, confidence threshold,
NMS mode, and IoU matching threshold.

### Assistant traces

The runtime writes structured records below its configured `runlog.save_dir`.
The trace builder consumes `iterations.jsonl`, `feedback.jsonl`, memory
signals, and scene-evidence records. Ground-truth timelines used for evaluation
remain external to the runtime.

### Robot views

Robot observations and view candidates use JSON or JSONL. A robot observation
contains an observation id, current view id, active or candidate claim states,
and evidence extracted from the current RGB frame. A candidate contains a view
id and calibrated lattice geometry. Candidate RGB images are not policy inputs.

The annotation interfaces can prepare a project from:

```text
data/robot_view_trials/
  robot_trial_gt.csv
  view_manifest.csv
  robot_view_utility_annotations.csv
```

Run `python tools/robot_annotation_studio/prepare_existing_inspect_dataset.py
--help` or `python tools/robot_decidability_studio/prepare_project.py --help`
for schema-specific options.

## Core evaluations

Detector evidence:

```bash
python scripts/evaluate_yolo_gt_dataset.py \
  --dataset-root data/object_evidence \
  --split test \
  --weights compact=models/detector.pt \
  --confs 0.10 \
  --device 0 \
  --output-dir outputs/evaluation/object_evidence
```

Claim verification and selective risk:

```bash
python scripts/evaluate_relation_aware_claim_verifier.py --help
python scripts/evaluate_verifier_risk_coverage.py --help
python scripts/evaluate_active_claim_cascade.py --help
```

Assistant-derived inspection:

```bash
python scripts/project_relative_events_to_claim_roles.py --help
python scripts/train_object_centric_reveal.py --help
python scripts/evaluate_decidability_gated_transport.py --help
python scripts/evaluate_robot_online_closed_loop.py --help
python scripts/evaluate_assistance_event_learning_curve.py --help
```

External IMPACT evaluation:

```bash
python scripts/external_benchmarks/prepare_impact.py --help
python scripts/external_benchmarks/extract_impact_rgb.py --help
python scripts/external_benchmarks/build_impact_train_graph.py --help
python scripts/external_benchmarks/evaluate_impact_reveal_policy.py --help
```

## Hosted VLM baselines

`scripts/evaluate_vlm_assistant_baselines.py` supports direct-image and
procedure-context protocols. Set the provider credential in the process
environment, use `--balanced-per-class` to cap cost, and use `--resume` to
reuse completed responses. The cache key includes provider, model, protocol,
sample identity, and prompt fingerprint.

```bash
python scripts/evaluate_vlm_assistant_baselines.py \
  --provider openai \
  --protocol procedure_context \
  --summary-csv data/assistant_eval/summary.csv \
  --timeline-csv data/assistant_eval/timeline.csv \
  --evidence-scorer models/assistant/evidence_scorer.json \
  --balanced-per-class 20 \
  --output-json outputs/vlm/openai_summary.json \
  --output-csv outputs/vlm/openai_rows.csv \
  --resume
```

Do not commit response caches: they can contain image paths, prompts, or
provider metadata.

## Protocol checks

Before reporting results:

1. Keep train/calibration/test groups disjoint at the setup or session level.
2. Record checkpoint hashes and command-line settings.
3. Use the same eligible examples and metric implementation for every row.
4. Confirm that the view policy never reads candidate images or robot utility
   labels.
5. Store all generated files below `outputs/`.
