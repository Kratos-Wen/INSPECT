# Robot Inspection Annotation Studio

Local annotation software for fixed-lattice robot inspection experiments. It separates physical setup truth from per-view visual evidence, stores every revision in SQLite, and exports analysis-ready CSV and JSON files.

## Design goals

- Annotate setup truth once and reuse it across all synchronized views.
- Judge each view independently in blind mode to avoid physical-state and policy-selection bias.
- Complete common view labels with one click or one key.
- Draw normalized bounding boxes and keypoints directly over robot images.
- Detect inconsistent utility, epistemic-state, role, and physical-truth combinations before completion.
- Preserve annotator identity, timestamps, revisions, and an append-only audit trail.
- Run locally with the Python standard library; no cloud service or third-party package is required.

## Prepare a manifest

Robot images should use an identifiable view token such as `V0`, `view_0`, or `camera_0`. A convenient layout is:

```text
robot_images/
  setup_001/
    setup_001_V0.jpg
    setup_001_V1.jpg
    ...
    setup_001_V5.jpg
```

Generate the manifest:

```powershell
python tools/robot_annotation_studio/build_manifest.py `
  --images path/to/robot_images `
  --metadata tools/robot_annotation_studio/metadata_template.csv `
  --output path/to/robot_annotation_manifest.json
```

The optional metadata CSV supplies claim text, required evidence roles, split, and setup-level prefills. Prefills are never silently accepted; an annotator must apply and review them in the setup-truth workflow.

If existing filenames do not contain a recognizable view token, create the manifest directly using the schema in `sample_manifest.json`. Media paths are relative to `--media-root`.

## Launch

```powershell
python tools/robot_annotation_studio/app.py `
  --manifest path/to/robot_annotation_manifest.json `
  --media-root path/to/robot_images `
  --db path/to/robot_annotations.sqlite3 `
  --open
```

The default address is `http://127.0.0.1:8786`. The server binds to localhost unless explicitly changed.

## Recommended workflow

1. Enter an annotator ID.
2. Open **Setup truth**, apply available metadata prefills, inspect the physical setup record, and complete it once.
3. Return to **View evidence** with blind mode enabled.
4. Use the quick decisions or `0`, `1`, `2`, `S`, `C`, and `I` keys.
5. Mark visible and missing evidence roles. Role buttons cycle through unrated, visible, and missing.
6. Add object visibility, relations, boxes, or keypoints only when required by the evaluation protocol.
7. Select **Complete & next**. Required consistency checks block completion; drafts continue to auto-save.
8. Use **Quality review** to resolve cross-field or physical-truth conflicts.
9. Select **Export** to download a ZIP containing flattened CSVs, full JSON, the manifest, and the audit log.

## Utility rubric

- `0`: no evidence that distinguishes the active claim from a counterfactual.
- `1`: relevant evidence is visible, but the claim is not decidable from this view.
- `2`: the view alone contains enough evidence to support or contradict the claim.

Utility `0` or `1` must be labeled `insufficient`. Utility `2` must be labeled `supported` or `contradicted` and name at least one visible evidence role.

## Keyboard shortcuts

| Key | Action |
|---|---|
| `0` | No evidence; insufficient |
| `1` | Partial evidence; insufficient |
| `2` | Fully decidable utility |
| `S` / `C` / `I` | Supported / contradicted / insufficient |
| `[` / `]` | Previous / next view |
| `K` / `J` | Previous / next setup |
| `B` / `P` / `E` | Box / point / erase tool |
| `Ctrl+S` | Save draft |
| `Enter` | Complete and move to next incomplete view |

## Export files

- `setup_annotations.csv`: physical setup truth.
- `view_annotations.csv`: per-view claim state, utility, evidence roles, relations, and geometry.
- `annotations.json`: complete structured records.
- `audit_log.json`: every saved revision.
- `manifest.json`: exact project input used for annotation.

## Test-set discipline

Keep every view of one physical setup in the same split. Robot ground truth and view utility remain evaluation-only for the assistant-supervised method. Candidate-view images must not enter online policy decisions. A robot-supervised diagnostic may train only on its declared calibration setups.
