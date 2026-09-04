# Existing INSPECT Robot Dataset

The current fixed-lattice source already contains 60 setup records, 360 synchronized robot images, and 132 completed human view-utility labels for the 22 active-inspection setups. Preparation preserves these records as reviewable drafts instead of asking annotators to repeat completed work.

## Prepare once

```powershell
python tools/robot_annotation_studio/prepare_existing_inspect_dataset.py
```

This creates:

```text
outputs/robot_annotation_studio/
  robot_annotation_manifest.json
  robot_annotations.sqlite3
  preparation_summary.json
  launch_annotation_studio.ps1
```

The 60 setup ground-truth rows are prefilled from `robot_trial_gt.csv`. The 132 existing human utility labels are imported from `robot_view_utility_annotations.csv` as drafts with source metadata. No imported item is marked complete automatically.

## Launch thereafter

```powershell
powershell -ExecutionPolicy Bypass -File outputs/robot_annotation_studio/launch_annotation_studio.ps1
```

Do not rerun preparation after annotation starts. The preparer refuses to replace an existing database unless `--overwrite-db` is supplied explicitly; when replacement is requested, it first writes a `.bak` copy.

## Efficient assignment

- **Active inspection first:** `trial_025`--`trial_040` and `trial_055`--`trial_060`. Their utility labels already exist; review evidence roles and relations.
- **Robot diagnostic second:** Step-1 trials. Complete object visibility, identity ambiguity, boxes, and identity evidence.
- **Geometry subset:** Draw keypoints only where relation, orientation, insertion, or gap evidence is evaluated. Do not draw redundant points on every image.
- **Independent quality sample:** use a separate SQLite database for the second annotator on the prespecified overlap subset. Never share a mutable annotation row between independent annotators.
