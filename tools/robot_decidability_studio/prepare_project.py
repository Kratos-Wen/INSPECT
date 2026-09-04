#!/usr/bin/env python3
"""Prepare the deduplicated 360-view robot annotation project."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from app import AnnotationStore


DEFAULT_DATA_ROOT = Path("data/robot_view_trials")
DEFAULT_OUTPUT = Path("outputs/robot_decidability_studio")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def build_manifest(data_root: Path) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, str]]]:
    gt_rows = read_csv(data_root / "robot_trial_gt.csv")
    view_rows = read_csv(data_root / "view_manifest.csv")
    utility_rows = read_csv(data_root / "robot_view_utility_annotations.csv")
    gt = {row["trial_id"]: row for row in gt_rows}
    utility = {(row["trial_id"], row["view_id"]): row for row in utility_rows}
    views_by_trial: dict[str, list[dict[str, Any]]] = {}
    for row in view_rows:
        ordered = Path(row["ordered_path"])
        relative = ordered.resolve().relative_to(data_root.resolve()).as_posix()
        views_by_trial.setdefault(row["trial_id"], []).append(
            {
                "view_id": row["view_id"],
                "view_name": row.get("view_name", ""),
                "elevation": row.get("elevation", ""),
                "azimuth": row.get("azimuth", ""),
                "image": relative,
            }
        )
    for views in views_by_trial.values():
        views.sort(key=lambda item: int(item["view_id"].lstrip("V")))

    setups: list[dict[str, Any]] = []
    for trial_id, row in gt.items():
        imported = sum(
            1
            for view in views_by_trial[trial_id]
            if utility.get((trial_id, view["view_id"]), {}).get("human_utility_0_1_2", "") in {"0", "1", "2"}
        )
        pending_imported = sum(
            1
            for view in views_by_trial[trial_id]
            if utility.get((trial_id, view["view_id"]), {}).get("human_utility_0_1_2", "") in {"0", "1"}
        )
        priority = 0 if pending_imported else 1 if imported else 2
        setups.append(
            {
                "setup_id": trial_id,
                "trial_index": int(row["trial_index"]),
                "assembly_family": row["product_variant"].strip().upper(),
                "target_step": row["target_step"],
                "claim_id": row["claim_id"],
                "claim_text": row["claim_text"],
                "views": views_by_trial[trial_id],
                "task_priority": priority,
                "imported_view_count": imported,
            }
        )
    setups.sort(key=lambda item: (item["task_priority"], item["trial_index"]))
    manifest = {
        "schema_version": 2,
        "project": {
            "name": "INSPECT Robot View Decidability",
            "annotation_unit": "binary_decidability_then_occlusion",
            "total_setups": len(setups),
            "total_views": sum(len(item["views"]) for item in setups),
            "family_reference": {
                "A": {"housing_cover": "Type 5", "small_gear": "Type 3", "big_gear": "Type 8"},
                "B": {"housing_cover": "Type 6", "small_gear": "Type 7", "big_gear": "Type 2"},
            },
        },
        "setups": setups,
    }
    return manifest, utility


def seed_database(
    db_path: Path,
    manifest: dict[str, Any],
    utility: dict[tuple[str, str], dict[str, str]],
) -> dict[str, int]:
    store = AnnotationStore(db_path)
    counts = {"legacy_decidable_complete": 0, "legacy_occlusion_pending": 0, "new_decidability_pending": 0}
    for setup in manifest["setups"]:
        for view in setup["views"]:
            row = utility.get((setup["setup_id"], view["view_id"]), {})
            old = row.get("human_utility_0_1_2", "").strip()
            if old == "2":
                decidable = "yes"
                status = "complete"
                counts["legacy_decidable_complete"] += 1
            elif old in {"0", "1"}:
                decidable = "no"
                status = "draft"
                counts["legacy_occlusion_pending"] += 1
            else:
                decidable = ""
                status = "draft"
                counts["new_decidability_pending"] += 1
            payload = {
                "claim_decidable": decidable,
                "explicit_occlusion": "",
                "legacy_human_utility": old,
                "decidability_locked": bool(old),
                "decidability_source": "legacy_human_utility" if old else "new_annotation",
                "flagged": False,
                "notes": "",
                "legacy_provenance": {
                    "annotator": row.get("annotator", ""),
                    "updated_at": row.get("updated_at", ""),
                } if old else {},
            }
            store.save(
                setup_id=setup["setup_id"],
                view_id=view["view_id"],
                claim_id=setup["claim_id"],
                payload=payload,
                workflow_status=status,
                annotator=row.get("annotator", "").strip() or "project_seed",
                expected_revision=0,
                seed=True,
            )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite-db", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "public_manifest.json"
    db_path = output / "annotations.sqlite3"
    if db_path.exists():
        if not args.overwrite_db:
            raise SystemExit(f"Database exists: {db_path}. Refusing to overwrite annotation work.")
        shutil.copy2(db_path, db_path.with_suffix(".sqlite3.bak"))
        db_path.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(db_path) + suffix)
            if sidecar.exists():
                sidecar.unlink()
    manifest, utility = build_manifest(data_root)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    counts = seed_database(db_path, manifest, utility)
    launcher = output / "launch.ps1"
    launcher.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        "$repo = Resolve-Path (Join-Path $PSScriptRoot '..\\..')\n"
        "$app = Join-Path $repo 'tools\\robot_decidability_studio\\app.py'\n"
        "$manifest = Join-Path $PSScriptRoot 'public_manifest.json'\n"
        "$db = Join-Path $PSScriptRoot 'annotations.sqlite3'\n"
        f"$media = '{data_root}'\n"
        "python $app --manifest $manifest --media-root $media --db $db --port 8787 --open\n",
        encoding="utf-8",
    )
    summary = {
        "manifest": str(manifest_path),
        "database": str(db_path),
        "launcher": str(launcher),
        "setups": len(manifest["setups"]),
        "views": sum(len(item["views"]) for item in manifest["setups"]),
        **counts,
    }
    (output / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
