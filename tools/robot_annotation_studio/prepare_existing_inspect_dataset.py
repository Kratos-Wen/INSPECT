#!/usr/bin/env python3
"""Create the annotation project for the existing 60-by-6 INSPECT dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from app import AnnotationStore


DEFAULT_ROOT = Path("data/robot_view_trials")
DEFAULT_OUTPUT = Path("outputs/robot_annotation_studio")

EVIDENCE_ROLES = [
    {"id": "boundary_alignment_view", "label": "Boundary alignment", "scope": "paper"},
    {"id": "claim_disambiguation_view", "label": "Claim disambiguation", "scope": "paper"},
    {"id": "contact_verification_view", "label": "Contact verification", "scope": "paper"},
    {"id": "containment_verification_view", "label": "Containment verification", "scope": "paper"},
    {"id": "gap_visibility_view", "label": "Gap visibility", "scope": "paper"},
    {"id": "identity_disambiguation_view", "label": "Identity disambiguation", "scope": "paper"},
    {"id": "insertion_verification_view", "label": "Insertion verification", "scope": "paper"},
    {"id": "slot_relation_view", "label": "Slot relation", "scope": "paper"},
    {"id": "occlusion_recovery_view", "label": "Occlusion recovery", "scope": "runtime_fallback"},
]

OBJECT_CLASSES = [
    {"id": "type_2_gear", "label": "Type 2 gear"},
    {"id": "type_3_gear", "label": "Type 3 gear"},
    {"id": "type_7_gear", "label": "Type 7 gear"},
    {"id": "type_8_gear", "label": "Type 8 gear"},
    {"id": "type_5_gearbox_housing", "label": "Type 5 housing"},
    {"id": "type_6_gearbox_housing", "label": "Type 6 housing"},
    {"id": "type_5_gearbox_cover", "label": "Type 5 cover"},
    {"id": "type_6_gearbox_cover", "label": "Type 6 cover"},
]

RELATION_TYPES = [
    {"id": "inside_housing", "label": "Target inside housing"},
    {"id": "aligned_with_housing", "label": "Target aligned with housing / slot"},
    {"id": "gear_mesh_contact", "label": "Gear mesh / contact"},
    {"id": "cover_housing_contact", "label": "Cover contacts housing"},
    {"id": "cover_housing_overlap", "label": "Cover overlaps housing boundary"},
    {"id": "cover_seated", "label": "Cover fully seated"},
    {"id": "gap_visible", "label": "Assembly gap visible"},
    {"id": "wrong_orientation_visible", "label": "Wrong orientation visible"},
]

KEYPOINT_TYPES = [
    {"id": "target_center", "label": "Target-part center"},
    {"id": "target_orientation_marker", "label": "Target orientation marker"},
    {"id": "slot_center", "label": "Slot center"},
    {"id": "slot_axis_marker", "label": "Slot axis marker"},
    {"id": "housing_center", "label": "Housing center"},
    {"id": "cover_boundary", "label": "Cover boundary point"},
    {"id": "gap_endpoint", "label": "Gap endpoint"},
]

CLAIM_ROLES = {
    "step1_component_identity": ["identity_disambiguation_view", "claim_disambiguation_view"],
    "step2_small_gear_inserted": [
        "identity_disambiguation_view",
        "insertion_verification_view",
        "containment_verification_view",
        "slot_relation_view",
        "claim_disambiguation_view",
    ],
    "step3_big_gear_inserted": [
        "identity_disambiguation_view",
        "insertion_verification_view",
        "containment_verification_view",
        "slot_relation_view",
        "claim_disambiguation_view",
    ],
    "step4_cover_seated": [
        "gap_visibility_view",
        "boundary_alignment_view",
        "contact_verification_view",
        "claim_disambiguation_view",
    ],
}

EXPECTED_TARGET = {
    ("A", "step2"): "type_3_gear",
    ("A", "step3"): "type_8_gear",
    ("A", "step4"): "type_5_gearbox_cover",
    ("B", "step2"): "type_7_gear",
    ("B", "step3"): "type_2_gear",
    ("B", "step4"): "type_6_gearbox_cover",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def semicolon_items(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(";") if item.strip()]


def normalized_error(value: str) -> str:
    text = str(value or "").strip()
    return "correct" if text in {"", "none"} else text


def expected_target(row: dict[str, str]) -> str:
    key = (row.get("product_variant", "").strip().upper(), row.get("target_step", "").strip().lower())
    return EXPECTED_TARGET.get(key, "")


def actual_target(row: dict[str, str]) -> str:
    parts = semicolon_items(row.get("actual_parts", ""))
    step = row.get("target_step", "").strip().lower()
    if step == "step4":
        candidates = [part for part in parts if part.endswith("_cover")]
    elif step in {"step2", "step3"}:
        candidates = [part for part in parts if part.endswith("_gear")]
    else:
        candidates = []
    expected = expected_target(row)
    if step == "step3" and expected in candidates:
        return expected
    return candidates[-1] if candidates else ""


def setup_prefill(row: dict[str, str]) -> dict[str, Any]:
    step = row.get("target_step", "").strip().lower()
    relation = row.get("relation_state", "").strip().lower()
    error = normalized_error(row.get("error_type", ""))
    outcome = row.get("claim_outcome", "").strip().lower()
    target = actual_target(row)
    truth = "true" if outcome == "supported" else "false" if outcome == "contradicted" else ""
    orientation = "wrong" if "wrong_orientation" in error else ""
    if not orientation and step in {"step2", "step3"} and relation in {"complete", "incomplete"}:
        orientation = "correct"
    if step in {"step1", "step4"}:
        orientation = "not_applicable"
    insertion = "not_applicable"
    if step in {"step2", "step3"}:
        insertion = "partial" if relation == "incomplete" else "full" if relation in {"complete", "reverse"} else ""
    alignment = "not_applicable" if relation == "not_applicable" else "aligned" if relation == "complete" else "misaligned"
    seating = "not_applicable"
    if step == "step4":
        seating = "seated" if relation == "complete" else "partial" if relation == "incomplete" else "unseated" if relation == "reverse" else ""
    return {
        "assembly_family": row.get("product_variant", "").strip(),
        "active_claim_id": row.get("claim_id", "").strip(),
        "physical_claim_truth": truth,
        "error_type": error,
        "target_part_identity": target,
        "expected_target_identity": expected_target(row),
        "housing_identity": row.get("housing_class", "").strip(),
        "cover_identity": target if target.endswith("_cover") else "",
        "gear_orientation": orientation,
        "insertion_state": insertion,
        "alignment_state": alignment,
        "cover_seating_state": seating,
        "counterfactual_type": error if error != "correct" else "negated_claim",
        "notes": row.get("raw_note", "").strip(),
        "source_metadata": dict(row),
        "prefill_provenance": "robot_trial_gt.csv:user_gt",
    }


def role_states() -> dict[str, str]:
    return {item["id"]: "unrated" for item in EVIDENCE_ROLES}


def legacy_view_prefill(row: dict[str, str], required_roles: list[str]) -> dict[str, Any] | None:
    utility = row.get("human_utility_0_1_2", "").strip()
    if utility not in {"0", "1", "2"}:
        return None
    outcome = row.get("claim_outcome", "").strip().lower()
    decision = "insufficient" if utility in {"0", "1"} else outcome
    return {
        "oracle_utility": utility,
        "observable_decision": decision if decision in {"supported", "contradicted", "insufficient"} else "",
        "role_states": role_states(),
        "object_visibility": {},
        "relations": {},
        "occlusion_level": "",
        "identity_ambiguous": False,
        "decisive_counterfactual": row.get("error_type", "").strip(),
        "short_rationale": row.get("human_notes", "").strip(),
        "boxes": [],
        "keypoints": [],
        "required_roles_snapshot": list(required_roles),
        "legacy_utility_provenance": {
            "source": "robot_view_utility_annotations.csv",
            "annotator": row.get("annotator", "").strip(),
            "updated_at": row.get("updated_at", "").strip(),
            "human_utility_0_1_2": utility,
        },
    }


def build_project(data_root: Path) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, str]]]:
    gt_rows = read_csv(data_root / "robot_trial_gt.csv")
    view_rows = read_csv(data_root / "view_manifest.csv")
    utility_rows = read_csv(data_root / "robot_view_utility_annotations.csv")
    gt_by_trial = {row["trial_id"].strip(): row for row in gt_rows}
    utility_by_key = {(row["trial_id"].strip(), row["view_id"].strip().upper()): row for row in utility_rows}
    views_by_trial: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in view_rows:
        trial = row.get("trial_id", "").strip()
        source = Path(row.get("ordered_path", "")).resolve()
        try:
            image = source.relative_to(data_root).as_posix()
        except ValueError:
            image = str(source)
        views_by_trial[trial].append({
            "view_id": row.get("view_id", "").strip().upper(),
            "image": image,
            "view_name": row.get("view_name", ""),
            "elevation": row.get("elevation", ""),
            "azimuth": row.get("azimuth", ""),
            "source_metadata": dict(row),
        })

    setups = []
    for trial in sorted(gt_by_trial):
        gt = gt_by_trial[trial]
        claim = gt.get("claim_id", "").strip()
        setups.append({
            "setup_id": trial,
            "split": "active_inspection_test" if gt.get("target_step", "").strip().lower() != "step1" else "robot_diagnostic",
            "claim_id": claim,
            "claim_text": gt.get("claim_text", "").strip(),
            "required_roles": CLAIM_ROLES.get(claim, ["claim_disambiguation_view"]),
            "target_step": gt.get("target_step", "").strip(),
            "setup_prefill": setup_prefill(gt),
            "views": sorted(views_by_trial[trial], key=lambda item: item["view_id"]),
        })

    errors = sorted({normalized_error(row.get("error_type", "")) for row in gt_rows})
    manifest = {
        "schema_version": 1,
        "project": {
            "name": "INSPECT Fixed-Lattice Robot Annotation",
            "description": "60 physical setups with six calibrated views per setup",
            "source_root": str(data_root),
            "setup_count": len(setups),
            "view_count": sum(len(item["views"]) for item in setups),
            "legacy_human_utility_count": sum(1 for row in utility_rows if row.get("human_utility_0_1_2", "").strip() in {"0", "1", "2"}),
        },
        "ontology": {
            "evidence_roles": EVIDENCE_ROLES,
            "object_classes": OBJECT_CLASSES,
            "relation_types": RELATION_TYPES,
            "keypoint_types": KEYPOINT_TYPES,
            "error_types": errors,
        },
        "setups": setups,
    }
    return manifest, utility_by_key


def seed_database(manifest: dict[str, Any], utilities: dict[tuple[str, str], dict[str, str]], database: Path) -> dict[str, int]:
    store = AnnotationStore(database)
    setup_count = 0
    view_count = 0
    for setup in manifest["setups"]:
        store.save(
            entity_type="setup", setup_id=setup["setup_id"], view_id=None, claim_id=None,
            payload=json.loads(json.dumps(setup["setup_prefill"])), workflow_status="draft",
            annotator="metadata_import", expected_revision=0,
        )
        setup_count += 1
        for view in setup["views"]:
            source = utilities.get((setup["setup_id"], view["view_id"]), {})
            payload = legacy_view_prefill(source, setup["required_roles"])
            if payload is None:
                continue
            store.save(
                entity_type="view", setup_id=setup["setup_id"], view_id=view["view_id"],
                claim_id=setup["claim_id"], payload=payload, workflow_status="draft",
                annotator=source.get("annotator", "").strip() or "legacy_utility_import", expected_revision=0,
            )
            view_count += 1
    return {"setup_prefills": setup_count, "legacy_view_prefills": view_count}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite-db", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    required = ["robot_trial_gt.csv", "view_manifest.csv", "robot_view_utility_annotations.csv"]
    missing = [name for name in required if not (data_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "robot_annotation_manifest.json"
    database_path = output / "robot_annotations.sqlite3"
    if database_path.exists():
        if not args.overwrite_db:
            raise FileExistsError(f"Database exists: {database_path}. Existing work was not modified.")
        backup = database_path.with_suffix(".sqlite3.bak")
        shutil.copy2(database_path, backup)
        database_path.unlink()

    manifest, utilities = build_project(data_root)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    counts = seed_database(manifest, utilities, database_path)
    escaped_media = str(data_root).replace("'", "''")
    launcher = output / "launch_annotation_studio.ps1"
    launcher.write_text("\n".join([
        "$ErrorActionPreference = 'Stop'",
        "$repo = Resolve-Path (Join-Path $PSScriptRoot '..\\..')",
        "$app = Join-Path $repo 'tools\\robot_annotation_studio\\app.py'",
        "$manifest = Join-Path $PSScriptRoot 'robot_annotation_manifest.json'",
        "$db = Join-Path $PSScriptRoot 'robot_annotations.sqlite3'",
        f"$media = '{escaped_media}'",
        "python $app --manifest $manifest --media-root $media --db $db --open",
        "",
    ]), encoding="utf-8")
    summary = {
        "manifest": str(manifest_path), "database": str(database_path), "launcher": str(launcher),
        "setups": len(manifest["setups"]), "views": sum(len(item["views"]) for item in manifest["setups"]),
        **counts,
    }
    (output / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
