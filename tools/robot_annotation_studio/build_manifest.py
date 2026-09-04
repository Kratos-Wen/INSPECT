#!/usr/bin/env python3
"""Build a robot-annotation manifest from image folders and optional metadata."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIEW_PATTERNS = (
    re.compile(r"(?i)(?:^|[_-])v(?:iew)?[_-]?([0-9]+)(?:[_-]|$)"),
    re.compile(r"(?i)(?:^|[_-])cam(?:era)?[_-]?([0-9]+)(?:[_-]|$)"),
)


DEFAULT_ONTOLOGY = {
    "evidence_roles": [
        {"id": "identity", "label": "Identity"},
        {"id": "insertion", "label": "Insertion"},
        {"id": "containment", "label": "Containment"},
        {"id": "alignment", "label": "Alignment"},
        {"id": "slot_relation", "label": "Slot relation"},
        {"id": "boundary_gap", "label": "Boundary / gap"},
        {"id": "contact", "label": "Contact"},
        {"id": "occlusion_recovery", "label": "Occlusion recovery"},
    ],
    "object_classes": [
        {"id": "target_part", "label": "Target part"},
        {"id": "gear", "label": "Gear"},
        {"id": "housing", "label": "Housing"},
        {"id": "cover", "label": "Cover"},
        {"id": "slot", "label": "Slot"},
    ],
    "relation_types": [
        {"id": "inside", "label": "Inside housing"},
        {"id": "inserted", "label": "Inserted"},
        {"id": "aligned", "label": "Aligned with slot"},
        {"id": "seated", "label": "Cover seated"},
        {"id": "touching", "label": "Contact / touching"},
        {"id": "gap_visible", "label": "Gap visible"},
        {"id": "wrong_orientation_visible", "label": "Wrong orientation visible"},
    ],
    "keypoint_types": [
        {"id": "gear_center", "label": "Gear center"},
        {"id": "gear_orientation", "label": "Gear orientation marker"},
        {"id": "slot_center", "label": "Slot center"},
        {"id": "slot_axis", "label": "Slot axis marker"},
        {"id": "cover_boundary", "label": "Cover boundary"},
        {"id": "gap_endpoint", "label": "Gap endpoint"},
    ],
    "error_types": [
        "correct",
        "wrong_identity",
        "wrong_family",
        "wrong_orientation",
        "incomplete_insertion",
        "misalignment",
        "cover_gap",
        "part_absent",
    ],
}


def infer_view_id(path: Path) -> str | None:
    for pattern in VIEW_PATTERNS:
        match = pattern.search(path.stem)
        if match:
            return f"V{int(match.group(1))}"
    if path.parent.name.lower().startswith("v") and path.parent.name[1:].isdigit():
        return f"V{int(path.parent.name[1:])}"
    return None


def infer_setup_id(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if len(relative.parts) > 1:
        return relative.parts[0]
    stem = path.stem
    for pattern in VIEW_PATTERNS:
        stem = pattern.sub("_", stem)
    return stem.strip("_-") or path.stem


def read_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        setup_id = str(row.get("setup_id", "")).strip()
        if setup_id:
            output[setup_id] = {str(key): str(value or "") for key, value in row.items()}
    return output


def build_manifest(root: Path, metadata_path: Path | None) -> dict[str, Any]:
    root = root.resolve()
    metadata = read_metadata(metadata_path)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped: list[str] = []

    for image in sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
        view_id = infer_view_id(image)
        if view_id is None:
            skipped.append(str(image.relative_to(root)))
            continue
        setup_id = infer_setup_id(image, root)
        groups[setup_id].append(
            {
                "view_id": view_id,
                "image": image.relative_to(root).as_posix(),
            }
        )

    setups: list[dict[str, Any]] = []
    for setup_id, views in sorted(groups.items()):
        row = metadata.get(setup_id, {})
        required_roles = [item.strip() for item in row.get("required_roles", "").split("|") if item.strip()]
        setup: dict[str, Any] = {
            "setup_id": setup_id,
            "split": row.get("split", "unassigned"),
            "claim_id": row.get("claim_id", ""),
            "claim_text": row.get("claim_text", ""),
            "required_roles": required_roles,
            "views": sorted(views, key=lambda value: value["view_id"]),
        }
        prefill = {
            key: value
            for key, value in row.items()
            if key
            not in {"setup_id", "split", "claim_id", "claim_text", "required_roles"}
            and value != ""
        }
        if prefill:
            setup["setup_prefill"] = prefill
        setups.append(setup)

    return {
        "schema_version": 1,
        "project": {
            "name": "Fixed-Lattice Robot Inspection",
            "description": "Setup truth and per-view claim evidence annotations",
            "media_root_hint": str(root),
            "skipped_without_view_id": skipped,
        },
        "ontology": DEFAULT_ONTOLOGY,
        "setups": setups,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an annotation manifest from robot images.")
    parser.add_argument("--images", type=Path, required=True, help="Image root to scan recursively.")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional setup-level CSV metadata.")
    parser.add_argument("--output", type=Path, required=True, help="Output manifest JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args.images, args.metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote {len(manifest['setups'])} setups to {args.output.resolve()}")
    skipped = manifest["project"]["skipped_without_view_id"]
    if skipped:
        print(f"Skipped {len(skipped)} images without an identifiable view id.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
