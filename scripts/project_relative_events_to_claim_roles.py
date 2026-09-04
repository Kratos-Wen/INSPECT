"""Project transferable assistant view events onto claim-specific evidence roles.

This script does not create new before/after supervision and does not read robot
six-view images or oracle utilities. It only rewrites already mined first-person
assistant transitions so that generic visibility events can train the same
claim-specific roles used by the online selector.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspect_system.active_view.ontology import infer_role_from_fields, normalize_claim, normalize_key, roles_for_claim
from inspect_system.active_view.counterfactual_transport import infer_counterfactual_family


GENERIC_ROLES = {
    "object_presence_view",
    "claim_evidence_visibility_view",
    "claim_disambiguation_view",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover(paths: Sequence[str | Path]) -> List[Path]:
    out: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out.extend(sorted(path.glob("*.jsonl")))
        elif path.exists():
            out.append(path)
    seen = set()
    unique: List[Path] = []
    for path in out:
        key = str(path.resolve()).lower()
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def event_role(row: Mapping[str, Any]) -> str:
    return infer_role_from_fields(
        claim_id=row.get("claim_id") or row.get("claim_type"),
        evidence_view_type=row.get("evidence_view_type", ""),
        reason=row.get("reason", ""),
        missing_evidence=row.get("missing_evidence"),
        revealed_evidence=row.get("revealed_evidence"),
    )


def target_roles(claim_id: str, source_role: str, max_roles: int) -> List[tuple[str, float]]:
    roles = roles_for_claim(claim_id)
    if source_role not in GENERIC_ROLES and source_role in roles:
        return [(source_role, float(roles[source_role]))]
    ranked = sorted(roles.items(), key=lambda item: float(item[1]), reverse=True)
    if max_roles > 0:
        ranked = ranked[:max_roles]
    return [(str(role), float(weight)) for role, weight in ranked]


def projected_rows(rows: Iterable[Dict[str, Any]], max_roles: int) -> List[Dict[str, Any]]:
    projected: List[Dict[str, Any]] = []
    for row in rows:
        claim_id = normalize_claim(row.get("claim_id") or row.get("claim_type"))
        source_role = normalize_key(event_role(row))
        roles = target_roles(claim_id, source_role, max_roles)
        for role_index, (role, role_weight) in enumerate(roles, start=1):
            out = dict(row)
            meta = dict(out.get("metadata", {}) or {})
            out["claim_id"] = claim_id
            out["claim_type"] = claim_id
            out["evidence_view_type"] = role
            out["missing_evidence"] = [role]
            signed_gain = meta.get("signed_counterfactual_margin_gain")
            degrading = signed_gain not in (None, "") and float(signed_gain) <= 0.0
            world_outcome = normalize_key(
                meta.get("world_outcome") or out.get("outcome")
            )
            if degrading:
                out["revealed_evidence"] = []
                out["contradictory_evidence"] = []
            elif world_outcome == "contradicted":
                out["revealed_evidence"] = []
                out["contradictory_evidence"] = [role]
            else:
                out["revealed_evidence"] = [role]
                out["contradictory_evidence"] = []
            base_event_id = str(out.get("event_id", "event"))
            out["event_id"] = f"{base_event_id}_roleproj_{role_index:02d}_{role}"
            meta.update(
                {
                    "claim_role_projection": True,
                    "source_evidence_role": source_role,
                    "projected_evidence_role": role,
                    "evidence_importance": role_weight,
                    "uses_human_segment_boundary": False,
                    "uses_human_segment_for_outcome_context": bool(meta.get("uses_human_segment_boundary", False)),
                    "uses_robot_view_training": False,
                }
            )
            meta["counterfactual_family"] = infer_counterfactual_family(
                claim_id,
                role,
                meta,
            )
            out["metadata"] = meta
            projected.append(out)
    return projected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Relative event JSONL files or directories.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--max-roles", type=int, default=3, help="Project generic events to this many top ontology roles; 0 means all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover(args.input)
    input_rows: List[Dict[str, Any]] = []
    for path in files:
        input_rows.extend(read_jsonl(path))
    rows = projected_rows(input_rows, max_roles=args.max_roles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "input_files": [str(path) for path in files],
        "input_events": len(input_rows),
        "output_events": len(rows),
        "output": str(args.output),
        "by_role": {},
        "uses_robot_view_training": False,
    }
    for row in rows:
        role = str(row.get("evidence_view_type", ""))
        report["by_role"][role] = int(report["by_role"].get(role, 0)) + 1
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
