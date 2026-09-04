#!/usr/bin/env python3
"""Local annotation server for fixed-lattice robot inspection data.

The server intentionally uses only the Python standard library. Annotations are
stored in SQLite with optimistic revision checks and an append-only audit log.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import os
import sqlite3
import sys
import threading
import time
import urllib.parse
import webbrowser
import zipfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
DEFAULT_MANIFEST = APP_DIR / "sample_manifest.json"
DEFAULT_DB = APP_DIR / "annotations.sqlite3"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("setups"), list):
        raise ValueError("Manifest must contain a top-level 'setups' array.")

    seen: set[tuple[str, str]] = set()
    for setup in manifest["setups"]:
        setup_id = str(setup.get("setup_id", "")).strip()
        if not setup_id:
            raise ValueError("Every setup requires a non-empty setup_id.")
        views = setup.get("views")
        if not isinstance(views, list) or not views:
            raise ValueError(f"Setup {setup_id!r} requires at least one view.")
        for view in views:
            view_id = str(view.get("view_id", "")).strip()
            image = str(view.get("image", "")).strip()
            if not view_id or not image:
                raise ValueError(f"Every view in setup {setup_id!r} requires view_id and image.")
            key = (setup_id, view_id)
            if key in seen:
                raise ValueError(f"Duplicate setup/view key: {setup_id}/{view_id}")
            seen.add(key)

    manifest.setdefault("project", {})
    manifest.setdefault("ontology", {})
    manifest["ontology"].setdefault("evidence_roles", [])
    manifest["ontology"].setdefault("object_classes", [])
    manifest["ontology"].setdefault("relation_types", [])
    manifest["ontology"].setdefault("keypoint_types", [])
    manifest["ontology"].setdefault("error_types", [])
    return manifest


class AnnotationStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS annotations (
                    entity_key TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL CHECK(entity_type IN ('setup', 'view')),
                    setup_id TEXT NOT NULL,
                    view_id TEXT,
                    claim_id TEXT,
                    payload TEXT NOT NULL,
                    workflow_status TEXT NOT NULL DEFAULT 'draft',
                    annotator TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_key TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    annotator TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_annotations_setup
                    ON annotations(setup_id, view_id);
                CREATE INDEX IF NOT EXISTS idx_audit_entity
                    ON audit_log(entity_key, id);
                """
            )

    def all_annotations(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM annotations ORDER BY setup_id, view_id"
            ).fetchall()
        return [self._decode_row(row) for row in rows]

    def save(
        self,
        *,
        entity_type: str,
        setup_id: str,
        view_id: str | None,
        claim_id: str | None,
        payload: dict[str, Any],
        workflow_status: str,
        annotator: str,
        expected_revision: int | None,
    ) -> dict[str, Any]:
        if entity_type not in {"setup", "view"}:
            raise ValueError("entity_type must be 'setup' or 'view'.")
        if not setup_id:
            raise ValueError("setup_id is required.")
        if entity_type == "view" and not view_id:
            raise ValueError("view_id is required for view annotations.")
        if workflow_status not in {"draft", "complete", "review"}:
            raise ValueError("workflow_status must be draft, complete, or review.")
        if not annotator.strip():
            raise ValueError("annotator is required.")

        entity_key = setup_id if entity_type == "setup" else f"{setup_id}::{view_id}::{claim_id or ''}"
        now = utc_now()
        encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

        with self._lock, self._conn:
            current = self._conn.execute(
                "SELECT revision FROM annotations WHERE entity_key = ?", (entity_key,)
            ).fetchone()
            current_revision = int(current["revision"]) if current else 0
            if expected_revision is not None and expected_revision != current_revision:
                raise RevisionConflict(current_revision)
            revision = current_revision + 1
            self._conn.execute(
                """
                INSERT INTO annotations(
                    entity_key, entity_type, setup_id, view_id, claim_id, payload,
                    workflow_status, annotator, revision, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_key) DO UPDATE SET
                    payload = excluded.payload,
                    workflow_status = excluded.workflow_status,
                    annotator = excluded.annotator,
                    revision = excluded.revision,
                    updated_at = excluded.updated_at
                """,
                (
                    entity_key,
                    entity_type,
                    setup_id,
                    view_id,
                    claim_id,
                    encoded,
                    workflow_status,
                    annotator.strip(),
                    revision,
                    now,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO audit_log(
                    entity_key, entity_type, action, payload, annotator, revision, created_at
                ) VALUES (?, ?, 'save', ?, ?, ?, ?)
                """,
                (entity_key, entity_type, encoded, annotator.strip(), revision, now),
            )
            row = self._conn.execute(
                "SELECT * FROM annotations WHERE entity_key = ?", (entity_key,)
            ).fetchone()
        return self._decode_row(row)

    def audit_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM audit_log ORDER BY id").fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["payload"] = json.loads(value["payload"])
        return value


class RevisionConflict(RuntimeError):
    def __init__(self, current_revision: int) -> None:
        super().__init__("Annotation changed in another session.")
        self.current_revision = current_revision


def flatten_setup_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = row["payload"]
    return {
        "setup_id": row["setup_id"],
        "workflow_status": row["workflow_status"],
        "annotator": row["annotator"],
        "revision": row["revision"],
        "updated_at": row["updated_at"],
        "assembly_family": payload.get("assembly_family", ""),
        "active_claim_id": payload.get("active_claim_id", ""),
        "physical_claim_truth": payload.get("physical_claim_truth", ""),
        "error_type": payload.get("error_type", ""),
        "target_part_identity": payload.get("target_part_identity", ""),
        "housing_identity": payload.get("housing_identity", ""),
        "cover_identity": payload.get("cover_identity", ""),
        "gear_orientation": payload.get("gear_orientation", ""),
        "insertion_state": payload.get("insertion_state", ""),
        "alignment_state": payload.get("alignment_state", ""),
        "cover_seating_state": payload.get("cover_seating_state", ""),
        "counterfactual_type": payload.get("counterfactual_type", ""),
        "notes": payload.get("notes", ""),
    }


def flatten_view_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = row["payload"]
    role_states = payload.get("role_states", {})
    return {
        "setup_id": row["setup_id"],
        "view_id": row["view_id"] or "",
        "claim_id": row["claim_id"] or "",
        "workflow_status": row["workflow_status"],
        "annotator": row["annotator"],
        "revision": row["revision"],
        "updated_at": row["updated_at"],
        "observable_decision": payload.get("observable_decision", ""),
        "oracle_utility": payload.get("oracle_utility", ""),
        "visible_evidence_roles": "|".join(
            sorted(key for key, state in role_states.items() if state == "visible")
        ),
        "missing_evidence_roles": "|".join(
            sorted(key for key, state in role_states.items() if state == "missing")
        ),
        "decisive_counterfactual": payload.get("decisive_counterfactual", ""),
        "occlusion_level": payload.get("occlusion_level", ""),
        "identity_ambiguous": payload.get("identity_ambiguous", ""),
        "object_visibility_json": json.dumps(payload.get("object_visibility", {}), ensure_ascii=True),
        "relations_json": json.dumps(payload.get("relations", {}), ensure_ascii=True),
        "boxes_json": json.dumps(payload.get("boxes", []), ensure_ascii=True),
        "keypoints_json": json.dumps(payload.get("keypoints", []), ensure_ascii=True),
        "short_rationale": payload.get("short_rationale", ""),
    }


def csv_content(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


class AppContext:
    def __init__(self, manifest_path: Path, media_root: Path, db_path: Path) -> None:
        self.manifest_path = manifest_path.resolve()
        self.manifest = load_manifest(self.manifest_path)
        self.media_root = media_root.resolve()
        self.store = AnnotationStore(db_path.resolve())

    def export_zip(self) -> bytes:
        annotations = self.store.all_annotations()
        setup_rows = [flatten_setup_row(row) for row in annotations if row["entity_type"] == "setup"]
        view_rows = [flatten_view_row(row) for row in annotations if row["entity_type"] == "view"]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(self.manifest, indent=2, ensure_ascii=True))
            archive.writestr("setup_annotations.csv", csv_content(setup_rows))
            archive.writestr("view_annotations.csv", csv_content(view_rows))
            archive.writestr("annotations.json", json.dumps(annotations, indent=2, ensure_ascii=True))
            archive.writestr("audit_log.json", json.dumps(self.store.audit_rows(), indent=2, ensure_ascii=True))
            archive.writestr(
                "export_metadata.json",
                json.dumps(
                    {
                        "exported_at": utc_now(),
                        "manifest": self.manifest_path.name,
                        "annotation_count": len(annotations),
                    },
                    indent=2,
                    ensure_ascii=True,
                ),
            )
        return buffer.getvalue()


class AnnotationHandler(BaseHTTPRequestHandler):
    server_version = "RobotAnnotationStudio/1.0"
    context: AppContext

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"[{self.log_date_time_string()}] {fmt % args}\n")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            return self._serve_static("index.html")
        if parsed.path.startswith("/static/"):
            return self._serve_static(parsed.path.removeprefix("/static/"))
        if parsed.path == "/api/bootstrap":
            return self._send_json(
                {
                    "manifest": self.context.manifest,
                    "annotations": self.context.store.all_annotations(),
                    "server_time": utc_now(),
                }
            )
        if parsed.path == "/api/health":
            return self._send_json({"ok": True, "time": utc_now()})
        if parsed.path == "/api/export":
            payload = self.context.export_zip()
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return self._send_bytes(
                payload,
                "application/zip",
                headers={"Content-Disposition": f'attachment; filename="robot_annotations_{stamp}.zip"'},
            )
        if parsed.path == "/media":
            query = urllib.parse.parse_qs(parsed.query)
            relative = query.get("path", [""])[0]
            return self._serve_media(relative)
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/save":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            body = self._read_json()
            saved = self.context.store.save(
                entity_type=str(body.get("entity_type", "")),
                setup_id=str(body.get("setup_id", "")).strip(),
                view_id=(str(body.get("view_id", "")).strip() or None),
                claim_id=(str(body.get("claim_id", "")).strip() or None),
                payload=body.get("payload") if isinstance(body.get("payload"), dict) else {},
                workflow_status=str(body.get("workflow_status", "draft")),
                annotator=str(body.get("annotator", "")).strip(),
                expected_revision=(
                    int(body["expected_revision"])
                    if body.get("expected_revision") is not None
                    else None
                ),
            )
        except RevisionConflict as exc:
            self._send_json(
                {"error": str(exc), "current_revision": exc.current_revision},
                status=HTTPStatus.CONFLICT,
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - last-resort server guard
            self._send_json({"error": f"Internal error: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
        else:
            self._send_json({"annotation": saved})

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 5_000_000:
            raise ValueError("Invalid request size.")
        raw = self.rfile.read(length)
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object.")
        return value

    def _serve_static(self, relative: str) -> None:
        target = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in target.parents and target != STATIC_DIR.resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send_bytes(target.read_bytes(), mime)

    def _serve_media(self, relative: str) -> None:
        if not relative:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing media path")
            return
        target = (self.context.media_root / urllib.parse.unquote(relative)).resolve()
        root = self.context.media_root
        if root not in target.parents and target != root:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send_bytes(target.read_bytes(), mime, cache=True)

    def _send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(json_bytes(value), "application/json; charset=utf-8", status=status)

    def _send_bytes(
        self,
        payload: bytes,
        content_type: str,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        headers: dict[str, str] | None = None,
        cache: bool = False,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "private, max-age=3600" if cache else "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed-lattice robot annotation studio.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--media-root", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8786)
    parser.add_argument("--open", action="store_true", help="Open the tool in the default browser.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    media_root = (args.media_root or manifest_path.parent).resolve()
    context = AppContext(manifest_path, media_root, args.db)
    AnnotationHandler.context = context
    server = ThreadingHTTPServer((args.host, args.port), AnnotationHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Robot Annotation Studio: {url}")
    print(f"Manifest: {manifest_path}")
    print(f"Media root: {media_root}")
    print(f"Database: {args.db.resolve()}")
    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
