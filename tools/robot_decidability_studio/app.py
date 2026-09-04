#!/usr/bin/env python3
"""Local server for binary robot-view decidability annotation."""

from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import sqlite3
import threading
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest.get("setups"), list):
        raise ValueError("Manifest requires a setups list.")
    return manifest


class RevisionConflict(RuntimeError):
    def __init__(self, current_revision: int) -> None:
        super().__init__(f"Current revision is {current_revision}.")
        self.current_revision = current_revision


class AnnotationStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS annotations (
                    entity_key TEXT PRIMARY KEY,
                    setup_id TEXT NOT NULL,
                    view_id TEXT NOT NULL,
                    claim_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    workflow_status TEXT NOT NULL,
                    annotator TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_key TEXT NOT NULL,
                    action TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    workflow_status TEXT NOT NULL,
                    annotator TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def key(setup_id: str, view_id: str, claim_id: str) -> str:
        return f"{setup_id}::{view_id}::{claim_id}"

    @staticmethod
    def decode(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["payload"] = json.loads(value["payload"])
        return value

    def all(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM annotations ORDER BY setup_id, view_id"
            ).fetchall()
        return [self.decode(row) for row in rows]

    def save(
        self,
        *,
        setup_id: str,
        view_id: str,
        claim_id: str,
        payload: dict[str, Any],
        workflow_status: str,
        annotator: str,
        expected_revision: int | None,
        seed: bool = False,
    ) -> dict[str, Any]:
        if workflow_status not in {"draft", "complete", "review"}:
            raise ValueError("Invalid workflow status.")
        if not seed and not annotator.strip():
            raise ValueError("Annotator ID is required.")
        key = self.key(setup_id, view_id, claim_id)
        now = utc_now()
        encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        with self.lock, self.conn:
            current = self.conn.execute(
                "SELECT revision, payload FROM annotations WHERE entity_key=?", (key,)
            ).fetchone()
            current_revision = int(current["revision"]) if current else 0
            if expected_revision is not None and expected_revision != current_revision:
                raise RevisionConflict(current_revision)
            if current and not seed:
                old_payload = json.loads(current["payload"])
                if old_payload.get("decidability_locked"):
                    if payload.get("claim_decidable") != old_payload.get("claim_decidable"):
                        raise ValueError("Imported decidability labels are locked.")
            revision = current_revision + 1
            self.conn.execute(
                """
                INSERT INTO annotations(
                    entity_key,setup_id,view_id,claim_id,payload,workflow_status,
                    annotator,revision,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(entity_key) DO UPDATE SET
                    payload=excluded.payload,
                    workflow_status=excluded.workflow_status,
                    annotator=excluded.annotator,
                    revision=excluded.revision,
                    updated_at=excluded.updated_at
                """,
                (
                    key,
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
            self.conn.execute(
                """
                INSERT INTO audit_log(
                    entity_key,action,payload,workflow_status,annotator,revision,created_at
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (
                    key,
                    "seed" if seed else "save",
                    encoded,
                    workflow_status,
                    annotator.strip(),
                    revision,
                    now,
                ),
            )
            row = self.conn.execute(
                "SELECT * FROM annotations WHERE entity_key=?", (key,)
            ).fetchone()
        return self.decode(row)

    def audit(self) -> list[dict[str, Any]]:
        with self.lock:
            return [dict(row) for row in self.conn.execute(
                "SELECT * FROM audit_log ORDER BY id"
            ).fetchall()]


class AppContext:
    def __init__(self, manifest_path: Path, media_root: Path, db_path: Path) -> None:
        self.manifest_path = manifest_path.resolve()
        self.manifest = load_manifest(self.manifest_path)
        self.media_root = media_root.resolve()
        self.store = AnnotationStore(db_path.resolve())
        self.view_meta: dict[str, dict[str, Any]] = {}
        for setup in self.manifest["setups"]:
            for view in setup["views"]:
                key = AnnotationStore.key(setup["setup_id"], view["view_id"], setup["claim_id"])
                self.view_meta[key] = {
                    "assembly_family": setup.get("assembly_family", ""),
                    "target_step": setup.get("target_step", ""),
                    "claim_text": setup.get("claim_text", ""),
                }

    def export_zip(self) -> bytes:
        records = self.store.all()
        rows: list[dict[str, Any]] = []
        for record in records:
            payload = record["payload"]
            metadata = self.view_meta.get(record["entity_key"], {})
            decidable = payload.get("claim_decidable", "")
            occluded = payload.get("explicit_occlusion", "")
            rows.append(
                {
                    "trial_id": record["setup_id"],
                    "view_id": record["view_id"],
                    "target_step": metadata.get("target_step", ""),
                    "claim_id": record["claim_id"],
                    "assembly_family": metadata.get("assembly_family", ""),
                    "claim_decidable_0_1": "1" if decidable == "yes" else "0" if decidable == "no" else "",
                    "explicit_occlusion_0_1": "1" if occluded == "yes" else "0" if occluded == "no" else "",
                    "legacy_human_utility_0_1_2": payload.get("legacy_human_utility", ""),
                    "decidability_source": payload.get("decidability_source", ""),
                    "workflow_status": record["workflow_status"],
                    "flagged": int(bool(payload.get("flagged"))),
                    "notes": payload.get("notes", ""),
                    "annotator": record["annotator"],
                    "revision": record["revision"],
                    "updated_at": record["updated_at"],
                }
            )
        csv_buffer = io.StringIO()
        if rows:
            writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("view_decidability_annotations.csv", csv_buffer.getvalue())
            archive.writestr("view_decidability_annotations.json", json.dumps(records, indent=2))
            archive.writestr("audit_log.json", json.dumps(self.store.audit(), indent=2))
            archive.writestr("public_manifest.json", json.dumps(self.manifest, indent=2))
            archive.writestr(
                "export_metadata.json",
                json.dumps({"exported_at": utc_now(), "records": len(records)}, indent=2),
            )
        return output.getvalue()


class Handler(BaseHTTPRequestHandler):
    context: AppContext

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_bytes(json.dumps(value, ensure_ascii=True).encode(), "application/json", status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_json({"ok": True, "time": utc_now()})
            return
        if parsed.path == "/api/bootstrap":
            self.send_json({"manifest": self.context.manifest, "annotations": self.context.store.all()})
            return
        if parsed.path == "/api/export":
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.send_bytes(
                self.context.export_zip(),
                "application/zip",
                headers={"Content-Disposition": f'attachment; filename="view_labels_{stamp}.zip"'},
            )
            return
        if parsed.path == "/media":
            query = urllib.parse.parse_qs(parsed.query)
            self.serve_media(query.get("path", [""])[0])
            return
        self.serve_static(parsed.path)

    def do_POST(self) -> None:  # noqa: N802
        if urllib.parse.urlparse(self.path).path != "/api/save":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            value = json.loads(self.rfile.read(length) or b"{}")
            payload = value.get("payload") or {}
            decidable = payload.get("claim_decidable", "")
            occluded = payload.get("explicit_occlusion", "")
            status = value.get("workflow_status", "draft")
            if status == "complete":
                if decidable not in {"yes", "no"}:
                    raise ValueError("Decidability is required.")
                if decidable == "no" and occluded not in {"yes", "no"}:
                    raise ValueError("Occlusion is required when the claim is not decidable.")
            record = self.context.store.save(
                setup_id=str(value.get("setup_id", "")),
                view_id=str(value.get("view_id", "")),
                claim_id=str(value.get("claim_id", "")),
                payload=payload,
                workflow_status=status,
                annotator=str(value.get("annotator", "")),
                expected_revision=value.get("expected_revision"),
            )
            self.send_json(record)
        except RevisionConflict as exc:
            self.send_json({"error": str(exc), "current_revision": exc.current_revision}, HTTPStatus.CONFLICT)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def serve_static(self, path: str) -> None:
        relative = "index.html" if path in {"", "/"} else path.lstrip("/")
        target = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in target.parents and target != STATIC_DIR.resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_bytes(target.read_bytes(), content_type)

    def serve_media(self, relative: str) -> None:
        if not relative:
            self.send_error(HTTPStatus.BAD_REQUEST)
            return
        target = (self.context.media_root / relative).resolve()
        if self.context.media_root not in target.parents:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_bytes(target.read_bytes(), content_type)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    context = AppContext(args.manifest, args.media_root, args.db)
    Handler.context = context
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Robot View Decidability Studio: {url}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
