"""Create a path-free SHA-256 manifest for a local model checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(model_root: Path, model_id: str) -> dict:
    root = model_root.resolve()
    files = []
    aggregate = hashlib.sha256()
    candidates = (
        item
        for item in root.rglob("*")
        if item.is_file()
        and not any(part in {".cache", ".git", "__pycache__"} for part in item.relative_to(root).parts)
    )
    for path in sorted(candidates, key=lambda p: p.as_posix()):
        relative_name = path.relative_to(root).as_posix()
        digest = sha256(path)
        size = path.stat().st_size
        aggregate.update(f"{relative_name}\0{size}\0{digest}\n".encode("utf-8"))
        files.append({"path": relative_name, "bytes": size, "sha256": digest})
    if not files:
        raise ValueError(f"No checkpoint files found under {root}")
    return {
        "schema": "inspect.model-checkpoint-fingerprint.v1",
        "model_id": model_id,
        "aggregate_sha256": aggregate.hexdigest(),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.model_root, args.model_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(args.output)
    print(f"files={len(manifest['files'])}")
    print(f"aggregate_sha256={manifest['aggregate_sha256']}")


if __name__ == "__main__":
    main()
