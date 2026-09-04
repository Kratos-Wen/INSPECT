"""Verify and selectively extract one RGB view from an IMPACT release ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe ZIP member: {name}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view", choices=("ego", "front", "left", "right", "top"), required=True)
    parser.add_argument("--expected-sha256", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    actual_sha256 = sha256(args.archive)
    if args.expected_sha256 and actual_sha256.lower() != args.expected_sha256.lower():
        raise RuntimeError(
            f"SHA-256 mismatch for {args.archive}: {actual_sha256} != {args.expected_sha256}"
        )

    output_view = args.output / args.view
    output_view.mkdir(parents=True, exist_ok=True)
    records = []
    with ZipFile(args.archive) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.lower().endswith(f"_{args.view}.mp4"):
                continue
            member = safe_path(info.filename)
            destination = output_view / member.name
            if destination.exists() and destination.stat().st_size == info.file_size:
                records.append({"member": info.filename, "path": str(destination), "size": info.file_size})
                continue
            with archive.open(info) as source, destination.open("wb") as target:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    target.write(chunk)
            records.append({"member": info.filename, "path": str(destination), "size": info.file_size})

    if not records:
        raise RuntimeError(f"No {args.view!r} RGB videos found in {args.archive}")
    manifest = {
        "dataset": "IMPACT",
        "release": "v1.1",
        "view": args.view,
        "source_archive": str(args.archive),
        "source_sha256": actual_sha256,
        "videos": len(records),
        "bytes": sum(int(item["size"]) for item in records),
        "records": records,
    }
    manifest_path = args.output / f"extract_{args.view}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"videos": len(records), "gib": round(manifest["bytes"] / 1024**3, 3), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
