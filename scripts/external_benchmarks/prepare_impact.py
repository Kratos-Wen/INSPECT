"""Extract the RGB and annotation subset from an IMPACT quick-start archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo


KEEP_MARKERS = (
    "/sample/annotations/",
    "/sample/videos/",
)
KEEP_NAMES = {
    "IMPACT-v1.1/CHANGELOG.md",
    "IMPACT-v1.1/LICENSE-DATA",
    "IMPACT-v1.1/README.md",
    "IMPACT-v1.1/VERSION",
    "IMPACT-v1.1/sample/CHANGELOG.md",
    "IMPACT-v1.1/sample/LICENSE-DATA",
    "IMPACT-v1.1/sample/VERSION",
}


def safe_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member: {name}")
    return path


def should_extract(info: ZipInfo) -> bool:
    if info.is_dir():
        return False
    return info.filename in KEEP_NAMES or any(
        marker in info.filename for marker in KEEP_MARKERS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    args.output.mkdir(parents=True, exist_ok=True)

    records = []
    total_bytes = 0
    with ZipFile(args.archive) as archive:
        for info in archive.infolist():
            if not should_extract(info):
                continue
            relative = safe_path(info.filename)
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as target:
                while True:
                    chunk = source.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
            total_bytes += info.file_size
            records.append(
                {
                    "member": info.filename,
                    "path": str(destination),
                    "size": info.file_size,
                }
            )

    manifest = {
        "dataset": "IMPACT",
        "release": "v1.1",
        "source_archive": str(args.archive),
        "selection": "quick-start annotations and five-view RGB only",
        "files": len(records),
        "bytes": total_bytes,
        "records": records,
    }
    manifest_path = args.output / "inspect_preparation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "files": len(records),
                "gib": round(total_bytes / (1024**3), 3),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
