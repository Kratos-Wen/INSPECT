from pathlib import Path

from scripts.fingerprint_model_checkpoint import build_manifest


def test_checkpoint_manifest_is_deterministic_and_path_free(tmp_path: Path) -> None:
    first = tmp_path / "weights.bin"
    nested = tmp_path / "config" / "model.json"
    nested.parent.mkdir()
    ignored = tmp_path / ".cache" / "download.lock"
    ignored.parent.mkdir()
    first.write_bytes(b"weights")
    nested.write_text("{}", encoding="utf-8")
    ignored.write_text("transient", encoding="utf-8")

    left = build_manifest(tmp_path, "public/model")
    right = build_manifest(tmp_path, "public/model")

    assert left == right
    assert left["model_id"] == "public/model"
    assert {row["path"] for row in left["files"]} == {
        "config/model.json",
        "weights.bin",
    }
    assert str(tmp_path) not in str(left)
    assert all(not row["path"].startswith(".cache/") for row in left["files"])
