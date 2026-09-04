from __future__ import annotations

from scripts.run_active_view_online_simulation import (
    resolution_supervision_video_names,
)


def test_resolution_supervision_excludes_unresolved_only_sessions(tmp_path) -> None:
    timeline = tmp_path / "timeline.csv"
    timeline.write_text(
        "video,outcome,skip\n"
        "supported.mp4,supported,\n"
        "contradicted.mp4,contradicted,\n"
        "unresolved.mp4,unresolved,\n"
        "skipped.mp4,supported,1\n",
        encoding="utf-8",
    )

    assert resolution_supervision_video_names(timeline) == {
        "supported.mp4",
        "contradicted.mp4",
    }
