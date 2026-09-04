from types import SimpleNamespace

from scripts.evaluate_vlm_assistant_baselines import (
    PROMPT_PROTOCOL_VERSION,
    SYSTEM_PROMPT,
    cache_key,
    evaluation_key,
    load_eligible_keys,
    prompt_sha256,
    row_key,
)


def test_prompt_fingerprint_changes_with_prompt_text() -> None:
    assert prompt_sha256("claim A") != prompt_sha256("claim B")


def test_response_cache_key_includes_prompt_fingerprint() -> None:
    record = SimpleNamespace(video="video.mp4", frame=7, claim_id="cover_seated")
    key = row_key("qwen_local", "checkpoint", "direct", record, "prompt")
    cached = cache_key(
        {
            "provider": "qwen_local",
            "model": "checkpoint",
            "protocol": "direct",
            "video": "video.mp4",
            "frame": 7,
            "claim_id": "cover_seated",
            "prompt_sha256": prompt_sha256("prompt"),
        }
    )
    assert key == cached


def test_triage_prompt_defines_all_three_decisions_symmetrically() -> None:
    assert PROMPT_PROTOCOL_VERSION == "triage_v2"
    for decision in ("supported", "contradicted", "unresolved"):
        assert decision in SYSTEM_PROMPT


def test_eligibility_keys_normalize_windows_paths(tmp_path) -> None:
    path = tmp_path / "eligible.csv"
    path.write_text(
        "video,frame,truth\nnested\\dataset\\video.mp4,12,supported\n",
        encoding="utf-8",
    )
    assert load_eligible_keys(path) == {evaluation_key("video.mp4", 12, "supported")}
