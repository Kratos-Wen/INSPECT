from __future__ import annotations

from inspect_runtime.components.voice import VoiceCommandService


def test_voice_backend_normalizes_package_style_name() -> None:
    service = VoiceCommandService(["S1", "S2"], backend="faster-whisper")

    assert service.backend == "faster_whisper"


def test_voice_transcript_routes_feedback_and_open_question() -> None:
    service = VoiceCommandService(["S1", "S2"])

    accept = service._parse_actions("confirm", "test")
    correction = service._parse_actions("step two", "test")
    question = service._parse_actions("What should I do next?", "test")

    assert [(item.action, item.label) for item in accept] == [
        ("feedback_accept", None)
    ]
    assert [(item.action, item.label) for item in correction] == [
        ("feedback_correct", "S2")
    ]
    assert [(item.action, item.text) for item in question] == [
        ("transcript", "What should I do next?")
    ]
