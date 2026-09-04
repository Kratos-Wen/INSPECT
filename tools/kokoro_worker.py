"""Standalone Kokoro worker used by the main runtime TTS service."""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize one utterance with Kokoro ONNX.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--voices-path", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--lang", default="en-us")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from kokoro_onnx import Kokoro

    kokoro = Kokoro(model_path=args.model_path, voices_path=args.voices_path)
    audio, sample_rate = kokoro.create(
        args.text,
        voice=args.voice,
        speed=max(0.5, min(2.0, float(args.speed))),
        lang=args.lang,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_wav(output_path, audio, sample_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
