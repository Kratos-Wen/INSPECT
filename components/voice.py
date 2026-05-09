"""Optional voice command service for the live assistant."""

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..types import RuntimeAction

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - dependency is optional
    sd = None


@dataclass
class _VoiceState:
    status: str = "idle"
    last_text: str = ""
    last_error: str = ""


class _BaseTranscriber:
    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str) -> str:
        raise NotImplementedError


_FASTER_WHISPER_PROBE_CACHE: dict[tuple[str, str, str, int], tuple[bool, str]] = {}


def _default_compute_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _default_compute_type(device: str, requested: str) -> str:
    normalized = str(requested or "auto").strip().lower()
    if normalized and normalized != "auto":
        return normalized
    return "float16" if str(device).lower() == "cuda" else "int8"


def _resolve_transformers_model_name(model_name: str) -> str:
    normalized = str(model_name or "").strip()
    aliases = {
        "distil-small.en": "distil-whisper/distil-small.en",
        "distil-medium.en": "distil-whisper/distil-medium.en",
        "distil-large-v3": "distil-whisper/distil-large-v3",
    }
    return aliases.get(normalized, normalized or "distil-whisper/distil-small.en")


def _probe_faster_whisper(
    model_name: str,
    cache_dir: Optional[str],
    compute_type: str,
    cpu_threads: int,
) -> tuple[bool, str]:
    cache_key = (
        str(model_name).strip(),
        str(cache_dir or "").strip(),
        str(compute_type or "auto").strip().lower(),
        int(cpu_threads),
    )
    if cache_key in _FASTER_WHISPER_PROBE_CACHE:
        return _FASTER_WHISPER_PROBE_CACHE[cache_key]
    code = (
        "import os\n"
        "try:\n"
        " import torch\n"
        " device='cuda' if torch.cuda.is_available() else 'cpu'\n"
        "except Exception:\n"
        " device='cpu'\n"
        f"compute_type={compute_type!r}\n"
        "compute = compute_type if compute_type and compute_type.lower() != 'auto' else ('float16' if device=='cuda' else 'int8')\n"
        "from faster_whisper import WhisperModel\n"
        f"kwargs={{'device': device, 'compute_type': compute}}\n"
        + (f"kwargs['download_root']={str(cache_dir)!r}\n" if cache_dir else "")
        + (f"kwargs['cpu_threads']={int(cpu_threads)}\n" if int(cpu_threads) > 0 else "")
        + f"WhisperModel({str(model_name)!r}, **kwargs)\n"
        "print('ok', flush=True)\n"
    )
    env = dict(os.environ)
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    completed = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    if completed.returncode == 0:
        result = (True, "")
        _FASTER_WHISPER_PROBE_CACHE[cache_key] = result
        return result
    detail = (completed.stderr or completed.stdout or "").strip()
    result = (False, detail[:400])
    _FASTER_WHISPER_PROBE_CACHE[cache_key] = result
    return result


class _FasterWhisperTranscriber(_BaseTranscriber):
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        compute_type: str = "auto",
        cpu_threads: int = 0,
    ) -> None:
        from faster_whisper import WhisperModel

        device = _default_compute_device()
        kwargs = {
            "device": device,
            "compute_type": _default_compute_type(device, compute_type),
        }
        if cache_dir:
            kwargs["download_root"] = str(cache_dir)
        if int(cpu_threads) > 0:
            kwargs["cpu_threads"] = int(cpu_threads)
        self.model = WhisperModel(str(model_name).strip() or "distil-small.en", **kwargs)

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str) -> str:
        segments, _ = self.model.transcribe(
            audio.astype("float32"),
            language=str(language or "en"),
            vad_filter=True,
            condition_on_previous_text=False,
            without_timestamps=True,
            beam_size=1,
        )
        return " ".join(str(segment.text).strip() for segment in segments if str(segment.text).strip()).strip()


class _TransformersTranscriber(_BaseTranscriber):
    def __init__(self, model_name: str, cache_dir: Optional[str] = None) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        resolved = _resolve_transformers_model_name(model_name)
        model_kwargs = {}
        if cache_dir:
            model_kwargs["cache_dir"] = str(cache_dir)
            repo_dir = Path(cache_dir) / f"models--{resolved.replace('/', '--')}"
            if repo_dir.exists():
                model_kwargs["local_files_only"] = True
        model = AutoModelForSpeechSeq2Seq.from_pretrained(resolved, **model_kwargs)
        processor = AutoProcessor.from_pretrained(resolved, **model_kwargs)
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=-1,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str) -> str:
        result = self.pipeline({"raw": audio.astype("float32"), "sampling_rate": int(sample_rate)})
        return str(result.get("text", "")).strip()


class VoiceCommandService:
    """Capture audio, transcribe it, and emit runtime actions."""

    def __init__(
        self,
        steps: Iterable[str],
        enabled: bool = True,
        mode: str = "manual",
        backend: str = "auto",
        model_name: str = "distil-small.en",
        cache_dir: str = "",
        language: str = "en",
        wake_words: Iterable[str] = ("mica", "assistant"),
        require_wake_word_in_always_on: bool = True,
        command_max_tokens: int = 6,
        compute_type: str = "auto",
        cpu_threads: int = 0,
        sample_rate: int = 16000,
        manual_duration_sec: float = 4.0,
        chunk_duration_sec: float = 2.5,
        cooldown_sec: float = 1.2,
        min_rms: float = 0.008,
        device: Optional[int] = None,
        mute_by_default: bool = False,
    ) -> None:
        self.steps = [str(step).strip().upper() for step in steps]
        self.enabled = bool(enabled)
        self.mode = str(mode).strip().lower()
        self.backend = str(backend).strip().lower()
        self.model_name = str(model_name).strip()
        self.cache_dir = str(cache_dir).strip()
        self.language = str(language).strip() or "en"
        self.wake_words = [str(item).strip().lower() for item in wake_words if str(item).strip()]
        self.require_wake_word_in_always_on = bool(require_wake_word_in_always_on)
        self.command_max_tokens = max(1, int(command_max_tokens))
        self.compute_type = str(compute_type).strip() or "auto"
        self.cpu_threads = max(0, int(cpu_threads))
        self.sample_rate = int(sample_rate)
        self.manual_duration_sec = float(manual_duration_sec)
        self.chunk_duration_sec = float(chunk_duration_sec)
        self.cooldown_sec = float(cooldown_sec)
        self.min_rms = float(min_rms)
        self.device = device
        self.muted = bool(mute_by_default)
        self._transcriber: Optional[_BaseTranscriber] = None
        self._backend_name: str = ""
        self._transcriber_error: str = ""
        self._running = False
        self._busy = False
        self._worker: Optional[threading.Thread] = None
        self._capture_lock = threading.Lock()
        self._queue: "queue.Queue[RuntimeAction]" = queue.Queue()
        self._state = _VoiceState(status="disabled" if not self.enabled else "idle")

    def start(self) -> None:
        """Start continuous listening when configured."""

        if not self.enabled or self.mode != "always_on":
            return
        if not self._ensure_backend():
            return
        if self._running:
            return
        self._running = True
        self._worker = threading.Thread(target=self._listen_loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        """Stop background listening."""

        self._running = False
        if self._worker is not None:
            self._worker.join(timeout=0.5)
            self._worker = None

    def toggle_mute(self) -> bool:
        """Toggle mute state and return the new value."""

        self.muted = not self.muted
        self._state.status = "muted" if self.muted else "listening"
        return self.muted

    def set_muted(self, value: bool) -> bool:
        """Set mute state explicitly and return it."""

        self.muted = bool(value)
        self._state.status = "muted" if self.muted else "listening"
        return self.muted

    def trigger_manual_capture(self) -> bool:
        """Start one asynchronous manual capture."""

        if not self.enabled:
            self._state.status = "disabled"
            return False
        if not self._ensure_backend():
            return False
        if self._busy:
            return False
        thread = threading.Thread(target=self._capture_once, args=(self.manual_duration_sec, "voice_manual"), daemon=True)
        thread.start()
        return True

    def poll_actions(self) -> list[RuntimeAction]:
        """Drain all queued runtime actions."""

        items: list[RuntimeAction] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return items

    def status_text(self) -> str:
        """Return one concise status line for the UI."""

        if not self.enabled:
            return "voice=off"
        backend_tag = self._backend_name or self.backend or "none"
        suffix = f" [{self.mode}:{backend_tag}]"
        if self._transcriber_error:
            return f"voice=unavailable{suffix}"
        if self.muted:
            return f"voice=muted{suffix}"
        if self._busy:
            return f"voice=capturing{suffix}"
        if self._running:
            return f"voice=listening{suffix}"
        return f"voice=ready{suffix}"

    def last_transcript(self) -> str:
        """Return the latest non-empty transcript."""

        return self._state.last_text

    def last_error(self) -> str:
        """Return the latest backend error, if any."""

        return self._state.last_error or self._transcriber_error

    def backend_name(self) -> str:
        """Return the active backend name after initialization."""

        return self._backend_name or self.backend

    def _listen_loop(self) -> None:
        while self._running:
            if self.muted:
                time.sleep(0.15)
                continue
            self._capture_once(self.chunk_duration_sec, "voice_auto")
            time.sleep(max(0.05, self.cooldown_sec))

    def _capture_once(self, duration_sec: float, source: str) -> None:
        if not self._ensure_backend():
            return
        with self._capture_lock:
            self._busy = True
            try:
                self._state.status = "capturing"
                audio = self._record_audio(duration_sec)
                if audio is None:
                    return
                rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
                if rms < self.min_rms:
                    self._state.status = "listening" if self._running and not self.muted else "ready"
                    return
                text = self._transcriber.transcribe(audio, self.sample_rate, self.language) if self._transcriber else ""
                text = str(text).strip()
                if not text:
                    self._state.status = "listening" if self._running and not self.muted else "ready"
                    return
                self._state.last_text = text
                for action in self._parse_actions(text, source):
                    self._queue.put(action)
            except Exception as exc:
                self._state.last_error = str(exc)
                self._state.status = "error"
            finally:
                self._busy = False
                if self._state.status != "error":
                    self._state.status = "muted" if self.muted else ("listening" if self._running else "ready")

    def _record_audio(self, duration_sec: float) -> Optional[np.ndarray]:
        if sd is None:
            self._transcriber_error = "sounddevice not installed"
            self._state.last_error = self._transcriber_error
            return None
        frames = max(1, int(float(duration_sec) * self.sample_rate))
        audio = sd.rec(
            frames,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
        )
        sd.wait()
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def _ensure_backend(self) -> bool:
        if self._transcriber is not None:
            return True
        if self._transcriber_error:
            return False
        try:
            if self.backend in {"faster_whisper", "auto"}:
                try:
                    probe_ok, probe_error = _probe_faster_whisper(
                        self.model_name,
                        self.cache_dir or None,
                        self.compute_type,
                        self.cpu_threads,
                    )
                    if not probe_ok:
                        raise RuntimeError(probe_error or "faster-whisper probe failed")
                    self._transcriber = _FasterWhisperTranscriber(
                        self.model_name,
                        self.cache_dir or None,
                        compute_type=self.compute_type,
                        cpu_threads=self.cpu_threads,
                    )
                    self._backend_name = "faster_whisper"
                    return True
                except Exception:
                    if self.backend == "faster_whisper":
                        raise
            if self.backend in {"transformers", "auto"}:
                self._transcriber = _TransformersTranscriber(self.model_name, self.cache_dir or None)
                self._backend_name = "transformers"
                return True
            raise RuntimeError(f"Unsupported voice backend: {self.backend}")
        except Exception as exc:
            self._transcriber_error = str(exc)
            self._state.last_error = str(exc)
            self._state.status = "unavailable"
            return False

    def _parse_actions(self, text: str, source: str) -> list[RuntimeAction]:
        normalized = re.sub(r"\s+", " ", str(text).strip().lower())
        actions: list[RuntimeAction] = []
        if not normalized:
            return actions

        command_text, has_wake = self._strip_wake_prefix(normalized)
        always_on_auto = self.mode == "always_on" and source == "voice_auto"
        if always_on_auto and self.require_wake_word_in_always_on and not has_wake:
            return actions
        normalized = command_text or normalized
        if not normalized:
            return actions

        if self._matches_phrase(normalized, ("quit", "exit", "close")):
            return [RuntimeAction("quit", source=source, text=text)]
        if self._matches_phrase(normalized, ("resume", "continue")):
            return [RuntimeAction("resume", source=source, text=text)]
        if self._matches_phrase(normalized, ("pause", "hold")):
            return [RuntimeAction("pause", source=source, text=text)]
        if self._matches_phrase(normalized, ("unmute", "listen")):
            return [RuntimeAction("unmute_voice", source=source, text=text)]
        if self._matches_phrase(normalized, ("mute", "silence")):
            return [RuntimeAction("mute_voice", source=source, text=text)]
        if self._matches_phrase(normalized, ("skip", "ignore")):
            return [RuntimeAction("feedback_skip", source=source, text=text)]
        if self._matches_phrase(
            normalized,
            ("accept", "confirm", "confirmed", "yes", "correct", "that's right", "that is right"),
        ):
            return [RuntimeAction("feedback_accept", source=source, text=text)]

        step_label = self._extract_step_label(normalized)
        if step_label is not None:
            return [RuntimeAction("feedback_correct", source=source, label=step_label, text=text)]

        actions.append(RuntimeAction("transcript", source=source, text=text))
        return actions

    def _strip_wake_prefix(self, text: str) -> tuple[str, bool]:
        candidate = str(text).strip().lower()
        if not candidate:
            return "", False
        prefixes: list[str] = []
        for wake in self.wake_words:
            prefixes.extend(
                [
                    wake,
                    f"hey {wake}",
                    f"hi {wake}",
                    f"okay {wake}",
                    f"ok {wake}",
                ]
            )
        for prefix in prefixes:
            if candidate == prefix:
                return "", True
            if candidate.startswith(prefix + " "):
                return candidate[len(prefix) + 1 :].strip(), True
        return candidate, False

    def _matches_phrase(self, text: str, phrases: Iterable[str]) -> bool:
        normalized = str(text).strip().lower()
        if not normalized:
            return False
        if len(normalized.split()) > self.command_max_tokens:
            return False
        return any(normalized == phrase or normalized.startswith(phrase + " ") for phrase in phrases)

    def _extract_step_label(self, text: str) -> Optional[str]:
        if len(str(text).split()) > self.command_max_tokens:
            return None
        direct = re.search(r"\bs\s*([0-9]+)\b", text)
        if direct:
            candidate = f"S{direct.group(1)}"
            return candidate if candidate in self.steps else None

        word_map = {
            "one": "S1",
            "two": "S2",
            "three": "S3",
            "four": "S4",
            "five": "S5",
            "six": "S6",
            "seven": "S7",
            "eight": "S8",
            "nine": "S9",
        }
        for word, label in word_map.items():
            if label in self.steps and re.search(rf"\b(step\s+)?{word}\b", text):
                return label
        for step_id in self.steps:
            if step_id.lower() == text.replace(" ", ""):
                return step_id
        return None
