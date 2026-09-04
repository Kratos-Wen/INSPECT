"""Offline text-to-speech helpers for spoken assistant answers."""

from __future__ import annotations

import queue
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class _SpeechState:
    status: str = "idle"
    last_text: str = ""
    last_error: str = ""


class _BaseSpeaker:
    def speak(self, text: str) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class _PowerShellSpeaker(_BaseSpeaker):
    def __init__(self, *, rate: int, volume: float, voice_name: str) -> None:
        self.rate = max(-10, min(10, int(rate)))
        self.volume = max(0, min(100, int(round(float(volume) * 100.0))))
        self.voice_name = str(voice_name or "").strip()
        self._process: Optional[subprocess.Popen[str]] = None
        self._lock = threading.Lock()

    def speak(self, text: str) -> None:
        process = subprocess.Popen(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                self._build_script(text),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        with self._lock:
            self._process = process
        try:
            process.wait()
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None

    def stop(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
        if process is None:
            return
        try:
            process.terminate()
            process.wait(timeout=0.5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def _build_script(self, text: str) -> str:
        quoted_text = self._ps_quote(text)
        select_voice = ""
        if self.voice_name:
            select_voice = f"$s.SelectVoice({self._ps_quote(self.voice_name)});"
        return (
            "Add-Type -AssemblyName System.Speech;"
            "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            f"$s.Rate={self.rate};"
            f"$s.Volume={self.volume};"
            f"{select_voice}"
            f"$s.Speak({quoted_text});"
        )

    @staticmethod
    def _ps_quote(value: str) -> str:
        return "'" + str(value).replace("'", "''") + "'"


class _KokoroSpeaker(_BaseSpeaker):
    def __init__(
        self,
        *,
        model_path: str,
        voices_path: str,
        voice_name: str,
        language: str,
        speed: float,
        worker_python_path: str,
    ) -> None:
        self.model_path, self.voices_path = self._resolve_model_assets(model_path, voices_path)
        self.voice_name = str(voice_name or "af_sarah").strip() or "af_sarah"
        self.language = str(language or "en-us").strip() or "en-us"
        self.speed = max(0.5, min(2.0, float(speed or 1.0)))
        self._winsound = __import__("winsound")
        self.worker_python_path = self._resolve_worker_python(worker_python_path)
        self._process: Optional[subprocess.Popen[str]] = None
        self._lock = threading.Lock()

    def speak(self, text: str) -> None:
        normalized = " ".join(str(text or "").split()).strip()
        if not normalized:
            return
        with tempfile.NamedTemporaryFile(prefix="kokoro_", suffix=".wav", delete=False) as handle:
            output_path = Path(handle.name)
        process = subprocess.Popen(
            [
                self.worker_python_path,
                str(self._resolve_worker_script()),
                "--model-path",
                self.model_path,
                "--voices-path",
                self.voices_path,
                "--voice",
                self.voice_name,
                "--lang",
                self.language,
                "--speed",
                str(self.speed),
                "--text",
                normalized,
                "--output",
                str(output_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        with self._lock:
            self._process = process
        try:
            _, stderr = process.communicate()
            if process.returncode:
                raise RuntimeError((stderr or "").strip() or f"Kokoro worker failed with code {process.returncode}.")
            # `SND_SYNC` is not available on every Python build of winsound.
            # Synchronous playback is the default, so only require filename mode.
            self._winsound.PlaySound(
                str(output_path),
                self._winsound.SND_FILENAME,
            )
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None
            try:
                output_path.unlink(missing_ok=True)
            except Exception:
                pass

    def stop(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=0.5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        try:
            self._winsound.PlaySound(None, 0)
        except Exception:
            pass

    @staticmethod
    def _resolve_worker_python(worker_python_path: str) -> str:
        repo_root = Path(__file__).resolve().parents[1]
        candidate = str(worker_python_path or "").strip()
        if candidate:
            resolved = Path(candidate)
            if not resolved.is_absolute():
                resolved = repo_root / resolved
        else:
            resolved = Path(sys.executable)
        if not resolved.exists():
            raise RuntimeError(
                "Kokoro backend requires a Python interpreter with kokoro_onnx. "
                f"Missing interpreter: {resolved}"
            )
        return str(resolved)

    @staticmethod
    def _resolve_worker_script() -> Path:
        repo_root = Path(__file__).resolve().parents[1]
        candidates = [
            repo_root / "tools" / "kokoro_worker.py",
            repo_root.parent / "tools" / "kokoro_worker.py",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise RuntimeError("Kokoro worker script not found under repo tools/ or parent tools/.")

    @staticmethod
    def _resolve_model_assets(model_path: str, voices_path: str) -> tuple[str, str]:
        repo_root = Path(__file__).resolve().parents[1]
        raw_model_path = str(model_path or "").strip()
        if not raw_model_path:
            raise RuntimeError("Kokoro backend requires speech.kokoro_model_path.")
        model_candidate = Path(raw_model_path)
        if not model_candidate.is_absolute():
            model_candidate = repo_root / model_candidate
        if model_candidate.is_dir():
            onnx_models = sorted(model_candidate.rglob("*.onnx"))
            if not onnx_models:
                raise RuntimeError(f"No ONNX model found under {model_candidate}.")
            model_candidate = onnx_models[0]
        if not model_candidate.exists():
            raise RuntimeError(f"Kokoro model not found: {model_candidate}")

        raw_voices_path = str(voices_path or "").strip()
        voices_candidate = Path(raw_voices_path) if raw_voices_path else None
        if voices_candidate is not None and not voices_candidate.is_absolute():
            voices_candidate = repo_root / voices_candidate
        if voices_candidate and not voices_candidate.exists():
            raise RuntimeError(f"Kokoro voices file not found: {voices_candidate}")
        if voices_candidate is None:
            search_dir = model_candidate.parent.parent if model_candidate.parent.name.lower() == "onnx" else model_candidate.parent
            preferred = search_dir / "voices" / "af_sarah.bin"
            if preferred.exists():
                voices_candidate = preferred
            else:
                patterns = ("voices/*.bin", "*.bin", "voices*.bin", "voices*.json", "voices*.onnx")
                found = []
                for pattern in patterns:
                    found.extend(sorted(search_dir.glob(pattern)))
                voices_candidate = found[0] if found else None
        if voices_candidate is None:
            raise RuntimeError(f"No Kokoro voice file found under {model_candidate.parent}.")
        return str(model_candidate), str(voices_candidate or "")


class AssistantSpeechService:
    """Background TTS service that reads assistant answers aloud."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        backend: str = "auto",
        rate: int = 1,
        volume: float = 1.0,
        voice_name: str = "",
        kokoro_model_path: str = "",
        kokoro_voices_path: str = "",
        kokoro_voice: str = "af_sarah",
        kokoro_language: str = "en-us",
        kokoro_speed: float = 1.0,
        kokoro_python_path: str = "",
        dedupe_window_sec: float = 1.5,
        drop_pending_on_new: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.backend = str(backend or "auto").strip().lower()
        self.rate = int(rate)
        self.volume = float(volume)
        self.voice_name = str(voice_name or "").strip()
        self.kokoro_model_path = str(kokoro_model_path or "").strip()
        self.kokoro_voices_path = str(kokoro_voices_path or "").strip()
        self.kokoro_voice = str(kokoro_voice or "af_sarah").strip() or "af_sarah"
        self.kokoro_language = str(kokoro_language or "en-us").strip() or "en-us"
        self.kokoro_speed = float(kokoro_speed or 1.0)
        self.kokoro_python_path = str(kokoro_python_path or "").strip()
        self.dedupe_window_sec = max(0.0, float(dedupe_window_sec))
        self.drop_pending_on_new = bool(drop_pending_on_new)

        self._speaker: Optional[_BaseSpeaker] = None
        self._backend_name = ""
        self._state = _SpeechState(status="disabled" if not self.enabled else "idle")
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._last_spoken_text = ""
        self._last_spoken_time = 0.0

    def start(self) -> None:
        if not self.enabled or self._running:
            return
        if not self._ensure_backend():
            return
        self._running = True
        self._state.status = "ready"
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)
        if self._speaker is not None:
            self._speaker.stop()
        if self._worker is not None:
            self._worker.join(timeout=1.0)
            self._worker = None

    def speak(self, text: str) -> bool:
        normalized = " ".join(str(text or "").split()).strip()
        if not normalized or not self.enabled:
            return False
        if not self._ensure_backend():
            return False
        if self._should_skip_duplicate(normalized):
            return False
        if self.drop_pending_on_new:
            self._drain_pending()
        self._queue.put(normalized)
        return True

    def interrupt(self) -> None:
        if self._speaker is not None:
            self._speaker.stop()
        if self.enabled and not self._state.last_error:
            self._state.status = "ready"

    def status_text(self) -> str:
        if not self.enabled:
            return "tts=off"
        suffix = f"[{self._backend_name or self.backend or 'none'}]"
        if self._state.last_error:
            return f"tts=unavailable{suffix}"
        if self._state.status == "speaking":
            return f"tts=speaking{suffix}"
        return f"tts=ready{suffix}"

    def last_error(self) -> str:
        return self._state.last_error

    def backend_name(self) -> str:
        return self._backend_name or self.backend

    def _worker_loop(self) -> None:
        while self._running:
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if text is None:
                continue
            try:
                self._state.status = "speaking"
                if self._speaker is not None:
                    self._speaker.speak(text)
                self._state.last_text = text
                self._last_spoken_text = text
                self._last_spoken_time = time.time()
            except Exception as exc:
                self._state.last_error = str(exc)
                self._state.status = "error"
            finally:
                if self._state.status != "error":
                    self._state.status = "ready"

    def _ensure_backend(self) -> bool:
        if self._speaker is not None:
            return True
        try:
            if self.backend in {"auto", "kokoro"}:
                if self.kokoro_model_path:
                    try:
                        self._speaker = _KokoroSpeaker(
                            model_path=self.kokoro_model_path,
                            voices_path=self.kokoro_voices_path,
                            voice_name=self.kokoro_voice,
                            language=self.kokoro_language,
                            speed=self.kokoro_speed,
                            worker_python_path=self.kokoro_python_path,
                        )
                        self._state.last_error = ""
                        self._backend_name = "kokoro"
                        return True
                    except Exception:
                        if self.backend == "kokoro":
                            raise
                elif self.backend == "kokoro":
                    raise RuntimeError("speech.kokoro_model_path is required for backend='kokoro'.")

            if self.backend in {"auto", "pyttsx3"}:
                try:
                    import pyttsx3  # type: ignore

                    engine = pyttsx3.init()
                    engine.setProperty("rate", int(175 + self.rate * 12))
                    engine.setProperty("volume", max(0.0, min(1.0, self.volume)))
                    if self.voice_name:
                        for voice in engine.getProperty("voices") or []:
                            if self.voice_name.lower() in str(getattr(voice, "name", "")).lower():
                                engine.setProperty("voice", voice.id)
                                break

                    class _Pyttsx3Speaker(_BaseSpeaker):
                        def __init__(self, engine_obj) -> None:
                            self.engine = engine_obj

                        def speak(self, text: str) -> None:
                            self.engine.say(text)
                            self.engine.runAndWait()

                        def stop(self) -> None:
                            try:
                                self.engine.stop()
                            except Exception:
                                pass

                    self._speaker = _Pyttsx3Speaker(engine)
                    self._state.last_error = ""
                    self._backend_name = "pyttsx3"
                    return True
                except Exception:
                    if self.backend == "pyttsx3":
                        raise

            if self.backend in {"auto", "powershell", "sapi"}:
                self._speaker = _PowerShellSpeaker(rate=self.rate, volume=self.volume, voice_name=self.voice_name)
                self._state.last_error = ""
                self._backend_name = "powershell"
                return True

            raise RuntimeError(f"Unsupported TTS backend: {self.backend}")
        except Exception as exc:
            self._state.last_error = str(exc)
            self._state.status = "unavailable"
            return False

    def _should_skip_duplicate(self, text: str) -> bool:
        if not self.dedupe_window_sec:
            return False
        if text != self._last_spoken_text:
            return False
        return (time.time() - self._last_spoken_time) <= self.dedupe_window_sec

    def _drain_pending(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
