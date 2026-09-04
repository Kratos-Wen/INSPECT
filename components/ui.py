"""OpenCV runtime UI and keyboard controls for the live assistant."""

from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from ..core_types import RuntimeAction


class OpenCVRuntimeUI:
    """Render live views and convert key presses into runtime actions."""

    def __init__(
        self,
        enabled: bool = True,
        window_name: str = "INSPECT Trace Engine",
        focus_window_name: str = "INSPECT Focus",
        show_focus: bool = True,
        show_help: bool = True,
        key_quit: str = "q",
        key_pause: str = "p",
        key_voice: str = "v",
        key_mute: str = "m",
        key_feedback: str = "f",
        key_help: str = "h",
    ) -> None:
        self.enabled = bool(enabled)
        self.window_name = str(window_name)
        self.focus_window_name = str(focus_window_name)
        self.show_focus = bool(show_focus)
        self.show_help = bool(show_help)
        self._opened = False
        self._keymap = {
            ord(str(key_quit).lower()[:1]): RuntimeAction("quit", source="keyboard"),
            ord(str(key_pause).lower()[:1]): RuntimeAction("toggle_pause", source="keyboard"),
            ord(" "): RuntimeAction("toggle_pause", source="keyboard"),
            ord(str(key_voice).lower()[:1]): RuntimeAction("voice_capture", source="keyboard"),
            ord(str(key_mute).lower()[:1]): RuntimeAction("toggle_voice_mute", source="keyboard"),
            ord(str(key_feedback).lower()[:1]): RuntimeAction("force_feedback", source="keyboard"),
            ord(str(key_help).lower()[:1]): RuntimeAction("toggle_help", source="keyboard"),
        }

    def open(self) -> None:
        """Create windows lazily."""

        if not self.enabled or self._opened:
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.show_focus:
            cv2.namedWindow(self.focus_window_name, cv2.WINDOW_NORMAL)
        self._opened = True

    def render(
        self,
        canvas: np.ndarray,
        status_lines: Optional[Iterable[str]] = None,
        focus_crop: Optional[np.ndarray] = None,
    ) -> None:
        """Render the main canvas and optional focus crop."""

        if not self.enabled:
            return
        self.open()
        frame = canvas.copy()
        lines = [str(line) for line in (status_lines or []) if str(line).strip()]
        y = 58
        for line in lines[:8]:
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            y += 20
        if self.show_help:
            cv2.putText(
                frame,
                "Q quit | P pause | V voice | M mute | F feedback | H help",
                (10, max(24, frame.shape[0] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (120, 255, 180),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow(self.window_name, frame)
        if self.show_focus and focus_crop is not None and focus_crop.size > 0:
            cv2.imshow(self.focus_window_name, focus_crop)

    def poll_action(self, delay_ms: int = 1) -> Optional[RuntimeAction]:
        """Poll one keyboard action from the UI window."""

        if not self.enabled:
            return None
        self.open()
        key = cv2.waitKey(int(delay_ms)) & 0xFF
        if key in {255, -1}:
            return None
        return self._keymap.get(key)

    def toggle_help(self) -> bool:
        """Flip the help overlay and return the new state."""

        self.show_help = not self.show_help
        return self.show_help

    def close(self) -> None:
        """Close all windows owned by this UI."""

        if not self.enabled:
            return
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
        if self.show_focus:
            try:
                cv2.destroyWindow(self.focus_window_name)
            except Exception:
                pass
        self._opened = False
