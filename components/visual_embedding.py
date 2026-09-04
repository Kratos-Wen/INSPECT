"""Shared visual encoders used by retrieval and memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..core_types import Detection

_HAS_CLIP = False
try:  # pragma: no cover - dependency-dependent path
    import open_clip
    import torch
    from PIL import Image

    _HAS_CLIP = True
except Exception:  # pragma: no cover - dependency-dependent path
    _HAS_CLIP = False


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector = vector / norm
    return vector.astype(np.float32)


def _rgb_mean_embedding(image_bgr: np.ndarray, size: int = 8) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)
    vector = image_rgb.astype(np.float32).reshape(-1) / 255.0
    vector = (vector - vector.mean()) / (vector.std() + 1e-6)
    return _normalize(vector)


def _hybrid_embedding(image_bgr: np.ndarray, grid: int = 4, bins: int = 8) -> np.ndarray:
    canvas = cv2.resize(image_bgr, (grid * 24, grid * 24), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    features = []
    cell_h = max(1, rgb.shape[0] // grid)
    cell_w = max(1, rgb.shape[1] // grid)
    for gy in range(grid):
        for gx in range(grid):
            patch = rgb[gy * cell_h : (gy + 1) * cell_h, gx * cell_w : (gx + 1) * cell_w]
            features.extend(np.mean(patch, axis=(0, 1)).tolist())

    hist_specs = (
        (0, [bins], [0, 180]),
        (1, [bins], [0, 256]),
        (2, [bins], [0, 256]),
    )
    for channel, hist_bins, hist_range in hist_specs:
        hist = cv2.calcHist([hsv], [channel], None, hist_bins, hist_range).flatten().astype(np.float32)
        hist = hist / (float(hist.sum()) + 1e-6)
        features.extend(hist.tolist())

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees=False)
    orientation_hist, _ = np.histogram(
        angle,
        bins=bins,
        range=(0.0, 2.0 * np.pi),
        weights=magnitude,
    )
    orientation_hist = orientation_hist.astype(np.float32)
    orientation_hist = orientation_hist / (float(orientation_hist.sum()) + 1e-6)
    features.extend(orientation_hist.tolist())

    quad_h = max(1, gray.shape[0] // 2)
    quad_w = max(1, gray.shape[1] // 2)
    for gy in range(2):
        for gx in range(2):
            patch_mag = magnitude[gy * quad_h : (gy + 1) * quad_h, gx * quad_w : (gx + 1) * quad_w]
            features.append(float(np.mean(patch_mag)) / 255.0)

    return _normalize(np.array(features, dtype=np.float32))


def _crop_detection(image_bgr: np.ndarray, detection: Detection, pad_ratio: float = 0.1) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = detection.xyxy
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    pad_x = int(round(box_w * pad_ratio))
    pad_y = int(round(box_h * pad_ratio))
    left = max(0, int(round(x1)) - pad_x)
    top = max(0, int(round(y1)) - pad_y)
    right = min(width, int(round(x2)) + pad_x)
    bottom = min(height, int(round(y2)) + pad_y)
    if left >= right or top >= bottom:
        return image_bgr
    return image_bgr[top:bottom, left:right]


@dataclass
class _ClipEncoder:  # pragma: no cover - dependency-dependent path
    device: Optional[str] = None

    def __post_init__(self) -> None:
        if not _HAS_CLIP:
            raise RuntimeError("CLIP embedding requires open_clip_torch and Pillow.")
        resolved_device = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_obj = torch.device(resolved_device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            device=self.device_obj,
        )
        self.model.eval()

    def encode(self, image_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        batch = self.preprocess(pil_image).unsqueeze(0).to(self.device_obj)
        with torch.no_grad():
            features = self.model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).float().cpu().numpy().astype(np.float32)


class SharedVisualEncoder:
    """Shared visual embedding backend for retrieval and memory."""

    def __init__(self, mode: str = "hybrid-4", device: Optional[str] = None) -> None:
        self.mode = str(mode).strip().lower()
        self.device = device
        self.clip_encoder: Optional[_ClipEncoder] = None
        if self.mode == "clip":
            self.clip_encoder = _ClipEncoder(device=device)

    def encode_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """Encode a full image into the shared visual space."""

        if self.clip_encoder is not None:
            return _normalize(self.clip_encoder.encode(image_bgr))
        if self.mode.startswith("rgb-mean-"):
            suffix = self.mode.replace("rgb-mean-", "")
            size = int(suffix) if suffix.isdigit() else 8
            return _rgb_mean_embedding(image_bgr, size=size)
        if self.mode.startswith("hybrid-"):
            suffix = self.mode.replace("hybrid-", "")
            grid = int(suffix) if suffix.isdigit() else 4
            return _hybrid_embedding(image_bgr, grid=grid)
        return _hybrid_embedding(image_bgr, grid=4)

    def encode_query(
        self,
        frame_bgr: np.ndarray,
        focus_detection: Optional[Detection] = None,
    ) -> np.ndarray:
        """Encode a query frame, optionally blending a focused crop."""

        full = self.encode_image(frame_bgr)
        if focus_detection is None:
            return full
        crop = _crop_detection(frame_bgr, focus_detection)
        crop_vector = self.encode_image(crop)
        return _normalize(0.6 * full + 0.4 * crop_vector)
