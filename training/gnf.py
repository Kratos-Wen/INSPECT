"""Procedural Generative Nuisance Fields for TwinSwap.

GNF backgrounds are class-agnostic visual interference fields rather than
scene-specific images. They combine texture, structure, and photometric
perturbations to make detection harder without introducing object co-occurrence
shortcuts.
"""

from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass
class GNFConfig:
    width: int = 640
    height: int = 640
    texture_layers: Tuple[int, ...] = (4, 8, 16, 32, 64)
    min_structures: int = 2
    max_structures: int = 6
    min_photometric: int = 2
    max_photometric: int = 5
    structure_alpha: Tuple[float, float] = (0.10, 0.38)
    texture_contrast: Tuple[float, float] = (0.55, 1.35)
    jpeg_quality: Tuple[int, int] = (72, 96)


@dataclass
class GNFField:
    image: Image.Image
    metadata: Dict[str, Any]


class GNFGenerator:
    """Generate class-agnostic nuisance fields for object insertion."""

    texture_styles = (
        "brushed_metal",
        "plastic_grain",
        "rubber_speckle",
        "paper_fiber",
        "fabric_weave",
        "matte_composite",
        "low_frequency_blotches",
        "high_frequency_clutter",
    )
    structure_styles = (
        "stripes",
        "grid",
        "rings",
        "radial_edges",
        "holes",
        "diagonal_repeats",
        "tooth_edges",
        "fragments",
        "micro_scratches",
    )
    photometric_styles = (
        "shadow_blob",
        "specular_patch",
        "vignette",
        "contrast_drop",
        "color_shift",
        "sensor_noise",
        "defocus",
    )

    def __init__(self, config: GNFConfig | None = None) -> None:
        self.config = config or GNFConfig()

    def generate(
        self,
        seed: int,
        width: int | None = None,
        height: int | None = None,
        profile: Dict[str, Any] | None = None,
    ) -> GNFField:
        width = int(width or self.config.width)
        height = int(height or self.config.height)
        rng = random.Random(int(seed))
        np_rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        profile = dict(profile or {})

        texture_style = str(profile.get("texture_style") or rng.choice(self.texture_styles))
        if texture_style not in self.texture_styles:
            texture_style = rng.choice(self.texture_styles)
        image = self._texture_field(texture_style, width, height, rng, np_rng)
        structure_count = rng.randint(self.config.min_structures, self.config.max_structures)
        structures = self._sample_profiled_styles(
            candidates=self.structure_styles,
            profile_values=profile.get("structures"),
            count=structure_count,
            rng=rng,
        )
        for style in structures:
            image = self._apply_structure(image, style, rng)

        photometric_count = rng.randint(self.config.min_photometric, self.config.max_photometric)
        photometric = self._sample_profiled_styles(
            candidates=self.photometric_styles,
            profile_values=profile.get("photometric"),
            count=photometric_count,
            rng=rng,
        )
        for style in photometric:
            image = self._apply_photometric(image, style, rng, np_rng)

        quality = rng.randint(*self.config.jpeg_quality)
        image = self._jpeg_roundtrip(image, quality)
        return GNFField(
            image=image.convert("RGB"),
            metadata={
                "type": "generative_nuisance_field",
                "seed": int(seed),
                "width": width,
                "height": height,
                "texture_style": texture_style,
                "structures": structures,
                "photometric": photometric,
                "jpeg_quality": quality,
                "profiled": bool(profile),
            },
        )

    @staticmethod
    def _sample_profiled_styles(
        candidates: Tuple[str, ...],
        profile_values: Any,
        count: int,
        rng: random.Random,
    ) -> List[str]:
        hard_styles = [str(value) for value in (profile_values or []) if str(value) in candidates]
        if not hard_styles:
            return [rng.choice(candidates) for _ in range(count)]
        styles: List[str] = []
        for index in range(count):
            if index < len(hard_styles) or rng.random() < 0.70:
                styles.append(rng.choice(hard_styles))
            else:
                styles.append(rng.choice(candidates))
        return styles

    def _texture_field(self, style: str, width: int, height: int, rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
        base = np.zeros((height, width, 3), dtype=np.float32)
        palette = self._palette(style, rng)
        for layer in self.config.texture_layers:
            grid_h = max(2, int(math.ceil(height / layer)))
            grid_w = max(2, int(math.ceil(width / layer)))
            noise = np_rng.random((grid_h, grid_w, 1), dtype=np.float32)
            noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)[..., None]
            color = np.asarray(rng.choice(palette), dtype=np.float32).reshape(1, 1, 3)
            weight = rng.uniform(0.08, 0.28)
            base += noise * color * weight

        base = base / max(1e-6, float(base.max())) * rng.uniform(150.0, 235.0)
        contrast = rng.uniform(*self.config.texture_contrast)
        base = (base - 127.5) * contrast + 127.5

        if style == "brushed_metal":
            streaks = np_rng.normal(0, 18, (height, max(1, width // 12), 1)).astype(np.float32)
            streaks = cv2.resize(streaks, (width, height), interpolation=cv2.INTER_LINEAR)
            base += streaks[..., None] if streaks.ndim == 2 else streaks
        elif style == "fabric_weave":
            yy, xx = np.mgrid[0:height, 0:width]
            weave = (np.sin(xx / rng.uniform(3.0, 8.0)) + np.sin(yy / rng.uniform(3.0, 8.0))) * 14
            base += weave[..., None]
        elif style == "paper_fiber":
            fibers = np_rng.normal(0, 11, (height, width, 1)).astype(np.float32)
            fibers = cv2.GaussianBlur(fibers, (0, 0), rng.uniform(0.4, 1.2))
            base += fibers[..., None] if fibers.ndim == 2 else fibers
        elif style == "high_frequency_clutter":
            base += np_rng.normal(0, 24, (height, width, 3)).astype(np.float32)

        return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")

    @staticmethod
    def _palette(style: str, rng: random.Random) -> List[Tuple[int, int, int]]:
        neutral = [
            (168, 176, 174),
            (118, 124, 128),
            (205, 200, 188),
            (92, 98, 96),
            (178, 164, 139),
            (132, 145, 155),
            (160, 154, 162),
            (108, 118, 104),
        ]
        if style == "brushed_metal":
            return [(160, 166, 166), (105, 112, 116), (210, 213, 208), (78, 86, 88)]
        if style == "rubber_speckle":
            return [(45, 48, 48), (74, 78, 75), (110, 104, 96), (136, 133, 125)]
        if style == "paper_fiber":
            return [(188, 184, 172), (216, 212, 202), (145, 151, 146), (172, 166, 155)]
        rng.shuffle(neutral)
        return neutral[:4]

    def _apply_structure(self, image: Image.Image, style: str, rng: random.Random) -> Image.Image:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = image.size
        alpha = int(255 * rng.uniform(*self.config.structure_alpha))
        color = self._structure_color(rng, alpha)

        if style == "stripes":
            spacing = rng.randint(16, 72)
            line_w = rng.randint(2, 9)
            angle = rng.choice([0, 18, 35, 55, 90, 125, 145])
            self._draw_parallel_lines(draw, width, height, spacing, line_w, angle, color)
        elif style == "grid":
            spacing = rng.randint(22, 88)
            line_w = rng.randint(1, 5)
            for x in range(rng.randint(-spacing, 0), width + spacing, spacing):
                draw.line((x, 0, x + rng.randint(-20, 20), height), fill=color, width=line_w)
            for y in range(rng.randint(-spacing, 0), height + spacing, spacing):
                draw.line((0, y, width, y + rng.randint(-20, 20)), fill=color, width=line_w)
        elif style == "rings":
            for _ in range(rng.randint(4, 16)):
                r = rng.randint(12, max(18, min(width, height) // 5))
                x = rng.randint(-r, width)
                y = rng.randint(-r, height)
                draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=rng.randint(2, 8))
        elif style == "radial_edges":
            cx = rng.randint(0, width)
            cy = rng.randint(0, height)
            for k in range(rng.randint(18, 64)):
                theta = (2 * math.pi * k / rng.randint(20, 70)) + rng.random() * 0.1
                length = rng.randint(width // 6, max(width, height))
                x2 = cx + int(math.cos(theta) * length)
                y2 = cy + int(math.sin(theta) * length)
                draw.line((cx, cy, x2, y2), fill=color, width=rng.randint(1, 4))
        elif style == "holes":
            for _ in range(rng.randint(10, 48)):
                r = rng.randint(4, 24)
                x = rng.randint(0, width)
                y = rng.randint(0, height)
                draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        elif style == "diagonal_repeats":
            spacing = rng.randint(12, 42)
            line_w = rng.randint(2, 6)
            self._draw_parallel_lines(draw, width, height, spacing, line_w, rng.choice([35, 55, 125, 145]), color)
        elif style == "tooth_edges":
            for _ in range(rng.randint(3, 10)):
                y = rng.randint(0, height)
                step = rng.randint(10, 28)
                amp = rng.randint(8, 26)
                points = []
                for x in range(-step, width + step, step):
                    points.append((x, y + (amp if (x // step) % 2 else -amp)))
                draw.line(points, fill=color, width=rng.randint(2, 6), joint="curve")
        elif style == "fragments":
            for _ in range(rng.randint(8, 28)):
                cx = rng.randint(0, width)
                cy = rng.randint(0, height)
                radius = rng.randint(12, 80)
                pts = []
                for k in range(rng.randint(3, 7)):
                    theta = 2 * math.pi * k / 6 + rng.uniform(-0.4, 0.4)
                    rr = radius * rng.uniform(0.4, 1.0)
                    pts.append((cx + int(math.cos(theta) * rr), cy + int(math.sin(theta) * rr)))
                draw.polygon(pts, outline=color)
        elif style == "micro_scratches":
            for _ in range(rng.randint(80, 240)):
                x = rng.randint(0, width)
                y = rng.randint(0, height)
                length = rng.randint(8, 80)
                theta = rng.uniform(0, math.pi)
                draw.line((x, y, x + int(math.cos(theta) * length), y + int(math.sin(theta) * length)), fill=color, width=1)

        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 1.2)))
        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    @staticmethod
    def _structure_color(rng: random.Random, alpha: int) -> Tuple[int, int, int, int]:
        base = rng.randint(38, 220)
        tint = rng.choice([(0, 0, 0), (18, -10, -6), (-10, 12, 18), (12, 10, -8)])
        return (
            int(np.clip(base + tint[0], 0, 255)),
            int(np.clip(base + tint[1], 0, 255)),
            int(np.clip(base + tint[2], 0, 255)),
            alpha,
        )

    @staticmethod
    def _draw_parallel_lines(draw: ImageDraw.ImageDraw, width: int, height: int, spacing: int, line_w: int, angle: float, color: Tuple[int, int, int, int]) -> None:
        theta = math.radians(angle)
        dx = math.cos(theta)
        dy = math.sin(theta)
        nx = -dy
        ny = dx
        diag = int(math.hypot(width, height))
        for offset in range(-diag, diag * 2, spacing):
            cx = width // 2 + int(nx * offset)
            cy = height // 2 + int(ny * offset)
            x1 = cx - int(dx * diag)
            y1 = cy - int(dy * diag)
            x2 = cx + int(dx * diag)
            y2 = cy + int(dy * diag)
            draw.line((x1, y1, x2, y2), fill=color, width=line_w)

    def _apply_photometric(self, image: Image.Image, style: str, rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
        arr = np.asarray(image).astype(np.float32)
        height, width = arr.shape[:2]
        if style == "shadow_blob":
            mask = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(mask)
            for _ in range(rng.randint(1, 4)):
                x = rng.randint(-width // 4, width)
                y = rng.randint(-height // 4, height)
                w = rng.randint(width // 5, width)
                h = rng.randint(height // 8, height // 2)
                draw.ellipse((x, y, x + w, y + h), fill=rng.randint(70, 160))
            mask = mask.filter(ImageFilter.GaussianBlur(radius=rng.uniform(20, 80)))
            m = np.asarray(mask).astype(np.float32)[..., None] / 255.0
            arr *= 1.0 - m * rng.uniform(0.25, 0.55)
        elif style == "specular_patch":
            mask = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(mask)
            for _ in range(rng.randint(1, 4)):
                x = rng.randint(0, width)
                y = rng.randint(0, height)
                w = rng.randint(width // 12, width // 3)
                h = rng.randint(height // 25, height // 7)
                draw.ellipse((x - w, y - h, x + w, y + h), fill=rng.randint(80, 190))
            mask = mask.filter(ImageFilter.GaussianBlur(radius=rng.uniform(5, 24)))
            arr += np.asarray(mask).astype(np.float32)[..., None] * rng.uniform(0.25, 0.55)
        elif style == "vignette":
            yy, xx = np.mgrid[0:height, 0:width]
            cx = width * rng.uniform(0.35, 0.65)
            cy = height * rng.uniform(0.35, 0.65)
            dist = np.sqrt(((xx - cx) / width) ** 2 + ((yy - cy) / height) ** 2)
            arr *= (1.0 - np.clip(dist * rng.uniform(0.45, 0.95), 0.0, 0.55))[..., None]
        elif style == "contrast_drop":
            local = cv2.GaussianBlur(arr, (0, 0), rng.uniform(8, 40))
            arr = arr * rng.uniform(0.55, 0.90) + local * rng.uniform(0.10, 0.35)
        elif style == "color_shift":
            shift = np.asarray([rng.uniform(0.82, 1.18), rng.uniform(0.82, 1.18), rng.uniform(0.82, 1.18)], dtype=np.float32)
            arr *= shift.reshape(1, 1, 3)
        elif style == "sensor_noise":
            arr += np_rng.normal(0, rng.uniform(4, 18), arr.shape).astype(np.float32)
        elif style == "defocus":
            radius = rng.uniform(0.4, 2.2)
            image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=radius))
            return image.convert("RGB")
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")

    @staticmethod
    def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(quality))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
