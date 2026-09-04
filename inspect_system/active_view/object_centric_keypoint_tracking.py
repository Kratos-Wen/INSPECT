"""Long-term, cycle-consistent point tracking for object-centric transport."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class LongTermTracks:
    before_pixels: np.ndarray
    after_pixels: np.ndarray
    visibility_ratio: float
    cycle_error: float
    query_count: int
    method: str = "cotracker3_offline"


def rectangle_mask(
    shape: tuple[int, int],
    bbox: Sequence[float] | None = None,
    *,
    inset: float = 0.06,
    exclude: Iterable[Sequence[float]] = (),
) -> np.ndarray:
    height, width = shape
    mask = np.full((height, width), 255, dtype=np.uint8)
    if bbox is not None:
        mask.fill(0)
        x1, y1, x2, y2 = (float(value) for value in bbox)
        dx = max(0.0, x2 - x1) * float(inset)
        dy = max(0.0, y2 - y1) * float(inset)
        left = max(0, min(width - 1, int(round(x1 + dx))))
        top = max(0, min(height - 1, int(round(y1 + dy))))
        right = max(left + 1, min(width, int(round(x2 - dx))))
        bottom = max(top + 1, min(height, int(round(y2 - dy))))
        mask[top:bottom, left:right] = 255
    else:
        margin_x = max(8, int(round(width * 0.04)))
        margin_y = max(8, int(round(height * 0.04)))
        mask[:margin_y] = 0
        mask[-margin_y:] = 0
        mask[:, :margin_x] = 0
        mask[:, -margin_x:] = 0
    for raw in exclude:
        x1, y1, x2, y2 = (float(value) for value in raw)
        dx = max(4.0, (x2 - x1) * 0.08)
        dy = max(4.0, (y2 - y1) * 0.08)
        left = max(0, int(round(x1 - dx)))
        top = max(0, int(round(y1 - dy)))
        right = min(width, int(round(x2 + dx)))
        bottom = min(height, int(round(y2 + dy)))
        mask[top:bottom, left:right] = 0
    return mask


def sample_queries(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    max_corners: int = 320,
    grid_step: int = 24,
) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max(32, int(max_corners)),
        qualityLevel=0.002,
        minDistance=4.0,
        mask=mask,
        blockSize=7,
        useHarrisDetector=False,
    )
    rows: list[np.ndarray] = []
    if corners is not None:
        rows.append(corners.reshape(-1, 2).astype(np.float64))
    step = max(8, int(grid_step))
    ys, xs = np.mgrid[
        step // 2 : gray.shape[0] : step,
        step // 2 : gray.shape[1] : step,
    ]
    grid = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    gx = np.clip(np.rint(grid[:, 0]).astype(int), 0, gray.shape[1] - 1)
    gy = np.clip(np.rint(grid[:, 1]).astype(int), 0, gray.shape[0] - 1)
    grid = grid[mask[gy, gx] > 0]
    if grid.size:
        rows.append(grid)
    if not rows:
        raise ValueError("tracking mask contains no query points")
    points = np.concatenate(rows, axis=0)
    quantized = np.rint(points / 2.0).astype(np.int64)
    _, unique = np.unique(quantized, axis=0, return_index=True)
    points = points[np.sort(unique)]
    if points.shape[0] < 12:
        raise ValueError("fewer than twelve long-term tracking queries")
    return points[: max(32, int(max_corners))]


class CoTracker3LongTermTracker:
    def __init__(
        self,
        *,
        device: str = "cuda:0",
        torch_home: str | Path | None = None,
        cycle_threshold: float = 0.035,
        minimum_visibility: float = 0.35,
        max_side: int = 960,
    ) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if torch_home is not None:
            os.environ["TORCH_HOME"] = str(Path(torch_home))
        self.model = torch.hub.load(
            "facebookresearch/co-tracker",
            "cotracker3_offline",
            trust_repo=True,
        ).to(self.device)
        self.model.eval()
        self.cycle_threshold = float(cycle_threshold)
        self.minimum_visibility = float(minimum_visibility)
        self.max_side = max(384, int(max_side))

    def _video_tensor(self, frames_bgr: Sequence[np.ndarray]):
        if len(frames_bgr) < 2:
            raise ValueError("at least two frames are required")
        rgb = np.stack(
            [np.ascontiguousarray(frame[:, :, ::-1]) for frame in frames_bgr]
        )
        return (
            self.torch.from_numpy(rgb)
            .permute(0, 3, 1, 2)[None]
            .float()
            .to(self.device)
        )

    def track(
        self,
        frames_bgr: Sequence[np.ndarray],
        queries_xy: np.ndarray,
    ) -> LongTermTracks:
        torch = self.torch
        original_height, original_width = frames_bgr[0].shape[:2]
        scale = min(1.0, self.max_side / float(max(original_height, original_width)))
        if scale < 1.0:
            tracked_frames = [
                cv2.resize(
                    frame,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
                for frame in frames_bgr
            ]
        else:
            tracked_frames = list(frames_bgr)
        video = self._video_tensor(tracked_frames)
        original_queries = np.asarray(queries_xy, dtype=np.float32)
        queries_xy = original_queries * scale
        query = np.column_stack(
            [np.zeros((queries_xy.shape[0],), dtype=np.float32), queries_xy]
        )
        query_tensor = torch.from_numpy(query)[None].to(self.device)
        with torch.inference_mode():
            tracks, visible = self.model(video, queries=query_tensor)
        forward = tracks[0, -1].detach().cpu().numpy()
        visible_np = visible[0].detach().cpu().numpy().astype(bool)
        final_visible = visible_np[-1]
        visibility_fraction = visible_np.mean(axis=0)

        reverse_query = np.column_stack(
            [
                np.full(
                    (forward.shape[0],),
                    len(tracked_frames) - 1,
                    dtype=np.float32,
                ),
                forward.astype(np.float32),
            ]
        )
        with torch.inference_mode():
            reverse_tracks, reverse_visible = self.model(
                video,
                queries=torch.from_numpy(reverse_query)[None].to(self.device),
                backward_tracking=True,
            )
        reverse_start = reverse_tracks[0, 0].detach().cpu().numpy()
        reverse_ok = reverse_visible[0, 0].detach().cpu().numpy().astype(bool)
        height, width = tracked_frames[-1].shape[:2]
        diagonal = float(np.hypot(width, height))
        cycle = np.linalg.norm(reverse_start - queries_xy, axis=1) / max(1.0, diagonal)
        inside = (
            (forward[:, 0] >= 0.0)
            & (forward[:, 0] < width)
            & (forward[:, 1] >= 0.0)
            & (forward[:, 1] < height)
        )
        valid = (
            final_visible
            & reverse_ok
            & inside
            & (visibility_fraction >= self.minimum_visibility)
            & (cycle <= self.cycle_threshold)
        )
        if int(valid.sum()) < 8:
            raise ValueError(
                "CoTracker retained fewer than eight cycle-consistent tracks"
            )
        return LongTermTracks(
            before_pixels=original_queries[valid].astype(np.float64),
            after_pixels=(forward[valid] / scale).astype(np.float64),
            visibility_ratio=float(np.mean(visibility_fraction[valid])),
            cycle_error=float(np.median(cycle[valid])),
            query_count=int(queries_xy.shape[0]),
        )
