"""Geometry provider backends, including an optional MoGe-2 integration."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Optional

import cv2
import numpy as np

from ..types import GeometryFrame


def _resolve_torch_device(device: Optional[str], torch_module: Any) -> Any:
    raw_device = str(device or "cpu")
    if raw_device.isdigit():
        if torch_module.cuda.is_available():
            return torch_module.device(f"cuda:{raw_device}")
        return torch_module.device("cpu")
    if raw_device.startswith("cuda") and not torch_module.cuda.is_available():
        return torch_module.device("cpu")
    return torch_module.device(raw_device)


def _to_numpy(array: Any) -> Any:
    if array is None:
        return None
    if hasattr(array, "detach"):
        array = array.detach().float().cpu().numpy()
    if isinstance(array, np.ndarray):
        if array.ndim > 0 and array.shape[0] == 1:
            return np.squeeze(array, axis=0)
        return array
    return np.asarray(array)


def _infer_with_safe_postprocess(
    model: Any,
    torch_module: Any,
    utils3d_module: Any,
    recover_focal_shift_fn: Any,
    image: Any,
    resolution_level: int,
    apply_mask: bool,
    use_fp16: bool,
) -> dict[str, Any]:
    """Run MoGe inference while avoiding the upstream CPU autocast bug."""

    torch = torch_module
    if image.dim() == 3:
        omit_batch_dim = True
        image = image.unsqueeze(0)
    else:
        omit_batch_dim = False
    image = image.to(dtype=model.dtype, device=model.device)

    original_height, original_width = image.shape[-2:]
    aspect_ratio = original_width / original_height
    min_tokens, max_tokens = model.num_tokens_range
    num_tokens = int(min_tokens + (int(resolution_level) / 9.0) * (max_tokens - min_tokens))

    forward_context = (
        torch.autocast(device_type=model.device.type, dtype=torch.float16)
        if use_fp16 and model.device.type == "cuda" and model.dtype != torch.float16
        else nullcontext()
    )
    with forward_context:
        output = model.forward(image, num_tokens=num_tokens)

    points = output.get("points", None)
    normal = output.get("normal", None)
    mask = output.get("mask", None)
    metric_scale = output.get("metric_scale", None)

    points, normal, mask, metric_scale = map(
        lambda item: item.float() if isinstance(item, torch.Tensor) else item,
        [points, normal, mask, metric_scale],
    )

    if mask is not None:
        mask_binary = mask > 0.5
    else:
        mask_binary = None

    if points is not None:
        focal, shift = recover_focal_shift_fn(points, mask_binary)
        fx = focal / 2 * (1 + aspect_ratio**2) ** 0.5 / aspect_ratio
        fy = focal / 2 * (1 + aspect_ratio**2) ** 0.5
        intrinsics = utils3d_module.pt.intrinsics_from_focal_center(
            fx,
            fy,
            torch.tensor(0.5, device=points.device, dtype=points.dtype),
            torch.tensor(0.5, device=points.device, dtype=points.dtype),
        )
        points[..., 2] += shift[..., None, None]
        if mask_binary is not None:
            mask_binary &= points[..., 2] > 0
        depth = points[..., 2].clone()
    else:
        depth, intrinsics = None, None

    if depth is not None:
        points = utils3d_module.pt.depth_map_to_point_map(depth, intrinsics=intrinsics)

    if metric_scale is not None:
        if points is not None:
            points *= metric_scale[:, None, None, None]
        if depth is not None:
            depth *= metric_scale[:, None, None]

    if apply_mask and mask_binary is not None:
        if points is not None:
            points = torch.where(mask_binary[..., None], points, torch.inf)
        if depth is not None:
            depth = torch.where(mask_binary, depth, torch.inf)
        if normal is not None:
            normal = torch.where(mask_binary[..., None], normal, torch.zeros_like(normal))

    result = {
        "points": points,
        "intrinsics": intrinsics,
        "depth": depth,
        "mask": mask_binary,
        "normal": normal,
    }
    result = {key: value for key, value in result.items() if value is not None}

    if omit_batch_dim:
        result = {key: value.squeeze(0) for key, value in result.items()}
    return result


class GradientGeometryProvider:
    """A lightweight geometry stub used when no foundation model is configured."""

    def __init__(self, device: Optional[str] = "cpu") -> None:
        self.device = device

    def infer(self, frame_bgr: np.ndarray) -> GeometryFrame:
        height, width = frame_bgr.shape[:2]
        depth = np.tile(np.linspace(0.2, 1.0, width, dtype=np.float32), (height, 1))
        xs = np.tile(np.linspace(-1.0, 1.0, width, dtype=np.float32), (height, 1))
        ys = np.tile(np.linspace(-1.0, 1.0, height, dtype=np.float32).reshape(height, 1), (1, width))
        points = np.stack([xs, ys, depth], axis=-1)
        normals = np.zeros_like(points)
        normals[..., 2] = 1.0
        valid_mask = np.ones((height, width), dtype=bool)
        intrinsics = np.array(
            [
                [width, 0.0, width * 0.5],
                [0.0, width, height * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return GeometryFrame(
            depth=depth,
            points=points,
            normals=normals,
            valid_mask=valid_mask,
            intrinsics=intrinsics,
            extras={"provider": "gradient"},
        )


class MoGeGeometryProvider:
    """Optional MoGe-2 geometry backend."""

    def __init__(
        self,
        model_name: str = "Ruicheng/moge-2-vits-normal",
        device: Optional[str] = "cpu",
        use_fp16: bool = True,
        resolution_level: int = 9,
        apply_mask: bool = True,
    ) -> None:
        try:
            import torch
            from moge.model.v2 import MoGeModel
            from moge.model import v2 as moge_v2
        except Exception as exc:  # pragma: no cover - dependency-dependent path
            raise RuntimeError(
                "MoGe-2 is not installed. Install it with "
                "`pip install git+https://github.com/microsoft/MoGe.git`."
            ) from exc

        self.torch = torch
        self.utils3d = moge_v2.utils3d
        self.recover_focal_shift = moge_v2.recover_focal_shift
        self.device = _resolve_torch_device(device, torch)
        self.use_fp16 = bool(use_fp16) and self.device.type == "cuda"
        self.model_name = model_name
        self.resolution_level = int(resolution_level)
        self.apply_mask = bool(apply_mask)
        self.model = MoGeModel.from_pretrained(model_name).to(self.device).eval()

    def infer(self, frame_bgr: np.ndarray) -> GeometryFrame:
        torch = self.torch
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output = _infer_with_safe_postprocess(
                model=self.model,
                torch_module=torch,
                utils3d_module=self.utils3d,
                recover_focal_shift_fn=self.recover_focal_shift,
                image=tensor,
                resolution_level=self.resolution_level,
                apply_mask=self.apply_mask,
                use_fp16=self.use_fp16,
            )

        depth = _to_numpy(output.get("depth"))
        points = _to_numpy(output.get("points"))
        normals = _to_numpy(output.get("normal"))
        valid_mask = _to_numpy(output.get("mask"))
        intrinsics = _to_numpy(output.get("intrinsics"))
        confidence = _to_numpy(output.get("confidence"))

        if depth is None:
            raise RuntimeError("MoGe-2 output did not include a depth map.")

        valid_ratio = None
        if valid_mask is not None:
            valid_ratio = float(np.mean(valid_mask > 0))

        return GeometryFrame(
            depth=np.asarray(depth, dtype=np.float32),
            points=None if points is None else np.asarray(points, dtype=np.float32),
            normals=None if normals is None else np.asarray(normals, dtype=np.float32),
            valid_mask=None if valid_mask is None else np.asarray(valid_mask).astype(bool),
            intrinsics=None if intrinsics is None else np.asarray(intrinsics, dtype=np.float32),
            confidence=None if confidence is None else np.asarray(confidence, dtype=np.float32),
            extras={
                "provider": "moge",
                "model_name": self.model_name,
                "device": str(self.device),
                "valid_ratio": valid_ratio,
                "has_points": points is not None,
                "has_normals": normals is not None,
            },
        )
