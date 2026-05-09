"""Promptable segmentation backends used by INSPECT interaction evidence."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from ..types import SegmentationMask


def _slug(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _mask_to_box(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0 or ys.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


class NullSegmentationBackend:
    """No-op segmentation backend."""

    backend_name = "none"

    def segment(self, frame_bgr: np.ndarray, prompts: Iterable[str]) -> List[SegmentationMask]:
        return []


class SAM3VideoSegmentationBackend:
    """Hugging Face SAM3 Video promptable concept segmentation backend.

    This backend follows the Hugging Face `Sam3VideoModel` / `Sam3VideoProcessor`
    streaming API. It is optional: dependencies and checkpoints are loaded only
    when this backend is selected.
    """

    backend_name = "sam3_video"

    def __init__(
        self,
        model_name: str = "facebook/sam3",
        device: str = "auto",
        prompts: Optional[Iterable[str]] = None,
        score_threshold: float = 0.45,
        max_masks: int = 32,
        processing_device: str = "cpu",
        video_storage_device: str = "cpu",
    ) -> None:
        try:
            import torch
            from transformers import Sam3VideoModel, Sam3VideoProcessor
        except ImportError as exc:  # pragma: no cover - optional backend
            raise RuntimeError(
                "SAM3 backend requires transformers with SAM3 support and torch. "
                "Install a recent transformers build and the SAM3 checkpoint dependencies."
            ) from exc

        self.torch = torch
        self.model_name = str(model_name)
        self.device = self._resolve_device(device)
        self.score_threshold = float(score_threshold)
        self.max_masks = max(1, int(max_masks))
        self.processor = Sam3VideoProcessor.from_pretrained(self.model_name)
        device_map = "auto" if str(device).strip().lower() == "auto" else None
        if device_map is None:
            self.model = Sam3VideoModel.from_pretrained(self.model_name).to(self.device)
        else:
            self.model = Sam3VideoModel.from_pretrained(self.model_name, device_map=device_map)
        self.model.eval()
        self.prompts = [_slug(prompt) for prompt in (prompts or []) if str(prompt).strip()]
        self.session = self.processor.init_video_session(
            inference_device=self.device,
            processing_device=processing_device,
            video_storage_device=video_storage_device,
        )
        if self.prompts:
            self.session = self.processor.add_text_prompt(self.session, list(self.prompts))

    def segment(self, frame_bgr: np.ndarray, prompts: Iterable[str]) -> List[SegmentationMask]:
        prompts = [_slug(prompt) for prompt in prompts if str(prompt).strip()]
        new_prompts = [prompt for prompt in prompts if prompt not in self.prompts]
        if new_prompts:
            self.session = self.processor.add_text_prompt(self.session, new_prompts)
            self.prompts.extend(new_prompts)
        if not self.prompts:
            return []

        frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        inputs = self.processor(images=frame_rgb, device=self.device, return_tensors="pt")
        model_device = getattr(self.model, "device", self.device)
        inputs = inputs.to(model_device)
        with self.torch.no_grad():
            outputs = self.model(
                inference_session=self.session,
                frame=inputs.pixel_values[0],
                reverse=False,
            )
        processed = self.processor.postprocess_outputs(
            self.session,
            outputs,
            original_sizes=getattr(inputs, "original_sizes", None),
        )
        return self._convert_outputs(processed)

    def _convert_outputs(self, processed: object) -> List[SegmentationMask]:
        if not isinstance(processed, dict):
            return []
        masks = processed.get("masks")
        boxes = processed.get("boxes")
        scores = processed.get("scores")
        object_ids = processed.get("object_ids")
        prompt_to_obj_ids = processed.get("prompt_to_obj_ids") or {}
        if masks is None:
            return []
        masks_np = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
        boxes_np = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes) if boxes is not None else None
        scores_np = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores) if scores is not None else None
        object_ids_np = (
            object_ids.detach().cpu().numpy()
            if hasattr(object_ids, "detach")
            else np.asarray(object_ids)
            if object_ids is not None
            else np.arange(len(masks_np))
        )
        id_to_prompt: dict[int, str] = {}
        for prompt, ids in dict(prompt_to_obj_ids).items():
            ids_np = ids.detach().cpu().numpy() if hasattr(ids, "detach") else np.asarray(ids)
            for object_id in ids_np.tolist():
                id_to_prompt[int(object_id)] = _slug(prompt)

        results: List[SegmentationMask] = []
        for index, mask in enumerate(masks_np[: self.max_masks]):
            score = float(scores_np[index]) if scores_np is not None and index < len(scores_np) else 1.0
            if score < self.score_threshold:
                continue
            object_id = int(object_ids_np[index]) if index < len(object_ids_np) else index
            box = (
                tuple(float(v) for v in boxes_np[index])
                if boxes_np is not None and index < len(boxes_np)
                else _mask_to_box(mask)
            )
            results.append(
                SegmentationMask(
                    name=id_to_prompt.get(object_id, "sam3_object"),
                    xyxy=box,  # type: ignore[arg-type]
                    score=score,
                    mask=mask.astype(bool),
                    object_id=object_id,
                    source=self.backend_name,
                )
            )
        return results

    def _resolve_device(self, device: str) -> str:
        text = str(device).strip().lower()
        if text == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        if text.isdigit():
            return f"cuda:{text}"
        return text


class SAM3OfficialImageSegmentationBackend:
    """Official Meta SAM3/SAM3.1 per-frame concept segmentation backend.

    SAM3.1 checkpoints currently require the official `facebookresearch/sam3`
    package rather than Hugging Face Transformers integration. This backend uses
    the official image processor per frame, which is the best-supported way to
    ground concepts inside this frame-by-frame pipeline. Video-level SAM3.1
    tracking should be used offline when full videos are available.
    """

    backend_name = "sam3_official_image"

    def __init__(
        self,
        model_name: str = "facebook/sam3.1",
        device: str = "auto",
        prompts: Optional[Iterable[str]] = None,
        score_threshold: float = 0.45,
        max_masks: int = 32,
    ) -> None:
        try:
            import torch
            from PIL import Image
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:  # pragma: no cover - optional backend
            raise RuntimeError(
                "SAM3.1 official backend requires the facebookresearch/sam3 package, "
                "Pillow, and torch. Install the latest SAM3 repo and request access "
                "to the SAM3.1 checkpoints."
            ) from exc

        self.torch = torch
        self.Image = Image
        self.model_name = str(model_name)
        self.device = self._resolve_device(device)
        self.prompts = [_slug(prompt) for prompt in (prompts or []) if str(prompt).strip()]
        self.score_threshold = float(score_threshold)
        self.max_masks = max(1, int(max_masks))
        self.model = build_sam3_image_model()
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.processor = Sam3Processor(self.model)

    def segment(self, frame_bgr: np.ndarray, prompts: Iterable[str]) -> List[SegmentationMask]:
        prompt_list = [_slug(prompt) for prompt in prompts if str(prompt).strip()]
        for prompt in prompt_list:
            if prompt not in self.prompts:
                self.prompts.append(prompt)
        if not self.prompts:
            return []
        frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = self.Image.fromarray(frame_rgb)
        state = self.processor.set_image(image)
        results: List[SegmentationMask] = []
        for prompt in self.prompts:
            with self.torch.no_grad():
                output = self.processor.set_text_prompt(state=state, prompt=prompt)
            results.extend(self._convert_outputs(output, prompt))
            if len(results) >= self.max_masks:
                break
        return sorted(results, key=lambda item: item.score, reverse=True)[: self.max_masks]

    def _convert_outputs(self, output: object, prompt: str) -> List[SegmentationMask]:
        if not isinstance(output, dict):
            return []
        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")
        if masks is None:
            return []
        masks_np = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
        boxes_np = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes) if boxes is not None else None
        scores_np = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores) if scores is not None else None
        results: List[SegmentationMask] = []
        for index, mask in enumerate(masks_np):
            score = float(scores_np[index]) if scores_np is not None and index < len(scores_np) else 1.0
            if score < self.score_threshold:
                continue
            box = (
                tuple(float(v) for v in boxes_np[index])
                if boxes_np is not None and index < len(boxes_np)
                else _mask_to_box(mask)
            )
            results.append(
                SegmentationMask(
                    name=prompt,
                    xyxy=box,  # type: ignore[arg-type]
                    score=score,
                    mask=np.asarray(mask).astype(bool),
                    object_id=index,
                    source=self.backend_name,
                )
            )
        return results

    def _resolve_device(self, device: str) -> str:
        text = str(device).strip().lower()
        if text == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        if text.isdigit():
            return f"cuda:{text}"
        return text


def build_segmentation_backend(
    backend: str = "none",
    model_name: str = "facebook/sam3.1",
    device: str = "auto",
    prompts: Optional[Iterable[str]] = None,
    score_threshold: float = 0.45,
    max_masks: int = 32,
) -> object:
    """Build the requested segmentation backend.

    `auto` means SAM3.1 official-first. If dependencies are unavailable, the
    caller receives a clear RuntimeError instead of silently using weaker masks.
    """

    name = str(backend or "none").strip().lower()
    if name in {"none", "off", "disabled", "false"}:
        return NullSegmentationBackend()
    if name in {"auto", "sam3", "sam31", "sam3.1", "sam3_official", "sam31_official", "sam3_official_image"}:
        return SAM3OfficialImageSegmentationBackend(
            model_name=model_name,
            device=device,
            prompts=prompts,
            score_threshold=score_threshold,
            max_masks=max_masks,
        )
    if name in {"sam3_hf", "sam3_video", "sam3_video_hf"}:
        return SAM3VideoSegmentationBackend(
            model_name=model_name,
            device=device,
            prompts=prompts,
            score_threshold=score_threshold,
            max_masks=max_masks,
        )
    raise ValueError(f"Unsupported segmentation backend: {backend}")
