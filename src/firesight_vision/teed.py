from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageEnhance, ImageFilter

from firesight_vision.hud_edges import (
    HUD_EDGE_COLOR,
    HudIgnoreRegion,
    mask_ignored_regions,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

try:
    import numpy as np
    import torch
except ModuleNotFoundError:
    np = None
    torch = None


MASK_MAX_VALUE = 255
MIN_EDGE_WIDTH = 1
MAX_EDGE_WIDTH = 15


class TeedError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class TeedResult:
    probability: Image.Image
    mask: Image.Image
    overlay: Image.Image
    edge_pixels: int
    total_pixels: int
    edge_ratio: float


class TeedPredictor:
    def __init__(self, checkpoint_path: Path, device: str = "auto") -> None:
        if not checkpoint_path.is_file():
            message = f"missing TEED checkpoint: {checkpoint_path}"
            raise TeedError(message)
        numpy_module, torch_module = _load_dependencies()
        self._numpy: Any = numpy_module
        self._torch: Any = torch_module
        self._device = _resolve_device(torch_module, device)
        model_module = __import__("firesight_vision.teed_model", fromlist=["TED"])
        self._model: Any = model_module.TED().to(self._device)
        state = _load_checkpoint(torch_module, checkpoint_path, self._device)
        try:
            self._model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            message = (
                "TEED checkpoint does not match the bundled network: "
                f"{checkpoint_path}"
            )
            raise TeedError(message) from error
        self._model.eval()
        self.checkpoint_path = checkpoint_path

    @property
    def device(self) -> str:
        return str(self._device)

    def predict_probability(self, image: Image.Image) -> Image.Image:
        source = image.convert("RGB")
        width, height = source.size
        array = self._numpy.asarray(source, dtype=self._numpy.float32)
        bgr = array[:, :, ::-1].copy()
        mean_bgr = self._numpy.asarray(
            (103.939, 116.779, 123.68),
            dtype=self._numpy.float32,
        )
        bgr -= mean_bgr
        tensor = self._torch.from_numpy(bgr.transpose((2, 0, 1))).float().unsqueeze(0)
        pad_width = (8 - width % 8) % 8
        pad_height = (8 - height % 8) % 8
        if pad_width or pad_height:
            tensor = self._torch.nn.functional.pad(
                tensor,
                (0, pad_width, 0, pad_height),
                mode="replicate",
            )
        tensor = tensor.to(self._device)
        with self._torch.inference_mode():
            outputs = self._model(tensor, single_test=True)
            probability = self._torch.sigmoid(outputs[-1])[0, 0, :height, :width]
        probability_array = (
            probability.detach().cpu().numpy() * 255.0
        ).clip(0.0, 255.0).astype(self._numpy.uint8)
        return Image.fromarray(probability_array, mode="L")

    def process(
        self,
        image: Image.Image,
        threshold: float = 0.35,
        background_scale: float = 0.24,
        edge_width: int = 3,
        ignored_regions: tuple[HudIgnoreRegion, ...] = (),
    ) -> TeedResult:
        _validate_threshold(threshold)
        _validate_background_scale(background_scale)
        _validate_edge_width(edge_width)
        source = image.convert("RGB")
        probability = mask_ignored_regions(
            self.predict_probability(source),
            ignored_regions,
        )
        cutoff = threshold * 255.0
        mask = probability.point(
            lambda value: 255 if value >= cutoff else 0,
        )
        if edge_width != 1:
            kernel_size = edge_width if edge_width % 2 == 1 else edge_width + 1
            mask = mask.filter(ImageFilter.MaxFilter(size=kernel_size))
        dimmed = ImageEnhance.Brightness(source).enhance(background_scale)
        edge_layer = Image.new("RGB", source.size, HUD_EDGE_COLOR)
        overlay = Image.composite(edge_layer, dimmed, mask)
        edge_pixels = sum(value == MASK_MAX_VALUE for value in mask.tobytes())
        total_pixels = source.size[0] * source.size[1]
        return TeedResult(
            probability=probability,
            mask=mask,
            overlay=overlay,
            edge_pixels=edge_pixels,
            total_pixels=total_pixels,
            edge_ratio=edge_pixels / total_pixels,
        )


def _load_dependencies() -> tuple[Any, Any]:
    if np is None or torch is None:
        message = "TEED requires optional dependencies; run `uv sync --extra teed`."
        raise TeedError(message)
    return np, torch


def _resolve_device(torch_module: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        message = "TEED requested CUDA, but no CUDA device is available"
        raise TeedError(message)
    if requested != "cpu" and not requested.startswith("cuda"):
        message = f"unsupported TEED device: {requested}"
        raise TeedError(message)
    return requested


def _load_checkpoint(torch_module: Any, path: Path, device: str) -> Mapping[str, Any]:
    try:
        loaded = torch_module.load(path, map_location=device, weights_only=True)
    except TypeError:
        loaded = torch_module.load(path, map_location=device)
    if isinstance(loaded, dict) and isinstance(loaded.get("state_dict"), dict):
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict):
        message = f"unsupported TEED checkpoint format: {path}"
        raise TeedError(message)
    if any(key.startswith("module.") for key in loaded):
        return {key.removeprefix("module."): value for key, value in loaded.items()}
    return loaded


def _validate_threshold(threshold: float) -> None:
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        message = "TEED threshold must be a finite value between 0 and 1"
        raise TeedError(message)


def _validate_background_scale(background_scale: float) -> None:
    if (
        not math.isfinite(background_scale)
        or background_scale <= 0.0
        or background_scale > 1.0
    ):
        message = "background scale must be a finite value in (0, 1]"
        raise TeedError(message)


def _validate_edge_width(edge_width: int) -> None:
    if edge_width < MIN_EDGE_WIDTH or edge_width > MAX_EDGE_WIDTH:
        message = "TEED edge width must be between 1 and 15"
        raise TeedError(message)
