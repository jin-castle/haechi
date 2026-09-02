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


class PidinetError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class PidinetResult:
    probability: Image.Image
    mask: Image.Image
    overlay: Image.Image
    edge_pixels: int
    total_pixels: int
    edge_ratio: float


class PidinetPredictor:
    def __init__(self, checkpoint_path: Path, device: str = "auto") -> None:
        if not checkpoint_path.is_file():
            raise PidinetError(f"missing PiDiNet checkpoint: {checkpoint_path}")
        numpy_module, torch_module = _load_dependencies()
        self._numpy: Any = numpy_module
        self._torch: Any = torch_module
        self._device = _resolve_device(torch_module, device)
        model_module = __import__(
            "firesight_vision.pidinet_model",
            fromlist=["PiDiNet"],
        )
        self._model: Any = model_module.PiDiNet(
            60,
            tuple(model_module.CARV4_OPERATIONS),
            dil=24,
            sa=True,
        ).to(self._device)
        state = _load_checkpoint(torch_module, checkpoint_path, self._device)
        converted = _convert_state_dict(state)
        try:
            self._model.load_state_dict(converted, strict=True)
        except RuntimeError as error:
            message = (
                "PiDiNet checkpoint does not match the bundled network: "
                f"{checkpoint_path}"
            )
            raise PidinetError(message) from error
        self._model.eval()
        self.checkpoint_path = checkpoint_path

    @property
    def device(self) -> str:
        return str(self._device)

    def predict_probability(self, image: Image.Image) -> Image.Image:
        source = image.convert("RGB")
        width, height = source.size
        array = self._numpy.asarray(source, dtype=self._numpy.float32) / 255.0
        mean = self._numpy.asarray(
            (0.485, 0.456, 0.406),
            dtype=self._numpy.float32,
        )
        std = self._numpy.asarray(
            (0.229, 0.224, 0.225),
            dtype=self._numpy.float32,
        )
        normalized = (array - mean) / std
        tensor = self._torch.from_numpy(
            normalized.transpose((2, 0, 1)),
        ).float().unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            outputs = self._model(tensor)
            probability = outputs[-1][0, 0, :height, :width]
        probability_array = (
            probability.detach().cpu().numpy() * 255.0
        ).clip(0.0, 255.0).astype(self._numpy.uint8)
        return Image.fromarray(probability_array, mode="L")

    def process(
        self,
        image: Image.Image,
        threshold: float = 0.30,
        background_scale: float = 0.24,
        edge_width: int = 1,
        ignored_regions: tuple[HudIgnoreRegion, ...] = (),
    ) -> PidinetResult:
        _validate_threshold(threshold)
        _validate_background_scale(background_scale)
        _validate_edge_width(edge_width)
        source = image.convert("RGB")
        probability = mask_ignored_regions(
            self.predict_probability(source),
            ignored_regions,
        )
        cutoff = threshold * MASK_MAX_VALUE
        mask = probability.point(
            lambda value: MASK_MAX_VALUE if value >= cutoff else 0,
        )
        if edge_width != MIN_EDGE_WIDTH:
            kernel_size = edge_width if edge_width % 2 == 1 else edge_width + 1
            mask = mask.filter(ImageFilter.MaxFilter(size=kernel_size))
        dimmed = ImageEnhance.Brightness(source).enhance(background_scale)
        edge_layer = Image.new("RGB", source.size, HUD_EDGE_COLOR)
        overlay = Image.composite(edge_layer, dimmed, mask)
        edge_pixels = sum(value == MASK_MAX_VALUE for value in mask.tobytes())
        total_pixels = source.size[0] * source.size[1]
        return PidinetResult(
            probability=probability,
            mask=mask,
            overlay=overlay,
            edge_pixels=edge_pixels,
            total_pixels=total_pixels,
            edge_ratio=edge_pixels / total_pixels,
        )


def _load_dependencies() -> tuple[Any, Any]:
    if np is None or torch is None:
        raise PidinetError(
            "PiDiNet requires optional dependencies; run `uv sync --extra teed`."
        )
    return np, torch


def _resolve_device(torch_module: Any, requested: str) -> str:
    cuda_available = bool(torch_module.cuda.is_available())
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested.startswith("cuda") and not cuda_available:
        raise PidinetError("PiDiNet requested CUDA, but no CUDA device is available")
    if requested != "cpu" and not requested.startswith("cuda"):
        raise PidinetError(f"unsupported PiDiNet device: {requested}")
    return requested


def _load_checkpoint(torch_module: Any, path: Path, device: str) -> Mapping[str, Any]:
    try:
        loaded = torch_module.load(path, map_location=device, weights_only=True)
    except TypeError:
        loaded = torch_module.load(path, map_location=device)
    if isinstance(loaded, dict) and isinstance(loaded.get("state_dict"), dict):
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise PidinetError(f"unsupported PiDiNet checkpoint format: {path}")
    return {
        key.removeprefix("module."): value for key, value in loaded.items()
    }


def _convert_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    operations = (
        "cd",
        "ad",
        "rd",
        "cv",
        "cd",
        "ad",
        "rd",
        "cv",
        "cd",
        "ad",
        "rd",
        "cv",
        "cd",
        "ad",
        "rd",
        "cv",
    )
    converted: dict[str, Any] = {}
    for name, weight in state.items():
        if name == "init_block.weight":
            converted[name] = _convert_pdc_weight(operations[0], weight)
        elif name.startswith("block") and name.endswith("conv1.weight"):
            block_index = _block_conv_index(name)
            converted[name] = _convert_pdc_weight(
                operations[block_index],
                weight,
            )
        else:
            converted[name] = weight
    return converted


def _block_conv_index(name: str) -> int:
    parts = name.split(".")
    block_name = parts[0]
    block_number = int(block_name[5])
    block_position = int(block_name[7])
    offsets = {1: 1, 2: 4, 3: 8, 4: 12}
    return offsets[block_number] + block_position - 1


def _convert_pdc_weight(operation: str, weight: Any) -> Any:
    if operation == "cv":
        return weight
    shape = weight.shape
    flat = weight.view(shape[0], shape[1], -1)
    if operation == "cd":
        converted = weight.clone().view(shape[0], shape[1], -1)
        converted[:, :, 4] -= flat.sum(dim=2)
        return converted.view(shape)
    if operation == "ad":
        return (
            flat - flat[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        ).view(shape)
    if operation == "rd":
        converted = weight.new_zeros((shape[0], shape[1], 25))
        converted[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = flat[:, :, 1:]
        converted[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -flat[:, :, 1:]
        return converted.view(shape[0], shape[1], 5, 5)
    raise PidinetError(f"unsupported PiDiNet operation: {operation}")


def _validate_threshold(threshold: float) -> None:
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise PidinetError("PiDiNet threshold must be a finite value between 0 and 1")


def _validate_background_scale(background_scale: float) -> None:
    if (
        not math.isfinite(background_scale)
        or background_scale <= 0.0
        or background_scale > 1.0
    ):
        raise PidinetError(
            "PiDiNet background scale must be a finite value in (0, 1]",
        )


def _validate_edge_width(edge_width: int) -> None:
    if edge_width < MIN_EDGE_WIDTH or edge_width > MAX_EDGE_WIDTH:
        raise PidinetError("PiDiNet edge width must be between 1 and 15")
