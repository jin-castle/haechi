from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from PIL import Image

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


DEFAULT_DEPTH_MODEL_ID: Final = (
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
)
DEFAULT_DEPTH_MODEL_REVISION: Final = "8078d68a9c75a972131914f6afd0c1723be0da7f"


class DepthAnythingError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class DepthAnythingResult:
    depth_m: NDArray[np.float32]
    preview: Image.Image
    valid_mask: Image.Image
    valid_pixels: int
    total_pixels: int
    valid_ratio: float
    min_depth_m: float
    median_depth_m: float
    p95_depth_m: float
    max_depth_m: float


class DepthAnythingPredictor:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_DEPTH_MODEL_ID,
        revision: str = DEFAULT_DEPTH_MODEL_REVISION,
        device: str = "auto",
        preview_max_depth_m: float = 6.0,
    ) -> None:
        if preview_max_depth_m <= 0.0:
            message = "preview max depth must be greater than zero"
            raise DepthAnythingError(message)
        (
            self._numpy,
            self._torch,
            image_processor_class,
            model_class,
        ) = _load_dependencies()
        self._device = _resolve_device(self._torch, device)
        try:
            self._processor: Any = image_processor_class.from_pretrained(
                model_id,
                revision=revision,
                use_fast=False,
            )
            self._model: Any = model_class.from_pretrained(
                model_id,
                revision=revision,
            ).to(self._device)
        except (OSError, ValueError) as error:
            message = f"unable to load Depth Anything model: {model_id}@{revision}"
            raise DepthAnythingError(message) from error
        self._model.eval()
        self.model_id = model_id
        self.revision = revision
        self.preview_max_depth_m = preview_max_depth_m
        self.model_max_depth_m = float(self._model.config.max_depth)

    @property
    def device(self) -> str:
        return str(self._device)

    def process(self, image: Image.Image) -> DepthAnythingResult:
        source = image.convert("RGB")
        width, height = source.size
        inputs = self._processor(images=source, return_tensors="pt")
        device_inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            outputs = self._model(**device_inputs)
            predicted = (
                self._torch.nn.functional.interpolate(
                    outputs.predicted_depth.unsqueeze(1),
                    size=(height, width),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )

        depth_m = (
            predicted.detach()
            .cpu()
            .numpy()
            .astype(
                self._numpy.float32,
                copy=False,
            )
        )
        valid = (
            self._numpy.isfinite(depth_m)
            & (depth_m > 0.0)
            & (depth_m <= self.model_max_depth_m)
        )
        clean_depth = self._numpy.where(valid, depth_m, 0.0).astype(
            self._numpy.float32,
            copy=False,
        )
        valid_values = clean_depth[valid]
        valid_pixels = int(valid.sum())
        total_pixels = width * height
        if valid_pixels == 0:
            min_depth_m = 0.0
            median_depth_m = 0.0
            p95_depth_m = 0.0
            max_depth_m = 0.0
        else:
            min_depth_m = float(valid_values.min())
            median_depth_m = float(self._numpy.median(valid_values))
            p95_depth_m = float(self._numpy.quantile(valid_values, 0.95))
            max_depth_m = float(valid_values.max())

        return DepthAnythingResult(
            depth_m=clean_depth,
            preview=_render_depth_preview(
                clean_depth,
                valid,
                self.preview_max_depth_m,
                self._numpy,
            ),
            valid_mask=Image.fromarray(
                (valid.astype(self._numpy.uint8) * 255),
                mode="L",
            ),
            valid_pixels=valid_pixels,
            total_pixels=total_pixels,
            valid_ratio=valid_pixels / total_pixels,
            min_depth_m=min_depth_m,
            median_depth_m=median_depth_m,
            p95_depth_m=p95_depth_m,
            max_depth_m=max_depth_m,
        )


def _load_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ModuleNotFoundError as error:
        message = (
            "Depth Anything requires optional dependencies; "
            "run `uv sync --extra depth`."
        )
        raise DepthAnythingError(message) from error
    return numpy, torch, AutoImageProcessor, AutoModelForDepthEstimation


def _resolve_device(torch_module: Any, requested: str) -> str:
    cuda_available = bool(torch_module.cuda.is_available())
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested.startswith("cuda") and not cuda_available:
        message = "Depth Anything requested CUDA, but no CUDA device is available"
        raise DepthAnythingError(message)
    if requested != "cpu" and not requested.startswith("cuda"):
        message = f"unsupported Depth Anything device: {requested}"
        raise DepthAnythingError(message)
    return requested


def _render_depth_preview(
    depth_m: NDArray[np.float32],
    valid: NDArray[np.bool_],
    preview_max_depth_m: float,
    numpy_module: Any,
) -> Image.Image:
    normalized = numpy_module.clip(depth_m / preview_max_depth_m, 0.0, 1.0)
    colors = numpy_module.zeros((*depth_m.shape, 3), dtype=numpy_module.uint8)
    colors[..., 0] = (255.0 * (1.0 - normalized)).astype(numpy_module.uint8)
    colors[..., 1] = (
        255.0 * (1.0 - numpy_module.abs((2.0 * normalized) - 1.0))
    ).astype(numpy_module.uint8)
    colors[..., 2] = (255.0 * normalized).astype(numpy_module.uint8)
    colors[~valid] = 0
    return Image.fromarray(colors, mode="RGB")
