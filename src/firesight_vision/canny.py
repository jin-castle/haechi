from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Protocol, cast

from PIL import Image, ImageEnhance, ImageFilter

from firesight_vision.hud_edges import (
    HUD_EDGE_COLOR,
    HudIgnoreRegion,
    mask_ignored_regions,
)

MASK_MAX_VALUE = 255
MIN_EDGE_WIDTH = 1
MAX_EDGE_WIDTH = 15
DEFAULT_LOW_THRESHOLD = 50
DEFAULT_HIGH_THRESHOLD = 150


class _ImageArray(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    def tobytes(self) -> bytes: ...


class _Cv2Module(Protocol):
    COLOR_RGB2GRAY: int

    def cvtColor(self, image: _ImageArray, code: int) -> _ImageArray: ...

    def GaussianBlur(
        self,
        image: _ImageArray,
        kernel_size: tuple[int, int],
        sigma_x: float,
    ) -> _ImageArray: ...

    def Canny(
        self,
        image: _ImageArray,
        low_threshold: int,
        high_threshold: int,
        *,
        apertureSize: int,
        L2gradient: bool,
    ) -> _ImageArray: ...


class _NumpyModule(Protocol):
    uint8: object

    def asarray(self, source: object, *, dtype: object) -> _ImageArray: ...


class CannyError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class CannyResult:
    mask: Image.Image
    overlay: Image.Image
    edge_pixels: int
    total_pixels: int
    edge_ratio: float


class CannyPredictor:
    def __init__(
        self,
        low_threshold: int = DEFAULT_LOW_THRESHOLD,
        high_threshold: int = DEFAULT_HIGH_THRESHOLD,
    ) -> None:
        _validate_thresholds(low_threshold, high_threshold)
        self._cv2, self._numpy = _load_dependencies()
        self.low_threshold: int = low_threshold
        self.high_threshold: int = high_threshold
        self._cv2: _Cv2Module
        self._numpy: _NumpyModule

    @property
    def device(self) -> str:
        return "cpu"

    def process(
        self,
        image: Image.Image,
        background_scale: float = 0.24,
        edge_width: int = 1,
        ignored_regions: tuple[HudIgnoreRegion, ...] = (),
    ) -> CannyResult:
        _validate_background_scale(background_scale)
        _validate_edge_width(edge_width)
        source = image.convert("RGB")
        array = self._numpy.asarray(source, dtype=self._numpy.uint8)
        grayscale = self._cv2.cvtColor(array, self._cv2.COLOR_RGB2GRAY)
        blurred = self._cv2.GaussianBlur(grayscale, (5, 5), 0)
        edges = self._cv2.Canny(
            blurred,
            self.low_threshold,
            self.high_threshold,
            apertureSize=3,
            L2gradient=True,
        )
        width, height = source.size
        mask = Image.frombytes("L", (width, height), edges.tobytes())
        mask = mask_ignored_regions(mask, ignored_regions)
        if edge_width != MIN_EDGE_WIDTH:
            kernel_size = edge_width if edge_width % 2 == 1 else edge_width + 1
            mask = mask.filter(ImageFilter.MaxFilter(size=kernel_size))
        dimmed = ImageEnhance.Brightness(source).enhance(background_scale)
        edge_layer = Image.new("RGB", source.size, HUD_EDGE_COLOR)
        overlay = Image.composite(edge_layer, dimmed, mask)
        edge_pixels = sum(value == MASK_MAX_VALUE for value in mask.tobytes())
        total_pixels = source.size[0] * source.size[1]
        return CannyResult(
            mask=mask,
            overlay=overlay,
            edge_pixels=edge_pixels,
            total_pixels=total_pixels,
            edge_ratio=edge_pixels / total_pixels,
        )


def _load_dependencies() -> tuple[_Cv2Module, _NumpyModule]:
    try:
        cv2 = importlib.import_module("cv2")
        numpy = importlib.import_module("numpy")
    except ModuleNotFoundError as error:
        message = (
            "Canny requires OpenCV and NumPy; install OpenCV for the deployment "
            "runtime, for example `uv run --with opencv-python-headless ...`."
        )
        raise CannyError(message) from error
    return cast("_Cv2Module", cast("object", cv2)), cast(
        "_NumpyModule",
        cast("object", numpy),
    )


def _validate_thresholds(low_threshold: int, high_threshold: int) -> None:
    if not 0 <= low_threshold < high_threshold <= MASK_MAX_VALUE:
        message = "Canny thresholds must satisfy 0 <= low < high <= 255"
        raise CannyError(message)


def _validate_background_scale(background_scale: float) -> None:
    if (
        not math.isfinite(background_scale)
        or background_scale <= 0.0
        or background_scale > 1.0
    ):
        message = "background scale must be a finite value in (0, 1]"
        raise CannyError(message)


def _validate_edge_width(edge_width: int) -> None:
    if edge_width < MIN_EDGE_WIDTH or edge_width > MAX_EDGE_WIDTH:
        message = "Canny edge width must be between 1 and 15"
        raise CannyError(message)
