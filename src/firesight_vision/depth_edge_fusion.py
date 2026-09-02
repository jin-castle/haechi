from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

if TYPE_CHECKING:
    from numpy.typing import NDArray


NEAR_COLOR: Final = (255, 64, 48)
MID_COLOR: Final = (255, 210, 40)
FAR_COLOR: Final = (40, 220, 255)
UNKNOWN_COLOR: Final = (110, 110, 110)
BACKGROUND_CODE: Final = 0
UNKNOWN_CODE: Final = 64
FAR_CODE: Final = 128
MID_CODE: Final = 192
NEAR_CODE: Final = 255


class DepthEdgeFusionError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class DepthEdgeFusionConfig:
    min_depth_m: float = 0.20
    near_max_m: float = 0.50
    mid_max_m: float = 1.00
    far_max_m: float = 1.50
    background_scale: float = 0.24


@dataclass(frozen=True, slots=True)
class DepthEdgeFusionResult:
    overlay: Image.Image
    band_mask: Image.Image
    edge_pixels: int
    model_valid_edge_pixels: int
    banded_edge_pixels: int
    edge_depth_model_valid_ratio: float
    edge_depth_banded_ratio: float
    near_edge_ratio: float
    mid_edge_ratio: float
    far_edge_ratio: float
    out_of_band_edge_ratio: float
    invalid_depth_edge_ratio: float
    unknown_edge_ratio: float


def build_depth_aware_edge_overlay(
    source_image: Image.Image,
    edge_mask: Image.Image,
    depth_m: NDArray[np.float32],
    *,
    config: DepthEdgeFusionConfig | None = None,
) -> DepthEdgeFusionResult:
    resolved = DepthEdgeFusionConfig() if config is None else config
    _validate_config(resolved)
    source = source_image.convert("RGB")
    mask = edge_mask.convert("L")
    width, height = source.size
    if mask.size != source.size:
        message = "edge mask size must match source image"
        raise DepthEdgeFusionError(message)
    if depth_m.shape != (height, width):
        message = "depth map shape must match source image"
        raise DepthEdgeFusionError(message)

    edge = np.asarray(mask, dtype=np.uint8) > 0
    model_valid_depth = np.isfinite(depth_m) & (depth_m > 0.0)
    banded_depth = (
        model_valid_depth
        & (depth_m >= resolved.min_depth_m)
        & (depth_m <= resolved.far_max_m)
    )
    near = edge & banded_depth & (depth_m <= resolved.near_max_m)
    mid = (
        edge
        & banded_depth
        & (depth_m > resolved.near_max_m)
        & (depth_m <= resolved.mid_max_m)
    )
    far = edge & banded_depth & (depth_m > resolved.mid_max_m)
    out_of_band = edge & model_valid_depth & ~banded_depth
    invalid_depth = edge & ~model_valid_depth
    unknown = out_of_band | invalid_depth

    band_codes = np.full((height, width), BACKGROUND_CODE, dtype=np.uint8)
    band_codes[unknown] = UNKNOWN_CODE
    band_codes[far] = FAR_CODE
    band_codes[mid] = MID_CODE
    band_codes[near] = NEAR_CODE

    overlay = ImageEnhance.Brightness(source).enhance(resolved.background_scale)
    overlay = _composite_band(overlay, unknown, UNKNOWN_COLOR, width=1)
    overlay = _composite_band(overlay, far, FAR_COLOR, width=1)
    overlay = _composite_band(overlay, mid, MID_COLOR, width=3)
    overlay = _composite_band(overlay, near, NEAR_COLOR, width=5)

    edge_pixels = int(np.count_nonzero(edge))
    near_pixels = int(np.count_nonzero(near))
    mid_pixels = int(np.count_nonzero(mid))
    far_pixels = int(np.count_nonzero(far))
    out_of_band_pixels = int(np.count_nonzero(out_of_band))
    invalid_depth_pixels = int(np.count_nonzero(invalid_depth))
    unknown_pixels = int(np.count_nonzero(unknown))
    model_valid_edge_pixels = int(np.count_nonzero(edge & model_valid_depth))
    banded_edge_pixels = near_pixels + mid_pixels + far_pixels
    denominator = edge_pixels if edge_pixels > 0 else 1
    return DepthEdgeFusionResult(
        overlay=overlay,
        band_mask=Image.fromarray(band_codes, mode="L"),
        edge_pixels=edge_pixels,
        model_valid_edge_pixels=model_valid_edge_pixels,
        banded_edge_pixels=banded_edge_pixels,
        edge_depth_model_valid_ratio=model_valid_edge_pixels / denominator,
        edge_depth_banded_ratio=banded_edge_pixels / denominator,
        near_edge_ratio=near_pixels / denominator,
        mid_edge_ratio=mid_pixels / denominator,
        far_edge_ratio=far_pixels / denominator,
        out_of_band_edge_ratio=out_of_band_pixels / denominator,
        invalid_depth_edge_ratio=invalid_depth_pixels / denominator,
        unknown_edge_ratio=unknown_pixels / denominator,
    )


def _composite_band(
    background: Image.Image,
    band: NDArray[np.bool_],
    color: tuple[int, int, int],
    *,
    width: int,
) -> Image.Image:
    mask = Image.fromarray((band.astype(np.uint8) * 255), mode="L")
    if width > 1:
        mask = mask.filter(ImageFilter.MaxFilter(size=width))
    layer = Image.new("RGB", background.size, color)
    return Image.composite(layer, background, mask)


def _validate_config(config: DepthEdgeFusionConfig) -> None:
    distances = (
        config.min_depth_m,
        config.near_max_m,
        config.mid_max_m,
        config.far_max_m,
    )
    if not all(math.isfinite(value) for value in distances):
        message = "distance bands must be finite"
        raise DepthEdgeFusionError(message)
    if not (
        0.0
        < config.min_depth_m
        < config.near_max_m
        < config.mid_max_m
        < config.far_max_m
    ):
        message = "distance bands must be strictly increasing"
        raise DepthEdgeFusionError(message)
    if (
        not math.isfinite(config.background_scale)
        or config.background_scale <= 0.0
        or config.background_scale > 1.0
    ):
        message = "background scale must be in (0, 1]"
        raise DepthEdgeFusionError(message)
