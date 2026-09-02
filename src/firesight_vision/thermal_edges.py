from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps

MASK_MAX_VALUE: Final = 255
HUD_EDGE_COLOR: Final = (48, 255, 64)
HUD_HOTSPOT_COLOR: Final = (255, 32, 32)
MIN_EDGE_WIDTH: Final = 1
MAX_EDGE_WIDTH: Final = 15


@dataclass(frozen=True, slots=True)
class ThermalMaskRegion:
    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True, slots=True)
class ThermalHudConfig:
    edge_threshold: int = 46
    hotspot_threshold: int = 205
    background_scale: float = 0.30
    edge_width: int = 5
    hotspot_outline_width: int = 5
    unsharp_percent: int = 240
    ignored_regions: tuple[ThermalMaskRegion, ...] = ()


@dataclass(frozen=True, slots=True)
class ThermalHudResult:
    overlay: Image.Image
    edge_mask: Image.Image
    hotspot_mask: Image.Image
    edge_pixels: int
    hotspot_pixels: int
    total_pixels: int
    edge_ratio: float
    hotspot_ratio: float


@dataclass(frozen=True, slots=True)
class ThermalHudError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def build_thermal_hud_overlay(
    image: Image.Image,
    *,
    config: ThermalHudConfig | None = None,
) -> ThermalHudResult:
    resolved_config = ThermalHudConfig() if config is None else config
    _validate_config(resolved_config)

    thermal = _normalized_thermal_grayscale(image)
    edge_mask = _exclude_regions(
        _thermal_edge_mask(thermal, resolved_config),
        resolved_config.ignored_regions,
    )
    hotspot_mask = _exclude_regions(
        _hotspot_outline_mask(thermal, resolved_config),
        resolved_config.ignored_regions,
    )

    base = ImageEnhance.Brightness(thermal.convert("RGB")).enhance(
        resolved_config.background_scale,
    )
    edge_layer = Image.new("RGB", thermal.size, HUD_EDGE_COLOR)
    hotspot_layer = Image.new("RGB", thermal.size, HUD_HOTSPOT_COLOR)
    overlay = Image.composite(edge_layer, base, edge_mask)
    overlay = Image.composite(hotspot_layer, overlay, hotspot_mask)

    total_pixels = thermal.width * thermal.height
    edge_pixels = _active_pixel_count(edge_mask)
    hotspot_pixels = _active_pixel_count(hotspot_mask)
    return ThermalHudResult(
        overlay=overlay,
        edge_mask=edge_mask,
        hotspot_mask=hotspot_mask,
        edge_pixels=edge_pixels,
        hotspot_pixels=hotspot_pixels,
        total_pixels=total_pixels,
        edge_ratio=edge_pixels / total_pixels,
        hotspot_ratio=hotspot_pixels / total_pixels,
    )


def _normalized_thermal_grayscale(image: Image.Image) -> Image.Image:
    if image.mode in {"I;16", "I;16B", "I;16L", "I"}:
        extrema = cast("tuple[float, float] | tuple[int, int]", image.getextrema())
        min_value, max_value = _int_extrema(extrema)
        if max_value <= min_value:
            return Image.new("L", image.size, 0)
        scale = MASK_MAX_VALUE / (max_value - min_value)
        return image.point(lambda value: round((value - min_value) * scale)).convert(
            "L"
        )
    return ImageOps.autocontrast(image.convert("L"), cutoff=0)


def _thermal_edge_mask(
    thermal: Image.Image,
    config: ThermalHudConfig,
) -> Image.Image:
    enhanced = thermal.filter(
        ImageFilter.UnsharpMask(
            radius=2.0,
            percent=config.unsharp_percent,
            threshold=0,
        ),
    )
    edges = enhanced.filter(ImageFilter.FIND_EDGES)
    contrasted = ImageEnhance.Contrast(edges).enhance(2.6)
    blurred = contrasted.filter(ImageFilter.GaussianBlur(radius=0.55))
    mask = blurred.point(
        lambda value: MASK_MAX_VALUE if value >= config.edge_threshold else 0,
    )
    return _dilate_mask(mask, config.edge_width).filter(
        ImageFilter.MedianFilter(size=3)
    )


def _hotspot_outline_mask(
    thermal: Image.Image,
    config: ThermalHudConfig,
) -> Image.Image:
    hot_fill = thermal.filter(ImageFilter.MedianFilter(size=3)).point(
        lambda value: MASK_MAX_VALUE if value >= config.hotspot_threshold else 0,
    )
    expanded = _dilate_mask(hot_fill, config.hotspot_outline_width)
    eroded = hot_fill.filter(ImageFilter.MinFilter(size=3))
    outline = ImageChops.subtract(expanded, eroded)
    return outline.point(lambda value: MASK_MAX_VALUE if value > 0 else 0)


def _dilate_mask(mask: Image.Image, width: int) -> Image.Image:
    kernel_size = width if width % 2 == 1 else width + 1
    return mask.filter(ImageFilter.MaxFilter(size=kernel_size))


def _exclude_regions(
    mask: Image.Image,
    regions: tuple[ThermalMaskRegion, ...],
) -> Image.Image:
    if len(regions) == 0:
        return mask
    masked = mask.copy()
    draw = ImageDraw.Draw(masked)
    width, height = masked.size
    for region in regions:
        draw.rectangle(
            (
                round(region.left * width),
                round(region.top * height),
                round(region.right * width),
                round(region.bottom * height),
            ),
            fill=0,
        )
    return masked


def _active_pixel_count(mask: Image.Image) -> int:
    return sum(1 for value in mask.convert("L").tobytes() if value == MASK_MAX_VALUE)


def _validate_config(config: ThermalHudConfig) -> None:
    _validate_threshold(config.edge_threshold, "edge_threshold")
    _validate_threshold(config.hotspot_threshold, "hotspot_threshold")
    _validate_width(config.edge_width, "edge_width")
    _validate_width(config.hotspot_outline_width, "hotspot_outline_width")
    if config.background_scale <= 0.0 or config.background_scale > 1.0:
        raise ThermalHudError(message="background_scale must be in (0, 1]")
    if config.unsharp_percent < 0:
        raise ThermalHudError(message="unsharp_percent must be non-negative")
    for region in config.ignored_regions:
        _validate_region(region)


def _validate_threshold(value: int, name: str) -> None:
    if value < 0 or value > MASK_MAX_VALUE:
        raise ThermalHudError(message=f"{name} must be between 0 and 255")


def _validate_width(value: int, name: str) -> None:
    if value < MIN_EDGE_WIDTH or value > MAX_EDGE_WIDTH:
        raise ThermalHudError(message=f"{name} must be between 1 and 15")


def _validate_region(region: ThermalMaskRegion) -> None:
    if not (0.0 <= region.left < region.right <= 1.0):
        raise ThermalHudError(
            message="ignored region horizontal bounds must be in [0, 1]"
        )
    if not (0.0 <= region.top < region.bottom <= 1.0):
        raise ThermalHudError(
            message="ignored region vertical bounds must be in [0, 1]"
        )


def _int_extrema(extrema: tuple[float, float] | tuple[int, int]) -> tuple[int, int]:
    min_value, max_value = extrema
    return int(min_value), int(max_value)
