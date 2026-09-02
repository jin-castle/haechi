from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final, TypedDict

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps

if TYPE_CHECKING:
    from pathlib import Path


class HudEdgeImagePayload(TypedDict):
    input_file: str
    overlay_file: str
    mask_file: str
    edge_pixels: int
    total_pixels: int
    edge_ratio: float
    edge_width: int
    profile: str


class HudEdgeRunPayload(TypedDict):
    protocol: str
    claim: str
    images: list[HudEdgeImagePayload]


class HudEdgeProfile(StrEnum):
    STANDARD = "standard"
    DENSE_SMOKE = "dense-smoke"
    FIRE_LINE = "dense-smoke-fire-line"


@dataclass(frozen=True, slots=True)
class HudIgnoreRegion:
    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True, slots=True)
class HudEdgeRequest:
    input_path: Path
    out_dir: Path
    threshold: int
    background_scale: float
    edge_width: int = 1
    profile: HudEdgeProfile = HudEdgeProfile.STANDARD
    ignored_regions: tuple[HudIgnoreRegion, ...] = ()


@dataclass(frozen=True, slots=True)
class HudEdgeResult:
    overlay: Image.Image
    mask: Image.Image
    edge_pixels: int
    total_pixels: int
    edge_ratio: float


@dataclass(frozen=True, slots=True)
class HudEdgeError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


PROTOCOL_NAME = "pillow_find_edges_hud_overlay_v1"
CLAIM_TEXT = (
    "Classical edge baseline for HUD-style visibility tests; not a trained "
    "firefighter perception model."
)
HUD_EDGE_COLOR: Final = (48, 255, 64)
HUD_FIRE_COLOR: Final = (255, 32, 32)
MASK_MAX_VALUE: Final = 255
MIN_EDGE_WIDTH: Final = 1
MAX_EDGE_WIDTH: Final = 15
FIRE_HUE_MIN: Final = 5
FIRE_HUE_MAX: Final = 42
FIRE_SATURATION_MIN: Final = 35
FIRE_VALUE_MIN: Final = 115
FIRE_RED_MIN: Final = 150
FIRE_GREEN_MIN: Final = 90
FIRE_BLUE_MAX: Final = 170
FIRE_MASK_MEDIAN: Final = 3
FIRE_HOTSPOT_OUTLINE_WIDTH: Final = 3
FIRE_LOCAL_EDGE_MIN: Final = 60
FIRE360_OSD_IGNORED_REGIONS: Final[tuple[HudIgnoreRegion, ...]] = (
    HudIgnoreRegion(left=0.13, top=0.72, right=0.27, bottom=1.0),
    HudIgnoreRegion(left=0.75, top=0.0, right=0.90, bottom=0.90),
)


def run_hud_edge_baseline(requests: tuple[HudEdgeRequest, ...]) -> HudEdgeRunPayload:
    images = [_write_hud_edge_result(request) for request in requests]
    payload = HudEdgeRunPayload(protocol=PROTOCOL_NAME, claim=CLAIM_TEXT, images=images)
    if len(requests) > 0:
        summary_path = requests[0].out_dir / "hud_edge_summary.json"
        _ = summary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def build_hud_edge_overlay(
    image: Image.Image,
    threshold: int,
    background_scale: float,
    edge_width: int = 1,
    profile: HudEdgeProfile = HudEdgeProfile.STANDARD,
    ignored_regions: tuple[HudIgnoreRegion, ...] = (),
) -> HudEdgeResult:
    _validate_threshold(threshold)
    _validate_edge_width(edge_width)
    if background_scale <= 0.0 or background_scale > 1.0:
        raise HudEdgeError(message="background scale must be in (0, 1]")
    _validate_ignored_regions(ignored_regions)

    source = image.convert("RGB")
    mask = mask_ignored_regions(
        _edge_mask(source, threshold, edge_width, profile),
        ignored_regions,
    )
    dimmed = ImageEnhance.Brightness(source).enhance(background_scale)
    edge_layer = Image.new("RGB", source.size, HUD_EDGE_COLOR)
    overlay = Image.composite(edge_layer, dimmed, mask)
    if profile is HudEdgeProfile.FIRE_LINE:
        fire_layer = Image.new("RGB", source.size, HUD_FIRE_COLOR)
        overlay = Image.composite(
            fire_layer,
            overlay,
            mask_ignored_regions(_fire_mask(source), ignored_regions),
        )
    edge_pixels = sum(1 for value in mask.tobytes() if value == MASK_MAX_VALUE)
    total_pixels = source.size[0] * source.size[1]
    return HudEdgeResult(
        overlay=overlay,
        mask=mask,
        edge_pixels=edge_pixels,
        total_pixels=total_pixels,
        edge_ratio=edge_pixels / total_pixels,
    )


def _write_hud_edge_result(request: HudEdgeRequest) -> HudEdgeImagePayload:
    request.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(request.input_path) as source_image:
            result = build_hud_edge_overlay(
                source_image,
                request.threshold,
                request.background_scale,
                request.edge_width,
                request.profile,
                request.ignored_regions,
            )
    except FileNotFoundError as error:
        message = f"missing input image: {request.input_path}"
        raise HudEdgeError(message=message) from error

    stem = request.input_path.stem
    overlay_name = f"{stem}_hud_edges.png"
    mask_name = f"{stem}_edge_mask.png"
    result.overlay.save(request.out_dir / overlay_name)
    result.mask.save(request.out_dir / mask_name)
    return HudEdgeImagePayload(
        input_file=request.input_path.as_posix(),
        overlay_file=overlay_name,
        mask_file=mask_name,
        edge_pixels=result.edge_pixels,
        total_pixels=result.total_pixels,
        edge_ratio=result.edge_ratio,
        edge_width=request.edge_width,
        profile=request.profile.value,
    )


def _edge_mask(
    image: Image.Image,
    threshold: int,
    edge_width: int,
    profile: HudEdgeProfile,
) -> Image.Image:
    grayscale = _visibility_preprocessed_grayscale(image, profile)
    edges = grayscale.filter(ImageFilter.FIND_EDGES)
    enhanced = ImageEnhance.Contrast(edges).enhance(2.8)
    blurred = enhanced.filter(ImageFilter.GaussianBlur(radius=0.45))
    mask = blurred.point(lambda value: MASK_MAX_VALUE if value >= threshold else 0)
    if edge_width != MIN_EDGE_WIDTH:
        kernel_size = edge_width if edge_width % 2 == 1 else edge_width + 1
        mask = mask.filter(ImageFilter.MaxFilter(size=kernel_size))
    match profile:
        case HudEdgeProfile.STANDARD:
            return mask
        case HudEdgeProfile.DENSE_SMOKE:
            return mask.filter(ImageFilter.MedianFilter(size=3))
        case HudEdgeProfile.FIRE_LINE:
            return mask.filter(ImageFilter.MedianFilter(size=3))


def mask_ignored_regions(
    mask: Image.Image,
    regions: tuple[HudIgnoreRegion, ...],
) -> Image.Image:
    _validate_ignored_regions(regions)
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


def _visibility_preprocessed_grayscale(
    image: Image.Image,
    profile: HudEdgeProfile,
) -> Image.Image:
    grayscale = image.convert("L")
    match profile:
        case HudEdgeProfile.STANDARD:
            return grayscale
        case HudEdgeProfile.DENSE_SMOKE:
            percent = 180
        case HudEdgeProfile.FIRE_LINE:
            percent = 230
    contrast_stretched = ImageOps.autocontrast(grayscale, cutoff=0)
    return contrast_stretched.filter(
        ImageFilter.UnsharpMask(radius=2.0, percent=percent, threshold=0),
    )


def _fire_mask(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    hsv = rgb.convert("HSV")
    red, green, blue = rgb.split()
    hue, saturation, value = hsv.split()
    mask = _mask_and(
        hue.point(
            lambda pixel: (
                MASK_MAX_VALUE if FIRE_HUE_MIN <= pixel <= FIRE_HUE_MAX else 0
            ),
        ),
        saturation.point(
            lambda pixel: MASK_MAX_VALUE if pixel >= FIRE_SATURATION_MIN else 0,
        ),
        value.point(lambda pixel: MASK_MAX_VALUE if pixel >= FIRE_VALUE_MIN else 0),
        red.point(lambda pixel: MASK_MAX_VALUE if pixel >= FIRE_RED_MIN else 0),
        green.point(lambda pixel: MASK_MAX_VALUE if pixel >= FIRE_GREEN_MIN else 0),
        blue.point(lambda pixel: MASK_MAX_VALUE if pixel <= FIRE_BLUE_MAX else 0),
    )
    cleaned = mask.filter(ImageFilter.MedianFilter(size=FIRE_MASK_MEDIAN))
    local_edges = ImageOps.autocontrast(rgb.convert("L"), cutoff=0).filter(
        ImageFilter.FIND_EDGES,
    )
    edge_mask = (
        ImageEnhance.Contrast(local_edges)
        .enhance(2.0)
        .point(
            lambda pixel: MASK_MAX_VALUE if pixel >= FIRE_LOCAL_EDGE_MIN else 0,
        )
    )
    return ImageChops.multiply(cleaned, edge_mask).filter(
        ImageFilter.MaxFilter(size=FIRE_HOTSPOT_OUTLINE_WIDTH),
    )


def _mask_and(first_mask: Image.Image, *other_masks: Image.Image) -> Image.Image:
    combined = first_mask
    for other_mask in other_masks:
        combined = ImageChops.multiply(combined, other_mask)
    return combined


def _validate_threshold(threshold: int) -> None:
    if threshold < 0 or threshold > MASK_MAX_VALUE:
        raise HudEdgeError(message="threshold must be between 0 and 255")


def _validate_edge_width(edge_width: int) -> None:
    if edge_width < MIN_EDGE_WIDTH or edge_width > MAX_EDGE_WIDTH:
        raise HudEdgeError(message="edge width must be between 1 and 15")


def _validate_ignored_regions(regions: tuple[HudIgnoreRegion, ...]) -> None:
    for region in regions:
        if not (0.0 <= region.left < region.right <= 1.0):
            raise HudEdgeError(
                message="ignored region horizontal bounds must be in [0, 1]",
            )
        if not (0.0 <= region.top < region.bottom <= 1.0):
            raise HudEdgeError(
                message="ignored region vertical bounds must be in [0, 1]",
            )
