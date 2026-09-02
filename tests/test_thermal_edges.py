from PIL import Image, ImageDraw

from firesight_vision.thermal_edges import (
    ThermalHudConfig,
    ThermalMaskRegion,
    build_thermal_hud_overlay,
)


def test_thermal_hud_recovers_low_contrast_structure_edges() -> None:
    source = _thermal_fixture(structure_delta=18)

    result = build_thermal_hud_overlay(
        source,
        config=ThermalHudConfig(edge_threshold=24, hotspot_threshold=215),
    )

    assert result.overlay.mode == "RGB"
    assert result.edge_pixels > 400
    assert 0.01 < result.edge_ratio < 0.45
    assert _green_pixel_count(result.overlay) > 250


def test_thermal_hotspot_uses_intensity_outline_not_rgb_fire_threshold() -> None:
    source = _thermal_fixture(structure_delta=22)

    result = build_thermal_hud_overlay(
        source,
        config=ThermalHudConfig(edge_threshold=26, hotspot_threshold=205),
    )

    assert result.hotspot_pixels > 80
    assert result.hotspot_ratio < 0.08
    assert _red_pixel_count(result.overlay) > 60


def test_thermal_mask_region_excludes_sensor_osd_from_edge_and_hotspot_masks() -> None:
    source = _thermal_fixture(structure_delta=22)

    result = build_thermal_hud_overlay(
        source,
        config=ThermalHudConfig(
            edge_threshold=26,
            hotspot_threshold=205,
            ignored_regions=(
                ThermalMaskRegion(left=0.60, top=0.0, right=1.0, bottom=1.0),
            ),
        ),
    )

    ignored_box = (132, 0, 220, 140)
    assert result.edge_mask.crop(ignored_box).getbbox() is None
    assert result.hotspot_mask.crop(ignored_box).getbbox() is None


def _thermal_fixture(*, structure_delta: int) -> Image.Image:
    image = Image.new("L", (220, 140), 88)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 200, 122), outline=88 + structure_delta, width=3)
    draw.rectangle((86, 58, 128, 122), outline=92 + structure_delta, width=4)
    draw.line((30, 96, 92, 70, 154, 100, 194, 64), fill=90 + structure_delta, width=3)
    draw.ellipse((150, 42, 190, 102), fill=226)
    draw.polygon(((168, 32), (190, 82), (162, 112), (154, 70)), fill=248)
    return image


def _green_pixel_count(image: Image.Image) -> int:
    values = image.convert("RGB").tobytes()
    return sum(
        1
        for red, green, blue in _rgb_triplets(values)
        if red < 90 and green > 190 and blue < 120
    )


def _red_pixel_count(image: Image.Image) -> int:
    values = image.convert("RGB").tobytes()
    return sum(
        1
        for red, green, blue in _rgb_triplets(values)
        if red > 180 and green < 90 and blue < 90
    )


def _rgb_triplets(values: bytes) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (values[index], values[index + 1], values[index + 2])
        for index in range(0, len(values), 3)
    )
