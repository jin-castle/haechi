import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from firesight_vision.hud_edges import (
    HudEdgeError,
    HudEdgeProfile,
    HudIgnoreRegion,
    build_hud_edge_overlay,
)


def test_hud_edge_overlay_marks_structural_pixels() -> None:
    with Image.open("tests/fixtures/images/sample_room.jpg") as source_image:
        result = build_hud_edge_overlay(
            source_image,
            threshold=36,
            background_scale=0.24,
        )

    assert result.edge_pixels > 0
    assert result.total_pixels == result.overlay.size[0] * result.overlay.size[1]
    assert 0.0 < result.edge_ratio < 1.0
    assert result.overlay.mode == "RGB"
    assert result.mask.mode == "L"


def test_hud_edge_overlay_width_thickens_mask() -> None:
    with Image.open("tests/fixtures/images/sample_room.jpg") as source_image:
        thin = build_hud_edge_overlay(
            source_image,
            threshold=36,
            background_scale=0.24,
            edge_width=1,
        )
        thick = build_hud_edge_overlay(
            source_image,
            threshold=36,
            background_scale=0.24,
            edge_width=5,
        )

    assert thick.edge_pixels > thin.edge_pixels
    assert thick.edge_ratio > thin.edge_ratio


def test_hud_edge_overlay_excludes_fixed_osd_regions() -> None:
    source_image = Image.new("RGB", (64, 64), (0, 0, 0))
    draw = ImageDraw.Draw(source_image)
    draw.rectangle((4, 8, 12, 56), fill=(255, 255, 255))
    draw.rectangle((28, 8, 36, 56), fill=(255, 255, 255))
    draw.rectangle((50, 8, 58, 56), fill=(255, 255, 255))

    result = build_hud_edge_overlay(
        source_image,
        threshold=36,
        background_scale=0.24,
        ignored_regions=(
            HudIgnoreRegion(left=0.0, top=0.0, right=0.25, bottom=1.0),
            HudIgnoreRegion(left=0.75, top=0.0, right=1.0, bottom=1.0),
        ),
    )

    assert result.edge_pixels > 0
    assert result.mask.crop((0, 0, 16, 64)).getbbox() is None
    assert result.mask.crop((48, 0, 64, 64)).getbbox() is None


def test_dense_smoke_profile_recovers_low_contrast_structure() -> None:
    source_image, structure_band = _build_dense_smoke_structure_fixture()

    standard = build_hud_edge_overlay(
        source_image,
        threshold=36,
        background_scale=0.24,
        edge_width=1,
    )
    dense_smoke = build_hud_edge_overlay(
        source_image,
        threshold=36,
        background_scale=0.24,
        edge_width=3,
        profile=HudEdgeProfile.DENSE_SMOKE,
    )

    assert _structure_hits(standard.mask, structure_band) == 0
    assert _structure_hits(dense_smoke.mask, structure_band) >= 400
    assert dense_smoke.edge_ratio < 0.30


def test_fire_line_profile_marks_fire_red_and_thickens_structure() -> None:
    source_image, structure_band, fire_band, fire_core = _build_fire_line_fixture()

    dense_smoke = build_hud_edge_overlay(
        source_image,
        threshold=144,
        background_scale=0.24,
        edge_width=3,
        profile=HudEdgeProfile.DENSE_SMOKE,
    )
    fire_line = build_hud_edge_overlay(
        source_image,
        threshold=128,
        background_scale=0.24,
        edge_width=5,
        profile=HudEdgeProfile.FIRE_LINE,
    )

    assert _structure_hits(fire_line.mask, structure_band) > _structure_hits(
        dense_smoke.mask,
        structure_band,
    )
    assert _red_overlay_hits(fire_line.overlay, fire_band) >= 120
    assert _red_overlay_hits(fire_line.overlay, fire_core) < _mask_area(fire_core) // 2
    assert fire_line.edge_ratio < 0.36


def test_hud_edge_overlay_rejects_invalid_threshold() -> None:
    with (
        Image.open("tests/fixtures/images/sample_room.jpg") as source_image,
        pytest.raises(HudEdgeError, match="threshold"),
    ):
        _ = build_hud_edge_overlay(
            source_image,
            threshold=300,
            background_scale=0.24,
        )


def test_hud_edge_overlay_rejects_invalid_edge_width() -> None:
    with (
        Image.open("tests/fixtures/images/sample_room.jpg") as source_image,
        pytest.raises(HudEdgeError, match="edge width"),
    ):
        _ = build_hud_edge_overlay(
            source_image,
            threshold=36,
            background_scale=0.24,
            edge_width=0,
        )


def test_run_hud_edge_baseline_writes_overlay_and_summary(tmp_path: Path) -> None:
    out_dir = tmp_path / "hud-edge"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_hud_edge_baseline.py",
            "--input",
            "tests/fixtures/images/sample_room.jpg",
            "--out",
            str(out_dir),
            "--edge-width",
            "5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    summary = (out_dir / "hud_edge_summary.json").read_text("utf-8")
    assert '"protocol": "pillow_find_edges_hud_overlay_v1"' in summary
    assert '"input_file": "tests/fixtures/images/sample_room.jpg"' in summary
    assert '"edge_width": 5' in summary
    assert (out_dir / "sample_room_hud_edges.png").exists()
    assert (out_dir / "sample_room_edge_mask.png").exists()


def test_run_hud_edge_baseline_dense_smoke_profile_defaults_to_thick_edges(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "hud-edge"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_hud_edge_baseline.py",
            "--input",
            "tests/fixtures/images/sample_room.jpg",
            "--out",
            str(out_dir),
            "--profile",
            "dense-smoke",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    summary = (out_dir / "hud_edge_summary.json").read_text("utf-8")
    assert '"profile": "dense-smoke"' in summary
    assert '"edge_width": 3' in summary


def test_run_hud_edge_baseline_fire_line_profile_defaults_to_clear_lines(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "hud-edge"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_hud_edge_baseline.py",
            "--input",
            "tests/fixtures/images/sample_room.jpg",
            "--out",
            str(out_dir),
            "--profile",
            "dense-smoke-fire-line",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    summary = (out_dir / "hud_edge_summary.json").read_text("utf-8")
    assert '"profile": "dense-smoke-fire-line"' in summary
    assert '"edge_width": 5' in summary


def _build_dense_smoke_structure_fixture() -> tuple[Image.Image, Image.Image]:
    width = 160
    height = 120
    background = 118
    source_image = Image.new(
        "RGB",
        (width, height),
        (background, background, background),
    )
    draw = ImageDraw.Draw(source_image)
    structure_color = (background + 8, background + 8, background + 8)
    draw.rectangle((34, 20, 126, 106), outline=structure_color, width=2)
    draw.rectangle((66, 48, 95, 106), outline=structure_color, width=2)
    draw.line((18, 40, 142, 34), fill=structure_color, width=2)
    blurred = source_image.filter(ImageFilter.GaussianBlur(radius=3.2))
    smoke_limited = ImageEnhance.Contrast(blurred).enhance(0.35)

    structure_band = Image.new("L", (width, height), 0)
    band_draw = ImageDraw.Draw(structure_band)
    band_draw.rectangle((31, 17, 129, 109), outline=255, width=7)
    band_draw.rectangle((63, 45, 98, 109), outline=255, width=7)
    band_draw.line((15, 40, 145, 34), fill=255, width=7)
    return smoke_limited, structure_band


def _build_fire_line_fixture() -> tuple[
    Image.Image,
    Image.Image,
    Image.Image,
    Image.Image,
]:
    source_image, structure_band = _build_dense_smoke_structure_fixture()
    draw = ImageDraw.Draw(source_image)
    draw.ellipse((100, 32, 130, 72), fill=(218, 132, 46))
    draw.polygon([(111, 30), (122, 54), (108, 66), (102, 48)], fill=(255, 196, 82))

    fire_band = Image.new("L", source_image.size, 0)
    band_draw = ImageDraw.Draw(fire_band)
    band_draw.ellipse((96, 28, 134, 76), fill=255)
    fire_core = Image.new("L", source_image.size, 0)
    core_draw = ImageDraw.Draw(fire_core)
    core_draw.ellipse((107, 43, 123, 61), fill=255)
    return source_image, structure_band, fire_band, fire_core


def _structure_hits(mask: Image.Image, structure_band: Image.Image) -> int:
    return sum(
        1
        for mask_value, band_value in zip(
            mask.tobytes(),
            structure_band.tobytes(),
            strict=True,
        )
        if mask_value > 0 and band_value > 0
    )


def _red_overlay_hits(overlay: Image.Image, fire_band: Image.Image) -> int:
    rgb_overlay = overlay.convert("RGB")
    overlay_values = rgb_overlay.tobytes()
    band_values = fire_band.tobytes()
    return sum(
        1
        for pixel_index, band_value in enumerate(band_values)
        for red, green, blue in (
            (
                overlay_values[pixel_index * 3],
                overlay_values[(pixel_index * 3) + 1],
                overlay_values[(pixel_index * 3) + 2],
            ),
        )
        if band_value > 0 and red > 180 and green < 80 and blue < 80
    )


def _mask_area(mask: Image.Image) -> int:
    return sum(1 for value in mask.tobytes() if value > 0)
