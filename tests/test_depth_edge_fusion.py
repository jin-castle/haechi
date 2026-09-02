import numpy as np
import pytest
from PIL import Image

from firesight_vision.depth_edge_fusion import (
    FAR_CODE,
    FAR_COLOR,
    MID_CODE,
    MID_COLOR,
    NEAR_CODE,
    NEAR_COLOR,
    UNKNOWN_CODE,
    DepthEdgeFusionError,
    build_depth_aware_edge_overlay,
)


def test_depth_aware_overlay_assigns_distance_bands() -> None:
    source = Image.new("RGB", (8, 4), (180, 180, 180))
    edge_mask_array = np.zeros((4, 8), dtype=np.uint8)
    edge_mask_array[2, (0, 2, 4, 6)] = 255
    edge_mask = Image.fromarray(edge_mask_array, mode="L")
    depth_m = np.full((4, 8), 2.0, dtype=np.float32)
    depth_m[2, 0] = 0.3
    depth_m[2, 2] = 0.7
    depth_m[2, 4] = 1.2

    result = build_depth_aware_edge_overlay(source, edge_mask, depth_m)

    band_codes = np.asarray(result.band_mask)
    assert band_codes[2, 0] == NEAR_CODE
    assert band_codes[2, 2] == MID_CODE
    assert band_codes[2, 4] == FAR_CODE
    assert band_codes[2, 6] == UNKNOWN_CODE
    assert result.edge_pixels == 4
    assert result.model_valid_edge_pixels == 4
    assert result.banded_edge_pixels == 3
    assert result.edge_depth_model_valid_ratio == pytest.approx(1.0)
    assert result.edge_depth_banded_ratio == pytest.approx(0.75)
    assert result.out_of_band_edge_ratio == pytest.approx(0.25)
    assert result.invalid_depth_edge_ratio == pytest.approx(0.0)
    assert result.unknown_edge_ratio == pytest.approx(0.25)
    assert result.overlay.size == source.size


def test_distance_band_colors_match_the_hud_legend() -> None:
    assert NEAR_COLOR == (255, 64, 48)
    assert MID_COLOR == (255, 210, 40)
    assert FAR_COLOR == (40, 220, 255)


def test_invalid_depth_is_distinct_from_out_of_band_depth() -> None:
    source = Image.new("RGB", (2, 1), (180, 180, 180))
    edge_mask = Image.new("L", source.size, 255)
    depth_m = np.array([[0.0, 2.0]], dtype=np.float32)

    result = build_depth_aware_edge_overlay(source, edge_mask, depth_m)

    assert result.edge_depth_model_valid_ratio == pytest.approx(0.5)
    assert result.edge_depth_banded_ratio == pytest.approx(0.0)
    assert result.invalid_depth_edge_ratio == pytest.approx(0.5)
    assert result.out_of_band_edge_ratio == pytest.approx(0.5)


def test_depth_aware_overlay_rejects_misaligned_depth() -> None:
    source = Image.new("RGB", (8, 4), (0, 0, 0))
    edge_mask = Image.new("L", source.size, 255)
    depth_m = np.ones((3, 8), dtype=np.float32)

    with pytest.raises(DepthEdgeFusionError, match="depth map shape"):
        _ = build_depth_aware_edge_overlay(source, edge_mask, depth_m)
