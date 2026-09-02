from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PIL import Image, ImageDraw

from firesight_vision import deployment
from firesight_vision.canny import CannyError, CannyPredictor, CannyResult
from firesight_vision.deployment import (
    FireSightDeploymentError,
    FireSightRuntime,
    FireSightRuntimeConfig,
    load_fire360_osd_profiles,
)

if TYPE_CHECKING:
    from firesight_vision.hud_edges import HudIgnoreRegion


def test_load_fire360_osd_profiles_matches_source_aware_profile() -> None:
    profiles = load_fire360_osd_profiles(
        Path("data/fire360/video_profiles.json"),
    )

    assert profiles["IFSI Video 4 (2).mp4"] is True
    assert profiles["02814 (2).MTS"] is False


def test_runtime_resizes_and_applies_only_selected_osd_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[int, int], int]] = []

    class FakeCannyPredictor:
        @property
        def device(self) -> str:
            return "cpu"

        def __init__(self, low_threshold: int, high_threshold: int) -> None:
            assert (low_threshold, high_threshold) == (50, 150)

        def process(
            self,
            image: Image.Image,
            *,
            background_scale: float,
            edge_width: int,
            ignored_regions: tuple[HudIgnoreRegion, ...],
        ) -> CannyResult:
            _ = background_scale, edge_width
            calls.append((image.size, len(ignored_regions)))
            mask = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.line((0, 0, image.width - 1, image.height - 1), fill=255)
            return CannyResult(
                mask=mask,
                overlay=image.convert("RGB"),
                edge_pixels=image.width,
                total_pixels=image.width * image.height,
                edge_ratio=1.0 / image.height,
            )

    monkeypatch.setattr(deployment, "CannyPredictor", FakeCannyPredictor)
    runtime = FireSightRuntime(
        FireSightRuntimeConfig(
            backend="canny",
            width=320,
            height=240,
            profile_path=Path("data/fire360/video_profiles.json"),
        ),
    )

    result = runtime.process_frame(
        Image.new("RGB", (640, 360), (32, 32, 32)),
        "IFSI Video 4 (2).mp4",
    )

    assert result.processed_size == (320, 240)
    assert result.ignored_region_count == 2
    assert calls == [((320, 240), 2)]


def test_runtime_leaves_unprofiled_source_content_unmasked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    class FakeCannyPredictor:
        @property
        def device(self) -> str:
            return "cpu"

        def __init__(self, low_threshold: int, high_threshold: int) -> None:
            _ = low_threshold, high_threshold

        def process(
            self,
            image: Image.Image,
            *,
            background_scale: float,
            edge_width: int,
            ignored_regions: tuple[HudIgnoreRegion, ...],
        ) -> CannyResult:
            _ = background_scale, edge_width
            calls.append(len(ignored_regions))
            mask = Image.new("L", image.size, 0)
            return CannyResult(
                mask=mask,
                overlay=image.convert("RGB"),
                edge_pixels=0,
                total_pixels=image.width * image.height,
                edge_ratio=0.0,
            )

    monkeypatch.setattr(deployment, "CannyPredictor", FakeCannyPredictor)
    runtime = FireSightRuntime(
        FireSightRuntimeConfig(
            backend="canny",
            profile_path=Path("data/fire360/video_profiles.json"),
        ),
    )

    _ = runtime.process_frame(Image.new("RGB", (320, 240)), "02814 (2).MTS")

    assert calls == [0]


def test_teed_runtime_requires_checkpoint_path() -> None:
    with pytest.raises(FireSightDeploymentError, match="checkpoint"):
        _ = FireSightRuntime(FireSightRuntimeConfig(backend="teed"))


def test_canny_predictor_rejects_invalid_thresholds() -> None:
    with pytest.raises(CannyError, match="thresholds"):
        _ = CannyPredictor(low_threshold=150, high_threshold=50)
