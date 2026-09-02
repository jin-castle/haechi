from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from PIL import Image

from firesight_vision.canny import CannyPredictor
from firesight_vision.hud_edges import (
    FIRE360_OSD_IGNORED_REGIONS,
    HudIgnoreRegion,
)
from firesight_vision.teed import TeedPredictor

DeploymentBackend = Literal["teed", "canny"]
PROFILE_PROTOCOL = "fire360_video_profile_v1"
MIN_RUNTIME_DIMENSION = 64
MAX_EDGE_WIDTH = 15
MAX_CANNY_THRESHOLD = 255


@dataclass(frozen=True, slots=True)
class FireSightDeploymentError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class FireSightRuntimeConfig:
    backend: DeploymentBackend = "teed"
    checkpoint_path: Path | None = None
    width: int = 320
    height: int = 240
    threshold: float = 0.75
    background_scale: float = 0.24
    edge_width: int = 1
    device: str = "auto"
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    profile_path: Path | None = None
    video_key: str | None = None


@dataclass(frozen=True, slots=True)
class FireSightFrameResult:
    overlay: Image.Image
    mask: Image.Image
    latency_ms: float
    edge_pixels: int
    total_pixels: int
    edge_ratio: float
    input_size: tuple[int, int]
    processed_size: tuple[int, int]
    backend: DeploymentBackend
    device: str
    ignored_region_count: int


@dataclass(frozen=True, slots=True)
class _EdgeOutput:
    overlay: Image.Image
    mask: Image.Image
    edge_pixels: int
    total_pixels: int
    edge_ratio: float


class _EdgeBackend(Protocol):
    @property
    def name(self) -> DeploymentBackend: ...

    @property
    def device(self) -> str: ...

    def process(
        self,
        image: Image.Image,
        *,
        background_scale: float,
        edge_width: int,
        ignored_regions: tuple[HudIgnoreRegion, ...],
    ) -> _EdgeOutput: ...


class _TeedBackend:
    def __init__(self, config: FireSightRuntimeConfig) -> None:
        if config.checkpoint_path is None:
            message = "TEED deployment requires --checkpoint"
            raise FireSightDeploymentError(message)
        self._predictor: TeedPredictor = TeedPredictor(
            config.checkpoint_path,
            device=config.device,
        )
        self._threshold: float = config.threshold

    @property
    def name(self) -> DeploymentBackend:
        return "teed"

    @property
    def device(self) -> str:
        return self._predictor.device

    def process(
        self,
        image: Image.Image,
        *,
        background_scale: float,
        edge_width: int,
        ignored_regions: tuple[HudIgnoreRegion, ...],
    ) -> _EdgeOutput:
        result = self._predictor.process(
            image,
            threshold=self._threshold,
            background_scale=background_scale,
            edge_width=edge_width,
            ignored_regions=ignored_regions,
        )
        return _EdgeOutput(
            overlay=result.overlay,
            mask=result.mask,
            edge_pixels=result.edge_pixels,
            total_pixels=result.total_pixels,
            edge_ratio=result.edge_ratio,
        )


class _CannyBackend:
    def __init__(self, config: FireSightRuntimeConfig) -> None:
        self._predictor: CannyPredictor = CannyPredictor(
            low_threshold=config.canny_low_threshold,
            high_threshold=config.canny_high_threshold,
        )

    @property
    def name(self) -> DeploymentBackend:
        return "canny"

    @property
    def device(self) -> str:
        return self._predictor.device

    def process(
        self,
        image: Image.Image,
        *,
        background_scale: float,
        edge_width: int,
        ignored_regions: tuple[HudIgnoreRegion, ...],
    ) -> _EdgeOutput:
        result = self._predictor.process(
            image,
            background_scale=background_scale,
            edge_width=edge_width,
            ignored_regions=ignored_regions,
        )
        return _EdgeOutput(
            overlay=result.overlay,
            mask=result.mask,
            edge_pixels=result.edge_pixels,
            total_pixels=result.total_pixels,
            edge_ratio=result.edge_ratio,
        )


class FireSightRuntime:
    def __init__(self, config: FireSightRuntimeConfig) -> None:
        _validate_config(config)
        self.config: FireSightRuntimeConfig = config
        self._profiles: dict[str, bool] = (
            load_fire360_osd_profiles(config.profile_path)
            if config.profile_path is not None
            else {}
        )
        self._backend: _EdgeBackend = _build_backend(config)

    @property
    def backend(self) -> DeploymentBackend:
        return self._backend.name

    @property
    def device(self) -> str:
        return self._backend.device

    @property
    def target_size(self) -> tuple[int, int]:
        return self.config.width, self.config.height

    def ignored_regions_for(
        self,
        video_key: str | None = None,
    ) -> tuple[HudIgnoreRegion, ...]:
        selected_key = video_key or self.config.video_key
        if selected_key is None:
            return ()
        candidate = Path(selected_key).name.casefold()
        for profile_key, ignores_osd in self._profiles.items():
            if Path(profile_key).name.casefold() == candidate and ignores_osd:
                return FIRE360_OSD_IGNORED_REGIONS
        return ()

    def process_frame(
        self,
        image: Image.Image,
        video_key: str | None = None,
    ) -> FireSightFrameResult:
        started_at = time.perf_counter()
        source = image.convert("RGB")
        input_size = source.size
        prepared = source.resize(self.target_size, Image.Resampling.LANCZOS)
        ignored_regions = self.ignored_regions_for(video_key)
        edge = self._backend.process(
            prepared,
            background_scale=self.config.background_scale,
            edge_width=self.config.edge_width,
            ignored_regions=ignored_regions,
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        return FireSightFrameResult(
            overlay=edge.overlay,
            mask=edge.mask,
            latency_ms=elapsed_ms,
            edge_pixels=edge.edge_pixels,
            total_pixels=edge.total_pixels,
            edge_ratio=edge.edge_ratio,
            input_size=input_size,
            processed_size=self.target_size,
            backend=self.backend,
            device=self.device,
            ignored_region_count=len(ignored_regions),
        )


def load_fire360_osd_profiles(path: Path) -> dict[str, bool]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        message = f"unable to read Fire360 OSD profile: {path}"
        raise FireSightDeploymentError(message) from error
    if not isinstance(raw, dict) or raw.get("protocol") != PROFILE_PROTOCOL:
        message = f"unsupported Fire360 OSD profile protocol: {path}"
        raise FireSightDeploymentError(message)
    profiles = raw.get("profiles")
    if not isinstance(profiles, list):
        message = f"Fire360 OSD profile has no profiles list: {path}"
        raise FireSightDeploymentError(message)
    result: dict[str, bool] = {}
    for entry in profiles:
        if not isinstance(entry, dict):
            message = f"Fire360 OSD profile contains an invalid entry: {path}"
            raise FireSightDeploymentError(message)
        input_file = entry.get("input_file")
        ignores_osd = entry.get("ignore_fire360_osd")
        if not isinstance(input_file, str) or not isinstance(ignores_osd, bool):
            message = f"Fire360 OSD profile entry is missing required fields: {path}"
            raise FireSightDeploymentError(message)
        result[input_file] = ignores_osd
    return result


def _build_backend(config: FireSightRuntimeConfig) -> _EdgeBackend:
    if config.backend == "teed":
        return _TeedBackend(config)
    return _CannyBackend(config)


def _validate_config(config: FireSightRuntimeConfig) -> None:
    if config.backend not in ("teed", "canny"):
        message = f"unsupported FireSight backend: {config.backend}"
        raise FireSightDeploymentError(message)
    if config.width < MIN_RUNTIME_DIMENSION or config.height < MIN_RUNTIME_DIMENSION:
        message = "runtime dimensions must be at least 64 pixels"
        raise FireSightDeploymentError(message)
    if not math.isfinite(config.threshold) or not 0.0 <= config.threshold <= 1.0:
        message = "runtime threshold must be a finite value between 0 and 1"
        raise FireSightDeploymentError(message)
    if (
        not math.isfinite(config.background_scale)
        or config.background_scale <= 0.0
        or config.background_scale > 1.0
    ):
        message = "runtime background scale must be a finite value in (0, 1]"
        raise FireSightDeploymentError(message)
    if config.edge_width < 1 or config.edge_width > MAX_EDGE_WIDTH:
        message = "runtime edge width must be between 1 and 15"
        raise FireSightDeploymentError(message)
    if config.profile_path is not None and not config.profile_path.is_file():
        message = f"missing Fire360 OSD profile: {config.profile_path}"
        raise FireSightDeploymentError(message)
    if config.backend == "teed" and config.checkpoint_path is None:
        message = "TEED deployment requires a checkpoint path"
        raise FireSightDeploymentError(message)
    if config.backend == "canny" and config.device not in ("auto", "cpu"):
        message = "Canny deployment supports only the CPU device"
        raise FireSightDeploymentError(message)
    if not 0 <= config.canny_low_threshold < config.canny_high_threshold <= (
        MAX_CANNY_THRESHOLD
    ):
        message = "Canny thresholds must satisfy 0 <= low < high <= 255"
        raise FireSightDeploymentError(message)
