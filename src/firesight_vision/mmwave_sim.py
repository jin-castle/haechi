from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from pathlib import Path


class ObstacleSector(StrEnum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class DistanceBand(StrEnum):
    NEAR = "near"
    MID = "mid"
    FAR = "far"


class ObstacleDetectionPayload(TypedDict):
    sector: str
    distance_band: str
    range_m: float
    angle_deg: float
    confidence: float


class ScenarioPayload(TypedDict):
    scenario: str
    detections: list[ObstacleDetectionPayload]


class SimulationPayload(TypedDict):
    protocol: str
    claim: str
    scenarios: list[ScenarioPayload]


class _TextDrawer(Protocol):
    def text(
        self,
        xy: tuple[int, int],
        text: str,
        *,
        fill: tuple[int, int, int],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class MmWaveSimulationConfig:
    range_min_m: float = 0.20
    range_max_m: float = 1.50
    range_bins: int = 64
    angle_min_deg: float = -60.0
    angle_max_deg: float = 60.0
    angle_bins: int = 73
    range_sigma_m: float = 0.08
    angle_sigma_deg: float = 6.5
    detection_threshold: float = 0.34


@dataclass(frozen=True, slots=True)
class SimulatedObstacle:
    label: str
    range_m: float
    angle_deg: float
    intensity: float = 1.0


@dataclass(frozen=True, slots=True)
class ObstacleDetection:
    sector: ObstacleSector
    distance_band: DistanceBand
    range_m: float
    angle_deg: float
    confidence: float


@dataclass(frozen=True, slots=True)
class MmWaveScenario:
    name: str
    obstacles: tuple[SimulatedObstacle, ...]


@dataclass(frozen=True, slots=True)
class MmWaveScenarioResult:
    scenario: MmWaveScenario
    heatmap: tuple[tuple[float, ...], ...]
    detections: tuple[ObstacleDetection, ...]


@dataclass(frozen=True, slots=True)
class MmWaveSimulationError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


PROTOCOL_NAME = "mmwave_front_obstacle_range_angle_sim_v1"
CLAIM_TEXT = (
    "Sensor-free mmWave-like range-angle simulation for front 1.5 m "
    "obstacle HUD feasibility; not a substitute for hardware validation."
)
MIN_GRID_BINS = 2
NMS_RANGE_M = 0.18
NMS_ANGLE_DEG = 12.0
NEAR_BAND_MAX_M = 0.50
MID_BAND_MAX_M = 1.00


def simulate_scenario(
    scenario: MmWaveScenario,
    config: MmWaveSimulationConfig,
) -> MmWaveScenarioResult:
    heatmap = simulate_heatmap(scenario.obstacles, config)
    detections = detect_obstacles(heatmap, config)
    return MmWaveScenarioResult(
        scenario=scenario,
        heatmap=heatmap,
        detections=detections,
    )


def simulate_heatmap(
    obstacles: tuple[SimulatedObstacle, ...],
    config: MmWaveSimulationConfig,
) -> tuple[tuple[float, ...], ...]:
    _validate_config(config)
    rows: list[tuple[float, ...]] = []
    for range_index in range(config.range_bins):
        range_m = _range_at_index(range_index, config)
        row: list[float] = []
        for angle_index in range(config.angle_bins):
            angle_deg = _angle_at_index(angle_index, config)
            value = _background_reflection(range_m, angle_deg, config)
            for obstacle in obstacles:
                value += _obstacle_response(range_m, angle_deg, obstacle, config)
            row.append(min(1.0, value))
        rows.append(tuple(row))
    return tuple(rows)


def detect_obstacles(
    heatmap: tuple[tuple[float, ...], ...],
    config: MmWaveSimulationConfig,
) -> tuple[ObstacleDetection, ...]:
    _validate_heatmap(heatmap, config)
    detections: list[ObstacleDetection] = []
    for sector in ObstacleSector:
        for distance_band in DistanceBand:
            peak = _peak_for_zone(heatmap, config, sector, distance_band)
            if peak is None:
                continue
            range_m, angle_deg, confidence = peak
            if confidence >= config.detection_threshold:
                detections.append(
                    ObstacleDetection(
                        sector=sector,
                        distance_band=distance_band,
                        range_m=range_m,
                        angle_deg=angle_deg,
                        confidence=confidence,
                    ),
                )
    return _suppress_overlapping_detections(tuple(detections))


def render_heatmap_panel(
    result: MmWaveScenarioResult,
    config: MmWaveSimulationConfig,
    *,
    cell_size: int = 6,
) -> Image.Image:
    width = config.angle_bins * cell_size
    height = config.range_bins * cell_size
    panel = Image.new("RGB", (width, height), (8, 10, 12))
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(panel)
    for range_index, row in enumerate(result.heatmap):
        y_start = (config.range_bins - range_index - 1) * cell_size
        for angle_index, value in enumerate(row):
            color = _heat_color(value)
            x_start = angle_index * cell_size
            draw.rectangle(
                (
                    x_start,
                    y_start,
                    x_start + cell_size - 1,
                    y_start + cell_size - 1,
                ),
                fill=color,
            )
    _draw_zone_guides(draw, width, height, config, cell_size)
    _draw_detection_markers(draw, result.detections, config, cell_size)
    return panel


def render_scenario_contact_sheet(
    results: tuple[MmWaveScenarioResult, ...],
    config: MmWaveSimulationConfig,
) -> Image.Image:
    panels = tuple(render_heatmap_panel(result, config) for result in results)
    header_height = 42
    separator_height = 6
    padding = 12
    sheet_width = max(680, *(panel.width for panel in panels))
    sheet_height = (
        sum(
            panel.height + header_height + separator_height + padding
            for panel in panels
        )
        - padding
    )
    sheet = Image.new("RGB", (sheet_width, sheet_height), (12, 12, 12))
    y = 0
    for panel, result in zip(panels, results, strict=True):
        draw: ImageDraw.ImageDraw = ImageDraw.Draw(sheet)
        fill = _scenario_separator_color(result.detections)
        draw.rectangle((0, y, sheet_width, y + header_height - 1), fill=(18, 18, 18))
        draw.rectangle(
            (
                0,
                y + header_height,
                sheet_width,
                y + header_height + separator_height - 1,
            ),
            fill=fill,
        )
        _draw_text(
            draw,
            (12, y + 7),
            f"{result.scenario.name} | {_detection_summary(result.detections)}",
            fill=(235, 235, 235),
        )
        panel_y = y + header_height + separator_height
        sheet.paste(panel, ((sheet_width - panel.width) // 2, panel_y))
        y += header_height + separator_height + panel.height + padding
    return sheet


def write_simulation_artifacts(
    scenarios: tuple[MmWaveScenario, ...],
    config: MmWaveSimulationConfig,
    out_dir: Path,
) -> SimulationPayload:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = tuple(simulate_scenario(scenario, config) for scenario in scenarios)
    sheet = render_scenario_contact_sheet(results, config)
    sheet.save(out_dir / "mmwave_front_obstacle_heatmaps.png")
    payload = SimulationPayload(
        protocol=PROTOCOL_NAME,
        claim=CLAIM_TEXT,
        scenarios=[
            ScenarioPayload(
                scenario=result.scenario.name,
                detections=[
                    _detection_payload(detection) for detection in result.detections
                ],
            )
            for result in results
        ],
    )
    _ = (out_dir / "mmwave_front_obstacle_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _obstacle_response(
    range_m: float,
    angle_deg: float,
    obstacle: SimulatedObstacle,
    config: MmWaveSimulationConfig,
) -> float:
    range_term = (range_m - obstacle.range_m) / config.range_sigma_m
    angle_term = (angle_deg - obstacle.angle_deg) / config.angle_sigma_deg
    return obstacle.intensity * math.exp(-0.5 * ((range_term**2) + (angle_term**2)))


def _background_reflection(
    range_m: float,
    angle_deg: float,
    config: MmWaveSimulationConfig,
) -> float:
    range_ratio = (range_m - config.range_min_m) / (
        config.range_max_m - config.range_min_m
    )
    angle_ratio = abs(angle_deg) / max(abs(config.angle_min_deg), config.angle_max_deg)
    return 0.025 + (0.035 * range_ratio) + (0.015 * angle_ratio)


def _peak_for_zone(
    heatmap: tuple[tuple[float, ...], ...],
    config: MmWaveSimulationConfig,
    sector: ObstacleSector,
    distance_band: DistanceBand,
) -> tuple[float, float, float] | None:
    best: tuple[int, int, float] | None = None
    range_limits = _distance_band_limits(distance_band, config)
    angle_limits = _sector_limits(sector)
    for range_index, row in enumerate(heatmap):
        range_m = _range_at_index(range_index, config)
        if range_m < range_limits[0] or range_m > range_limits[1]:
            continue
        for angle_index, value in enumerate(row):
            angle_deg = _angle_at_index(angle_index, config)
            if angle_deg < angle_limits[0] or angle_deg > angle_limits[1]:
                continue
            if best is None or value > best[2]:
                best = (range_index, angle_index, value)
    if best is None:
        return None
    return (
        _range_at_index(best[0], config),
        _angle_at_index(best[1], config),
        best[2],
    )


def _suppress_overlapping_detections(
    detections: tuple[ObstacleDetection, ...],
) -> tuple[ObstacleDetection, ...]:
    kept: list[ObstacleDetection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if any(_is_same_peak(detection, existing) for existing in kept):
            continue
        kept.append(detection)
    return tuple(
        sorted(
            kept,
            key=lambda detection: (
                detection.range_m,
                detection.sector.value,
                -detection.confidence,
            ),
        ),
    )


def _is_same_peak(
    candidate: ObstacleDetection,
    existing: ObstacleDetection,
) -> bool:
    return (
        abs(candidate.range_m - existing.range_m) <= NMS_RANGE_M
        and abs(candidate.angle_deg - existing.angle_deg) <= NMS_ANGLE_DEG
    )


def _distance_band_limits(
    distance_band: DistanceBand,
    config: MmWaveSimulationConfig,
) -> tuple[float, float]:
    match distance_band:
        case DistanceBand.NEAR:
            return config.range_min_m, NEAR_BAND_MAX_M
        case DistanceBand.MID:
            return NEAR_BAND_MAX_M, MID_BAND_MAX_M
        case DistanceBand.FAR:
            return MID_BAND_MAX_M, config.range_max_m


def _sector_limits(sector: ObstacleSector) -> tuple[float, float]:
    match sector:
        case ObstacleSector.LEFT:
            return -60.0, -20.0
        case ObstacleSector.CENTER:
            return -20.0, 20.0
        case ObstacleSector.RIGHT:
            return 20.0, 60.0


def _range_at_index(index: int, config: MmWaveSimulationConfig) -> float:
    if config.range_bins == 1:
        return config.range_min_m
    step = (config.range_max_m - config.range_min_m) / (config.range_bins - 1)
    return config.range_min_m + (index * step)


def _angle_at_index(index: int, config: MmWaveSimulationConfig) -> float:
    if config.angle_bins == 1:
        return config.angle_min_deg
    step = (config.angle_max_deg - config.angle_min_deg) / (config.angle_bins - 1)
    return config.angle_min_deg + (index * step)


def _heat_color(value: float) -> tuple[int, int, int]:
    normalized = max(0.0, min(1.0, value))
    red = int(255 * min(1.0, normalized * 1.35))
    green = int(255 * max(0.0, 1.0 - abs(normalized - 0.55) * 1.8))
    blue = int(70 + (110 * (1.0 - normalized)))
    return red, green, blue


def _draw_zone_guides(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    config: MmWaveSimulationConfig,
    cell_size: int,
) -> None:
    for angle_deg in (-20.0, 20.0):
        x = int((_angle_to_column(angle_deg, config) + 0.5) * cell_size)
        draw.line((x, 0, x, height), fill=(110, 110, 110), width=1)
    for range_m in (NEAR_BAND_MAX_M, MID_BAND_MAX_M):
        y = height - int((_range_to_row(range_m, config) + 0.5) * cell_size)
        draw.line((0, y, width, y), fill=(110, 110, 110), width=1)


def _draw_detection_markers(
    draw: ImageDraw.ImageDraw,
    detections: tuple[ObstacleDetection, ...],
    config: MmWaveSimulationConfig,
    cell_size: int,
) -> None:
    for detection in detections:
        x = int((_angle_to_column(detection.angle_deg, config) + 0.5) * cell_size)
        y = (config.range_bins * cell_size) - int(
            (_range_to_row(detection.range_m, config) + 0.5) * cell_size,
        )
        radius = 7
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=(255, 255, 255),
            width=2,
        )


def _range_to_row(range_m: float, config: MmWaveSimulationConfig) -> float:
    return (
        (range_m - config.range_min_m)
        / (config.range_max_m - config.range_min_m)
        * (config.range_bins - 1)
    )


def _angle_to_column(angle_deg: float, config: MmWaveSimulationConfig) -> float:
    return (
        (angle_deg - config.angle_min_deg)
        / (config.angle_max_deg - config.angle_min_deg)
        * (config.angle_bins - 1)
    )


def _scenario_separator_color(
    detections: tuple[ObstacleDetection, ...],
) -> tuple[int, int, int]:
    if len(detections) == 0:
        return 30, 90, 60
    closest_range = min(detection.range_m for detection in detections)
    if closest_range <= NEAR_BAND_MAX_M:
        return 180, 40, 45
    if closest_range <= MID_BAND_MAX_M:
        return 195, 130, 35
    return 70, 120, 190


def _detection_summary(detections: tuple[ObstacleDetection, ...]) -> str:
    if len(detections) == 0:
        return "clear path"
    return "; ".join(
        (
            f"{detection.sector.value}/{detection.distance_band.value} "
            f"{detection.range_m:.2f}m {detection.angle_deg:+.1f}deg"
        )
        for detection in detections
    )


def _draw_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    *,
    fill: tuple[int, int, int],
) -> None:
    text_drawer = cast("_TextDrawer", draw)
    text_drawer.text(position, text, fill=fill)


def _detection_payload(detection: ObstacleDetection) -> ObstacleDetectionPayload:
    return ObstacleDetectionPayload(
        sector=detection.sector.value,
        distance_band=detection.distance_band.value,
        range_m=round(detection.range_m, 3),
        angle_deg=round(detection.angle_deg, 3),
        confidence=round(detection.confidence, 3),
    )


def _validate_config(config: MmWaveSimulationConfig) -> None:
    if config.range_min_m <= 0 or config.range_min_m >= config.range_max_m:
        raise MmWaveSimulationError(
            message="range_min_m must be positive and below range_max_m",
        )
    if config.range_bins < MIN_GRID_BINS or config.angle_bins < MIN_GRID_BINS:
        raise MmWaveSimulationError(
            message="range_bins and angle_bins must be at least 2",
        )
    if config.angle_min_deg >= config.angle_max_deg:
        raise MmWaveSimulationError(
            message="angle_min_deg must be below angle_max_deg",
        )
    if config.range_sigma_m <= 0 or config.angle_sigma_deg <= 0:
        raise MmWaveSimulationError(message="sigma values must be positive")
    if config.detection_threshold <= 0 or config.detection_threshold >= 1:
        raise MmWaveSimulationError(
            message="detection_threshold must be in (0, 1)",
        )


def _validate_heatmap(
    heatmap: tuple[tuple[float, ...], ...],
    config: MmWaveSimulationConfig,
) -> None:
    if len(heatmap) != config.range_bins:
        raise MmWaveSimulationError(
            message="heatmap range dimension does not match config",
        )
    if any(len(row) != config.angle_bins for row in heatmap):
        raise MmWaveSimulationError(
            message="heatmap angle dimension does not match config",
        )
