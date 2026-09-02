from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

from PIL import Image, ImageDraw

from firesight_vision.hud_edges import (
    HudEdgeProfile,
    HudEdgeResult,
    build_hud_edge_overlay,
)
from firesight_vision.mmwave_sim import (
    DistanceBand,
    MmWaveScenario,
    MmWaveScenarioResult,
    MmWaveSimulationConfig,
    ObstacleDetection,
    ObstacleDetectionPayload,
    simulate_scenario,
)
from firesight_vision.thermal_edges import (
    ThermalHudConfig,
    ThermalHudResult,
    build_thermal_hud_overlay,
)

if TYPE_CHECKING:
    from pathlib import Path


ROS2_FRONT_OBSTACLE_TOPIC = "/firesight/mmwave/front_obstacles"
PROTOCOL_NAME = "ros2_mock_mmwave_hud_fusion_v1"
THERMAL_PROTOCOL_NAME = "ros2_mock_thermal_mmwave_hud_fusion_v1"
CLAIM_TEXT = (
    "Sensor-free ROS2-style mmWave topic stream fused with HUD edge/fire overlay; "
    "validates message and rendering contracts, not radar hardware physics."
)
THERMAL_CLAIM_TEXT = (
    "Sensor-free ROS2-style thermal edge/hotspot overlay fused with mmWave front "
    "obstacle cues; validates HUD contracts before LWIR and TI mmWave hardware."
)
DEFAULT_FRAME_ID = "mmwave_front_link"
DEFAULT_START_STAMP_NS = 1_725_000_000_000_000_000
DEFAULT_PERIOD_NS = 100_000_000
PANEL_BACKGROUND = (0, 0, 0, 150)
PANEL_GRID = (140, 150, 155, 125)
NEAR_WARNING = (255, 50, 50, 230)
MID_WARNING = (255, 178, 36, 220)
FAR_WARNING = (66, 180, 255, 210)
CLEAR_STATUS = (42, 220, 120, 230)


class HeaderPayload(TypedDict):
    stamp_ns: int
    frame_id: str
    seq: int


class MmWaveMockMessagePayload(TypedDict):
    topic: str
    header: HeaderPayload
    scenario: str
    closest_range_m: float | None
    detections: list[ObstacleDetectionPayload]


class FusionFramePayload(TypedDict):
    image_file: str
    edge_ratio: float
    message: MmWaveMockMessagePayload


class FusionRunPayload(TypedDict):
    protocol: str
    claim: str
    topic: str
    frames: list[FusionFramePayload]


class ThermalFusionFramePayload(TypedDict):
    image_file: str
    edge_ratio: float
    hotspot_ratio: float
    hotspot_pixels: int
    message: MmWaveMockMessagePayload


class ThermalFusionRunPayload(TypedDict):
    protocol: str
    claim: str
    thermal_input_topic: str
    thermal_edge_mask_topic: str
    thermal_edge_overlay_topic: str
    mmwave_obstacle_topic: str
    fused_overlay_topic: str
    frames: list[ThermalFusionFramePayload]


class _TextDrawer(Protocol):
    def text(
        self,
        xy: tuple[int, int],
        text: str,
        *,
        fill: tuple[int, int, int, int],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class MmWaveMockMessage:
    topic: str
    stamp_ns: int
    frame_id: str
    seq: int
    result: MmWaveScenarioResult


@dataclass(frozen=True, slots=True)
class MmWaveStreamConfig:
    topic: str = ROS2_FRONT_OBSTACLE_TOPIC
    frame_id: str = DEFAULT_FRAME_ID
    start_stamp_ns: int = DEFAULT_START_STAMP_NS
    period_ns: int = DEFAULT_PERIOD_NS


@dataclass(frozen=True, slots=True)
class FusionRenderConfig:
    threshold: int = 128
    background_scale: float = 0.24
    edge_width: int = 5
    profile: HudEdgeProfile = HudEdgeProfile.FIRE_LINE


@dataclass(frozen=True, slots=True)
class FusionFrame:
    source_image: Image.Image
    image: Image.Image
    hud_result: HudEdgeResult
    message: MmWaveMockMessage


@dataclass(frozen=True, slots=True)
class ThermalFusionFrame:
    source_image: Image.Image
    image: Image.Image
    thermal_result: ThermalHudResult
    message: MmWaveMockMessage


@dataclass(frozen=True, slots=True)
class FusionPanelGeometry:
    left: int
    right: int
    top: int
    bottom: int
    center_x: int


@dataclass(frozen=True, slots=True)
class FusionError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def build_mock_mmwave_stream(
    scenarios: tuple[MmWaveScenario, ...],
    config: MmWaveSimulationConfig,
    *,
    stream_config: MmWaveStreamConfig | None = None,
) -> tuple[MmWaveMockMessage, ...]:
    resolved_stream_config = (
        MmWaveStreamConfig() if stream_config is None else stream_config
    )
    return tuple(
        MmWaveMockMessage(
            topic=resolved_stream_config.topic,
            stamp_ns=resolved_stream_config.start_stamp_ns
            + (index * resolved_stream_config.period_ns),
            frame_id=resolved_stream_config.frame_id,
            seq=index,
            result=simulate_scenario(scenario, config),
        )
        for index, scenario in enumerate(scenarios)
    )


def render_hud_mmwave_fusion(
    source_image: Image.Image,
    message: MmWaveMockMessage,
    *,
    render_config: FusionRenderConfig | None = None,
) -> FusionFrame:
    resolved_render_config = (
        FusionRenderConfig() if render_config is None else render_config
    )
    hud_result = build_hud_edge_overlay(
        source_image,
        threshold=resolved_render_config.threshold,
        background_scale=resolved_render_config.background_scale,
        edge_width=resolved_render_config.edge_width,
        profile=resolved_render_config.profile,
    )
    fused = hud_result.overlay.convert("RGBA")
    radar_layer = Image.new("RGBA", fused.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(radar_layer)
    _draw_front_obstacle_panel(draw, fused.size, message)
    return FusionFrame(
        source_image=source_image.copy(),
        image=Image.alpha_composite(fused, radar_layer).convert("RGB"),
        hud_result=hud_result,
        message=message,
    )


def render_thermal_hud_mmwave_fusion(
    source_image: Image.Image,
    message: MmWaveMockMessage,
    *,
    thermal_config: ThermalHudConfig | None = None,
) -> ThermalFusionFrame:
    thermal_result = build_thermal_hud_overlay(
        source_image,
        config=thermal_config,
    )
    fused = thermal_result.overlay.convert("RGBA")
    radar_layer = Image.new("RGBA", fused.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(radar_layer)
    _draw_front_obstacle_panel(draw, fused.size, message)
    return ThermalFusionFrame(
        source_image=source_image.copy(),
        image=Image.alpha_composite(fused, radar_layer).convert("RGB"),
        thermal_result=thermal_result,
        message=message,
    )


def write_fusion_artifacts(
    source_images: tuple[Image.Image, ...],
    messages: tuple[MmWaveMockMessage, ...],
    out_dir: Path,
    *,
    render_config: FusionRenderConfig | None = None,
    render_configs: tuple[FusionRenderConfig, ...] | None = None,
) -> FusionRunPayload:
    if len(source_images) != len(messages):
        message = "source_images and messages must have the same length"
        raise FusionError(message=message)
    if render_configs is not None and len(source_images) != len(render_configs):
        message = "source_images and render_configs must have the same length"
        raise FusionError(message=message)

    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[FusionFrame] = []
    payload_frames: list[FusionFramePayload] = []
    for index, (source_image, message) in enumerate(
        zip(source_images, messages, strict=True),
    ):
        frame = render_hud_mmwave_fusion(
            source_image,
            message,
            render_config=(
                render_config if render_configs is None else render_configs[index]
            ),
        )
        image_file = f"frame_{index:03d}_hud_mmwave.png"
        frame.image.save(out_dir / image_file)
        frames.append(frame)
        payload_frames.append(
            FusionFramePayload(
                image_file=image_file,
                edge_ratio=round(frame.hud_result.edge_ratio, 4),
                message=message_payload(message),
            ),
        )

    _write_jsonl(out_dir / "mmwave_front_obstacle_mock_stream.jsonl", messages)
    contact_sheet = render_fusion_contact_sheet(tuple(frames))
    contact_sheet.save(out_dir / "mmwave_hud_fusion_contact_sheet.png")
    payload = FusionRunPayload(
        protocol=PROTOCOL_NAME,
        claim=CLAIM_TEXT,
        topic=ROS2_FRONT_OBSTACLE_TOPIC,
        frames=payload_frames,
    )
    _ = (out_dir / "mmwave_hud_fusion_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def write_thermal_fusion_artifacts(
    source_images: tuple[Image.Image, ...],
    messages: tuple[MmWaveMockMessage, ...],
    out_dir: Path,
    *,
    thermal_config: ThermalHudConfig | None = None,
) -> ThermalFusionRunPayload:
    if len(source_images) != len(messages):
        message = "source_images and messages must have the same length"
        raise FusionError(message=message)

    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[ThermalFusionFrame] = []
    payload_frames: list[ThermalFusionFramePayload] = []
    for index, (source_image, message) in enumerate(
        zip(source_images, messages, strict=True),
    ):
        frame = render_thermal_hud_mmwave_fusion(
            source_image,
            message,
            thermal_config=thermal_config,
        )
        image_file = f"frame_{index:03d}_thermal_hud_mmwave.png"
        frame.image.save(out_dir / image_file)
        frame.thermal_result.edge_mask.save(
            out_dir / f"frame_{index:03d}_edge_mask.png"
        )
        frame.thermal_result.hotspot_mask.save(
            out_dir / f"frame_{index:03d}_hotspot_mask.png",
        )
        frames.append(frame)
        payload_frames.append(
            ThermalFusionFramePayload(
                image_file=image_file,
                edge_ratio=round(frame.thermal_result.edge_ratio, 4),
                hotspot_ratio=round(frame.thermal_result.hotspot_ratio, 4),
                hotspot_pixels=frame.thermal_result.hotspot_pixels,
                message=message_payload(message),
            ),
        )

    _write_jsonl(out_dir / "mmwave_front_obstacles_mock_stream.jsonl", messages)
    contact_sheet = render_thermal_fusion_contact_sheet(tuple(frames))
    contact_sheet.save(out_dir / "thermal_mmwave_hud_fusion_contact_sheet.png")
    payload = ThermalFusionRunPayload(
        protocol=THERMAL_PROTOCOL_NAME,
        claim=THERMAL_CLAIM_TEXT,
        thermal_input_topic="/thermal_camera/image_raw",
        thermal_edge_mask_topic="/firesight/thermal/edge_mask",
        thermal_edge_overlay_topic="/firesight/thermal/edge_overlay",
        mmwave_obstacle_topic=ROS2_FRONT_OBSTACLE_TOPIC,
        fused_overlay_topic="/firesight/hud/fused_overlay",
        frames=payload_frames,
    )
    _ = (out_dir / "thermal_mmwave_hud_fusion_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def render_fusion_contact_sheet(frames: tuple[FusionFrame, ...]) -> Image.Image:
    if len(frames) == 0:
        return Image.new("RGB", (1, 1), (12, 12, 12))

    tile_width = 300
    header_height = 28
    caption_height = 34
    pad = 12
    tile_height = round(tile_width * frames[0].image.height / frames[0].image.width)
    sheet = Image.new(
        "RGB",
        (
            (3 * tile_width) + (4 * pad),
            header_height + (len(frames) * (tile_height + caption_height + pad)) + pad,
        ),
        (12, 12, 12),
    )
    draw = ImageDraw.Draw(sheet)
    for index, label in enumerate(
        ("RGB INPUT", "EDGE + FIRE LINE", "FUSED HUD + MMWAVE"),
    ):
        _draw_text(
            draw,
            (pad + (index * (tile_width + pad)), 7),
            label,
            fill=(230, 230, 230, 255),
        )
    for index, frame in enumerate(frames):
        y = header_height + (index * (tile_height + caption_height + pad))
        for column, image in enumerate(
            (frame.source_image, frame.hud_result.overlay, frame.image),
        ):
            x = pad + (column * (tile_width + pad))
            resized = image.convert("RGB").resize(
                (tile_width, tile_height), Image.Resampling.LANCZOS
            )
            sheet.paste(resized, (x, y))
        _draw_text(
            draw,
            (pad, y + tile_height + 5),
            _caption(frame.message),
            fill=(230, 230, 230, 255),
        )
    return sheet


def render_thermal_fusion_contact_sheet(
    frames: tuple[ThermalFusionFrame, ...],
) -> Image.Image:
    if len(frames) == 0:
        return Image.new("RGB", (1, 1), (12, 12, 12))

    tile_width = 300
    header_height = 28
    caption_height = 34
    pad = 12
    tile_height = round(tile_width * frames[0].image.height / frames[0].image.width)
    sheet = Image.new(
        "RGB",
        (
            (3 * tile_width) + (4 * pad),
            header_height + (len(frames) * (tile_height + caption_height + pad)) + pad,
        ),
        (12, 12, 12),
    )
    draw = ImageDraw.Draw(sheet)
    for index, label in enumerate(
        ("THERMAL INPUT", "EDGE + HOTSPOT", "FUSED HUD + MMWAVE"),
    ):
        _draw_text(
            draw,
            (pad + (index * (tile_width + pad)), 7),
            label,
            fill=(230, 230, 230, 255),
        )
    for index, frame in enumerate(frames):
        y = header_height + (index * (tile_height + caption_height + pad))
        for column, image in enumerate(
            (
                frame.source_image,
                frame.thermal_result.overlay,
                frame.image,
            ),
        ):
            x = pad + (column * (tile_width + pad))
            resized = image.convert("RGB").resize(
                (tile_width, tile_height), Image.Resampling.LANCZOS
            )
            sheet.paste(resized, (x, y))
        _draw_text(
            draw,
            (pad, y + tile_height + 5),
            _thermal_caption(frame),
            fill=(230, 230, 230, 255),
        )
    return sheet


def message_payload(message: MmWaveMockMessage) -> MmWaveMockMessagePayload:
    return MmWaveMockMessagePayload(
        topic=message.topic,
        header=HeaderPayload(
            stamp_ns=message.stamp_ns,
            frame_id=message.frame_id,
            seq=message.seq,
        ),
        scenario=message.result.scenario.name,
        closest_range_m=_closest_range(message.result.detections),
        detections=[
            _detection_payload(detection) for detection in message.result.detections
        ],
    )


def _draw_front_obstacle_panel(
    draw: ImageDraw.ImageDraw,
    image_size: tuple[int, int],
    message: MmWaveMockMessage,
) -> None:
    width, height = image_size
    panel_height = max(92, round(height * 0.28))
    geometry = FusionPanelGeometry(
        left=round(width * 0.08),
        right=round(width * 0.92),
        top=height - panel_height - 10,
        bottom=height - 10,
        center_x=width // 2,
    )
    draw.rectangle(
        (geometry.left, geometry.top, geometry.right, geometry.bottom),
        fill=PANEL_BACKGROUND,
    )
    for angle_deg in (-20.0, 20.0):
        x = _angle_to_x(angle_deg, geometry.left, geometry.right)
        draw.line((x, geometry.top, x, geometry.bottom), fill=PANEL_GRID, width=1)
    for range_m in (0.5, 1.0):
        y = _range_to_y(range_m, geometry.top, geometry.bottom)
        draw.line((geometry.left, y, geometry.right, y), fill=PANEL_GRID, width=1)
    draw.line(
        (geometry.center_x, geometry.bottom, geometry.center_x, geometry.top),
        fill=(185, 185, 185, 150),
        width=1,
    )

    if len(message.result.detections) == 0:
        _draw_text(
            draw,
            (geometry.left + 8, geometry.top + 8),
            "mmWave: clear",
            fill=CLEAR_STATUS,
        )
        return

    closest = min(message.result.detections, key=lambda detection: detection.range_m)
    _draw_text(
        draw,
        (geometry.left + 8, geometry.top + 8),
        f"mmWave: {closest.distance_band.value} {closest.range_m:.2f}m",
        fill=_band_color(closest.distance_band),
    )
    for detection in message.result.detections:
        _draw_detection_marker(draw, detection, geometry)


def _draw_detection_marker(
    draw: ImageDraw.ImageDraw,
    detection: ObstacleDetection,
    geometry: FusionPanelGeometry,
) -> None:
    x = _angle_to_x(detection.angle_deg, geometry.left, geometry.right)
    y = _range_to_y(detection.range_m, geometry.top, geometry.bottom)
    color = _band_color(detection.distance_band)
    radius = round(7 + (detection.confidence * 7))
    draw.line((geometry.center_x, geometry.bottom, x, y), fill=color, width=2)
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius), outline=color, width=3
    )
    inner_radius = max(3, radius // 2)
    draw.ellipse(
        (x - inner_radius, y - inner_radius, x + inner_radius, y + inner_radius),
        fill=color,
    )


def _write_jsonl(path: Path, messages: tuple[MmWaveMockMessage, ...]) -> None:
    lines: list[str] = [
        json.dumps(message_payload(message), sort_keys=True) for message in messages
    ]
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _detection_payload(detection: ObstacleDetection) -> ObstacleDetectionPayload:
    return ObstacleDetectionPayload(
        sector=detection.sector.value,
        distance_band=detection.distance_band.value,
        range_m=round(detection.range_m, 3),
        angle_deg=round(detection.angle_deg, 3),
        confidence=round(detection.confidence, 3),
    )


def _closest_range(detections: tuple[ObstacleDetection, ...]) -> float | None:
    if len(detections) == 0:
        return None
    return round(min(detection.range_m for detection in detections), 3)


def _caption(message: MmWaveMockMessage) -> str:
    closest = _closest_range(message.result.detections)
    if closest is None:
        return f"{message.seq:02d} {message.result.scenario.name}: clear"
    return f"{message.seq:02d} {message.result.scenario.name}: {closest:.2f}m"


def _thermal_caption(frame: ThermalFusionFrame) -> str:
    base = _caption(frame.message)
    return (
        f"{base} | edge {frame.thermal_result.edge_ratio:.2%} "
        f"hot {frame.thermal_result.hotspot_ratio:.2%}"
    )


def _band_color(distance_band: DistanceBand) -> tuple[int, int, int, int]:
    match distance_band:
        case DistanceBand.NEAR:
            return NEAR_WARNING
        case DistanceBand.MID:
            return MID_WARNING
        case DistanceBand.FAR:
            return FAR_WARNING


def _angle_to_x(angle_deg: float, left: int, right: int) -> int:
    normalized = (max(-60.0, min(60.0, angle_deg)) + 60.0) / 120.0
    return left + round((right - left) * normalized)


def _range_to_y(range_m: float, top: int, bottom: int) -> int:
    normalized = (max(0.2, min(1.5, range_m)) - 0.2) / 1.3
    return bottom - round((bottom - top) * normalized)


def _draw_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    *,
    fill: tuple[int, int, int, int],
) -> None:
    text_drawer = cast("_TextDrawer", draw)
    text_drawer.text(position, text, fill=fill)
