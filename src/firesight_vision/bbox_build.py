from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

    from firesight_vision.image_level_eval_types import JsonValue

CATEGORIES = ("door", "person", "exit", "obstacle")
SOURCE_COUNT_PER_TYPE = 30


class BboxBuildError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SelectedImage:
    id: int
    file_name: str
    width: int
    height: int
    source_type: str


@dataclass(frozen=True, slots=True)
class DraftBox:
    image_id: int
    category: str
    bbox: tuple[float, float, float, float]
    source: str


@dataclass(frozen=True, slots=True)
class SourceFillRequest:
    seed_names: tuple[str, ...]
    source_names: tuple[str, ...]
    prefix: str
    target_count: int


@dataclass(frozen=True, slots=True)
class RelativeBoxSpec:
    category: str
    rel: tuple[float, float, float, float]
    source: str


@dataclass(frozen=True, slots=True)
class RawBox:
    x: float
    y: float
    width: float
    height: float


def select_images(image_root: Path, seed_manifest: Path) -> tuple[SelectedImage, ...]:
    seed_items = _read_seed_items(seed_manifest)
    seed_names: list[str] = []
    for item in seed_items:
        if isinstance(item, dict):
            image_name = item.get("image")
            if isinstance(image_name, str):
                seed_names.append(image_name)
    photos = sorted(path.name for path in image_root.glob("photo__*.jpg"))
    videos = sorted(path.name for path in image_root.glob("video__*.jpg"))
    names = _fill_source(
        SourceFillRequest(
            tuple(seed_names),
            tuple(photos),
            "photo__",
            SOURCE_COUNT_PER_TYPE,
        ),
    )
    names += _fill_source(
        SourceFillRequest(
            tuple(seed_names),
            tuple(videos),
            "video__",
            SOURCE_COUNT_PER_TYPE,
        ),
    )
    return tuple(
        _selected_image(image_root, name, index)
        for index, name in enumerate(names, start=1)
    )


def build_boxes(selected: tuple[SelectedImage, ...]) -> tuple[DraftBox, ...]:
    boxes: list[DraftBox] = []
    for image in selected:
        boxes.extend(_heuristic_boxes(image))
    return tuple(boxes)


def _read_seed_items(seed_manifest: Path) -> list[JsonValue]:
    data = json.loads(seed_manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        message = "JSON root must be object"
        raise BboxBuildError(message)
    items = data.get("items")
    if not isinstance(items, list):
        message = "seed manifest items must be a list"
        raise BboxBuildError(message)
    return items


def _fill_source(request: SourceFillRequest) -> list[str]:
    selected = list(
        dict.fromkeys(
            name for name in request.seed_names if name.startswith(request.prefix)
        ),
    )
    pool = [name for name in request.source_names if name not in selected]
    if request.prefix == "video__":
        pool = _spread(pool, request.target_count - len(selected))
    for name in pool:
        if len(selected) >= request.target_count:
            break
        selected.append(name)
    if len(selected) != request.target_count:
        message = f"insufficient source images for {request.prefix}"
        raise BboxBuildError(message)
    return selected


def _spread(names: list[str], count: int) -> list[str]:
    if count <= 0:
        return []
    if count >= len(names):
        return names
    last_index = len(names) - 1
    indexes = sorted(
        {round(index * last_index / (count - 1)) for index in range(count)},
    )
    return [names[index] for index in indexes]


def _selected_image(image_root: Path, name: str, index: int) -> SelectedImage:
    with Image.open(image_root / name) as image:
        width, height = image.size
    source_type = "photo_image" if name.startswith("photo__") else "video_frame"
    return SelectedImage(index, name, width, height, source_type)


def _heuristic_boxes(image: SelectedImage) -> list[DraftBox]:
    boxes: list[DraftBox] = []
    if image.file_name in _PHOTO_DOOR_EXIT:
        boxes.append(_relative_box(image, _photo_door_spec("door")))
        boxes.append(_relative_box(image, _photo_door_spec("exit")))
    if image.file_name == _PHOTO_OBSTACLE:
        boxes.append(_relative_box(image, _photo_obstacle_spec()))
    if image.file_name.startswith("video__ifsi_video_8_"):
        boxes.append(_relative_box(image, _thermal_door_spec("door")))
        boxes.append(_relative_box(image, _thermal_door_spec("exit")))
        boxes.append(_relative_box(image, _thermal_obstacle_spec()))
    elif image.file_name.startswith("video__"):
        boxes.append(
            _relative_box(image, _obstacle_spec("manual_draft_fire360_obstacle")),
        )
    return boxes


_PHOTO_DOOR_EXIT = {
    "photo__train__-2022-04-05-195924_png_jpg.rf.266e62bd01acfb308a344ab027ce5031.jpg",
    "photo__valid__000692_jpg.rf.64524cf258cb55a892782007e41006f5.jpg",
}
_PHOTO_OBSTACLE = "photo__valid__000838_jpg.rf.6995f161450456670a68691f8c5963b9.jpg"


def _photo_door_spec(category: str) -> RelativeBoxSpec:
    source = (
        "manual_draft_photo_door" if category == "door" else "manual_draft_photo_egress"
    )
    return RelativeBoxSpec(category, (0.52, 0.06, 0.34, 0.86), source)


def _thermal_door_spec(category: str) -> RelativeBoxSpec:
    source = (
        "manual_draft_thermal_door"
        if category == "door"
        else "manual_draft_thermal_egress"
    )
    return RelativeBoxSpec(category, (0.04, 0.04, 0.42, 0.86), source)


def _obstacle_spec(source: str) -> RelativeBoxSpec:
    return RelativeBoxSpec("obstacle", (0.05, 0.55, 0.55, 0.38), source)


def _photo_obstacle_spec() -> RelativeBoxSpec:
    return RelativeBoxSpec(
        "obstacle",
        (0.00, 0.55, 0.70, 0.40),
        "manual_draft_photo_obstacle",
    )


def _thermal_obstacle_spec() -> RelativeBoxSpec:
    return RelativeBoxSpec(
        "obstacle",
        (0.04, 0.52, 0.58, 0.40),
        "manual_draft_thermal_obstacle",
    )


def _relative_box(image: SelectedImage, spec: RelativeBoxSpec) -> DraftBox:
    x, y, width, height = spec.rel
    raw = RawBox(
        x * image.width,
        y * image.height,
        width * image.width,
        height * image.height,
    )
    bbox = _clip_bbox(raw, image)
    return DraftBox(image.id, spec.category, bbox, spec.source)


def _clip_bbox(raw: RawBox, image: SelectedImage) -> tuple[float, float, float, float]:
    clipped_x = max(0.0, min(raw.x, image.width - 1.0))
    clipped_y = max(0.0, min(raw.y, image.height - 1.0))
    clipped_w = max(1.0, min(raw.width, image.width - clipped_x))
    clipped_h = max(1.0, min(raw.height, image.height - clipped_y))
    return (
        round(clipped_x, 3),
        round(clipped_y, 3),
        round(clipped_w, 3),
        round(clipped_h, 3),
    )
