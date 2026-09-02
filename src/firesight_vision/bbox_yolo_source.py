from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from firesight_vision.bbox_build import BboxBuildError, DraftBox, SelectedImage

YOLO_FIELD_COUNT = 5
YOLO_PERSON_CLASS_ID = "0"


@dataclass(frozen=True, slots=True)
class YoloPersonProvenance:
    image_id: int
    file_name: str
    status: str
    person_box_count: int


@dataclass(frozen=True, slots=True)
class YoloNormalizedBox:
    xc: float
    yc: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class RawYoloBox:
    x: float
    y: float
    width: float
    height: float


def person_boxes_from_yolo(
    selected: tuple[SelectedImage, ...],
    yolo_label_dir: Path,
) -> tuple[DraftBox, ...]:
    _ensure_label_dir(yolo_label_dir)
    boxes: list[DraftBox] = []
    for image in selected:
        boxes.extend(_person_boxes_for_image(image, yolo_label_dir))
    return tuple(boxes)


def build_yolo_person_provenance(
    selected: tuple[SelectedImage, ...],
    yolo_label_dir: Path,
) -> tuple[YoloPersonProvenance, ...]:
    _ensure_label_dir(yolo_label_dir)
    rows: list[YoloPersonProvenance] = []
    for image in selected:
        label_path = yolo_label_dir / f"{Path(image.file_name).stem}.txt"
        if not label_path.exists():
            rows.append(
                YoloPersonProvenance(image.id, image.file_name, "label_missing", 0),
            )
            continue
        person_count = _count_yolo_person_rows(label_path)
        status = "person_detected" if person_count > 0 else "no_person_row"
        rows.append(
            YoloPersonProvenance(image.id, image.file_name, status, person_count),
        )
    return tuple(rows)


def _ensure_label_dir(yolo_label_dir: Path) -> None:
    if not yolo_label_dir.is_dir():
        message = f"missing YOLO label directory: {yolo_label_dir}"
        raise BboxBuildError(message)


def _person_boxes_for_image(image: SelectedImage, label_dir: Path) -> list[DraftBox]:
    label_path = label_dir / f"{Path(image.file_name).stem}.txt"
    if not label_path.exists():
        return []
    boxes: list[DraftBox] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != YOLO_FIELD_COUNT:
            message = f"invalid YOLO row field count: {label_path}"
            raise BboxBuildError(message)
        if parts[0] == YOLO_PERSON_CLASS_ID:
            boxes.append(_yolo_person_box(image, parts[1:], label_path))
    return boxes


def _count_yolo_person_rows(label_path: Path) -> int:
    count = 0
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != YOLO_FIELD_COUNT:
            message = f"invalid YOLO row field count: {label_path}"
            raise BboxBuildError(message)
        if parts[0] == YOLO_PERSON_CLASS_ID:
            count += 1
    return count


def _yolo_person_box(
    image: SelectedImage,
    values: list[str],
    label_path: Path,
) -> DraftBox:
    xc, yc, width, height = (_parse_yolo_value(value, label_path) for value in values)
    normalized = YoloNormalizedBox(xc, yc, width, height)
    _validate_yolo_box_values(normalized, label_path)
    x = (normalized.xc - normalized.width / 2.0) * image.width
    y = (normalized.yc - normalized.height / 2.0) * image.height
    raw = RawYoloBox(
        x,
        y,
        normalized.width * image.width,
        normalized.height * image.height,
    )
    bbox = _validate_raw_box(raw, image, label_path)
    return DraftBox(image.id, "person", bbox, "yolo11n_coco_person")


def _parse_yolo_value(value: str, label_path: Path) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        message = f"invalid YOLO numeric value: {label_path}"
        raise BboxBuildError(message) from error
    if not math.isfinite(parsed):
        message = f"nonfinite YOLO value: {label_path}"
        raise BboxBuildError(message)
    return parsed


def _validate_yolo_box_values(box: YoloNormalizedBox, label_path: Path) -> None:
    if not 0.0 <= box.xc <= 1.0 or not 0.0 <= box.yc <= 1.0:
        message = f"YOLO center outside normalized range: {label_path}"
        raise BboxBuildError(message)
    if not 0.0 < box.width <= 1.0 or not 0.0 < box.height <= 1.0:
        message = f"YOLO size outside normalized range: {label_path}"
        raise BboxBuildError(message)


def _validate_raw_box(
    raw: RawYoloBox,
    image: SelectedImage,
    label_path: Path,
) -> tuple[float, float, float, float]:
    if raw.x < 0 or raw.y < 0:
        message = f"YOLO bbox starts outside image: {label_path}"
        raise BboxBuildError(message)
    if raw.x + raw.width > image.width or raw.y + raw.height > image.height:
        message = f"YOLO bbox exceeds image bounds: {label_path}"
        raise BboxBuildError(message)
    return (
        round(raw.x, 3),
        round(raw.y, 3),
        round(raw.width, 3),
        round(raw.height, 3),
    )
