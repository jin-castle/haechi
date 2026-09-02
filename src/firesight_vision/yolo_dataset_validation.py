from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from firesight_vision.bbox_build import CATEGORIES

if TYPE_CHECKING:
    from pathlib import Path

YOLO_ROW_FIELD_COUNT = 5


@dataclass(frozen=True, slots=True)
class YoloDatasetValidationSummary:
    image_count: int
    label_count: int
    row_count: int


@dataclass(frozen=True, slots=True)
class YoloDatasetError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def validate_yolo_dataset(
    dataset_dir: Path,
    expected_image_count: int,
) -> YoloDatasetValidationSummary:
    _validate_data_yaml(dataset_dir / "data.yaml")
    image_files = sorted((dataset_dir / "images" / "all").glob("*.jpg"))
    label_files = sorted((dataset_dir / "labels" / "all").glob("*.txt"))
    if len(image_files) != expected_image_count:
        message = f"unexpected YOLO image count: {len(image_files)}"
        raise YoloDatasetError(message)
    if len(label_files) != expected_image_count:
        message = f"unexpected YOLO label count: {len(label_files)}"
        raise YoloDatasetError(message)
    expected_labels = {image.stem for image in image_files}
    actual_labels = {label.stem for label in label_files}
    if actual_labels != expected_labels:
        message = "YOLO label files must match image stems"
        raise YoloDatasetError(message)
    row_count = sum(_validate_label_file(label_path) for label_path in label_files)
    return YoloDatasetValidationSummary(
        image_count=len(image_files),
        label_count=len(label_files),
        row_count=row_count,
    )


def _validate_data_yaml(path: Path) -> None:
    if not path.exists():
        message = f"missing YOLO data.yaml: {path}"
        raise YoloDatasetError(message)
    text = path.read_text(encoding="utf-8")
    required_fragments = (
        "path: .",
        "train: images/all",
        "val: images/all",
        "test: images/all",
    )
    for fragment in required_fragments:
        if fragment not in text:
            message = f"missing YOLO data.yaml fragment: {fragment}"
            raise YoloDatasetError(message)
    for index, category in enumerate(CATEGORIES):
        if f"  {index}: {category}" not in text:
            message = f"missing YOLO class mapping: {category}"
            raise YoloDatasetError(message)


def _validate_label_file(path: Path) -> int:
    row_count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != YOLO_ROW_FIELD_COUNT:
            message = f"invalid YOLO row field count: {path}"
            raise YoloDatasetError(message)
        _validate_class_id(parts[0], path)
        for value in parts[1:]:
            _validate_normalized_value(value, path)
        row_count += 1
    return row_count


def _validate_class_id(value: str, path: Path) -> None:
    try:
        class_id = int(value)
    except ValueError as error:
        message = f"invalid YOLO class id: {path}"
        raise YoloDatasetError(message) from error
    if not 0 <= class_id < len(CATEGORIES):
        message = f"YOLO class id outside category range: {path}"
        raise YoloDatasetError(message)


def _validate_normalized_value(value: str, path: Path) -> None:
    try:
        number = float(value)
    except ValueError as error:
        message = f"invalid YOLO normalized value: {path}"
        raise YoloDatasetError(message) from error
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        message = f"YOLO normalized value outside range: {path}"
        raise YoloDatasetError(message)
