from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from pathlib import Path

    from firesight_vision.image_level_eval_types import JsonValue

AREA_TOLERANCE = 1e-3
BBOX_COORDINATE_COUNT = 4
REQUIRED_ANNOTATION_STATUS = "draft_single_agent"
REQUIRED_HUMAN_REVIEW_STATUS = "required_before_training_or_benchmark"
REQUIRED_CATEGORIES = ("door", "person", "exit", "obstacle")


class BboxInfoPayload(TypedDict, total=False):
    annotation_status: str
    human_review_status: str
    source: str


class BboxImagePayload(TypedDict):
    id: int
    file_name: str
    width: int
    height: int


class BboxCategoryPayload(TypedDict):
    id: int
    name: str


class BboxAnnotationPayload(TypedDict):
    id: int
    image_id: int
    category_id: int
    bbox: list[float]
    area: float
    iscrowd: int


@dataclass(frozen=True, slots=True)
class BboxValidationSummary:
    image_count: int
    annotation_count: int
    category_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class BboxAnnotationError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def validate_coco_bbox_file(path: Path) -> BboxValidationSummary:
    payload = _read_json_object(path)
    _validate_metadata(payload)
    images = _read_images(payload)
    categories = _read_categories(payload)
    category_counts = {category.name: 0 for category in categories.values()}
    annotations = _read_annotations(payload)
    for annotation in annotations:
        image = images.get(annotation.image_id)
        if image is None:
            raise BboxAnnotationError(
                message=f"annotation references unknown image: {annotation.id}",
            )
        category = categories.get(annotation.category_id)
        if category is None:
            raise BboxAnnotationError(
                message=f"annotation references unknown category: {annotation.id}",
            )
        _validate_bbox_bounds(annotation, image)
        category_counts[category.name] += 1
    return BboxValidationSummary(
        image_count=len(images),
        annotation_count=len(annotations),
        category_counts=category_counts,
    )


def _validate_metadata(payload: dict[str, JsonValue]) -> None:
    raw_info = payload.get("info")
    if not isinstance(raw_info, dict):
        raise BboxAnnotationError(message="info must be an object")
    annotation_status = raw_info.get("annotation_status")
    if annotation_status != REQUIRED_ANNOTATION_STATUS:
        raise BboxAnnotationError(
            message="annotation_status must be draft_single_agent",
        )
    human_review_status = raw_info.get("human_review_status")
    if human_review_status != REQUIRED_HUMAN_REVIEW_STATUS:
        raise BboxAnnotationError(message="human_review_status is required")
    raw_licenses = payload.get("licenses")
    if not isinstance(raw_licenses, list):
        raise BboxAnnotationError(message="licenses must be a list")


@dataclass(frozen=True, slots=True)
class _Image:
    id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class _Category:
    id: int
    name: str


@dataclass(frozen=True, slots=True)
class _Annotation:
    id: int
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    area: float
    iscrowd: int


def _read_json_object(path: Path) -> dict[str, JsonValue]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise BboxAnnotationError(message=f"missing annotation file: {path}") from error
    except json.JSONDecodeError as error:
        raise BboxAnnotationError(message=f"invalid annotation JSON: {path}") from error
    if not isinstance(data, dict):
        raise BboxAnnotationError(message=f"annotation root must be an object: {path}")
    return _json_object(data)


def _json_object(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: _json_value(item) for key, item in value.items()}


def _json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return _json_object(value)


def _read_images(payload: dict[str, JsonValue]) -> dict[int, _Image]:
    raw_images = payload.get("images")
    if not isinstance(raw_images, list):
        raise BboxAnnotationError(message="images must be a list")
    images: dict[int, _Image] = {}
    for raw_image in raw_images:
        if not isinstance(raw_image, dict):
            raise BboxAnnotationError(message="image entry must be an object")
        image_id = _required_int(raw_image, "id")
        if image_id in images:
            raise BboxAnnotationError(message=f"duplicate image id: {image_id}")
        images[image_id] = _Image(
            id=image_id,
            file_name=_required_str(raw_image, "file_name"),
            width=_required_positive_int(raw_image, "width"),
            height=_required_positive_int(raw_image, "height"),
        )
    return images


def _read_categories(payload: dict[str, JsonValue]) -> dict[int, _Category]:
    raw_categories = payload.get("categories")
    if not isinstance(raw_categories, list):
        raise BboxAnnotationError(message="categories must be a list")
    categories: dict[int, _Category] = {}
    names: set[str] = set()
    for raw_category in raw_categories:
        if not isinstance(raw_category, dict):
            raise BboxAnnotationError(message="category entry must be an object")
        category_id = _required_int(raw_category, "id")
        name = _required_str(raw_category, "name")
        if category_id in categories:
            raise BboxAnnotationError(message=f"duplicate category id: {category_id}")
        if name in names:
            raise BboxAnnotationError(message=f"duplicate category name: {name}")
        categories[category_id] = _Category(id=category_id, name=name)
        names.add(name)
    expected = dict(enumerate(REQUIRED_CATEGORIES, start=1))
    actual = {category.id: category.name for category in categories.values()}
    if actual != expected:
        raise BboxAnnotationError(message="categories must match bbox eval contract")
    return categories


def _read_annotations(payload: dict[str, JsonValue]) -> tuple[_Annotation, ...]:
    raw_annotations = payload.get("annotations")
    if not isinstance(raw_annotations, list):
        raise BboxAnnotationError(message="annotations must be a list")
    annotations: list[_Annotation] = []
    ids: set[int] = set()
    for raw_annotation in raw_annotations:
        if not isinstance(raw_annotation, dict):
            raise BboxAnnotationError(message="annotation entry must be an object")
        annotation_id = _required_int(raw_annotation, "id")
        if annotation_id in ids:
            raise BboxAnnotationError(
                message=f"duplicate annotation id: {annotation_id}",
            )
        bbox = _required_bbox(raw_annotation)
        annotations.append(
            _Annotation(
                id=annotation_id,
                image_id=_required_int(raw_annotation, "image_id"),
                category_id=_required_int(raw_annotation, "category_id"),
                bbox=bbox,
                area=_required_float(raw_annotation, "area"),
                iscrowd=_required_int(raw_annotation, "iscrowd"),
            ),
        )
        ids.add(annotation_id)
    return tuple(annotations)


def _validate_bbox_bounds(annotation: _Annotation, image: _Image) -> None:
    x, y, width, height = annotation.bbox
    if annotation.iscrowd != 0:
        raise BboxAnnotationError(message=f"iscrowd must be 0: {annotation.id}")
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise BboxAnnotationError(message=f"invalid bbox dimensions: {annotation.id}")
    if x + width > image.width or y + height > image.height:
        raise BboxAnnotationError(message=f"bbox out of image bounds: {annotation.id}")
    if abs(annotation.area - (width * height)) > AREA_TOLERANCE:
        raise BboxAnnotationError(message=f"bbox area mismatch: {annotation.id}")


def _required_int(data: dict[str, JsonValue], key: str) -> int:
    value = data.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise BboxAnnotationError(message=f"missing int field: {key}")


def _required_positive_int(data: dict[str, JsonValue], key: str) -> int:
    value = _required_int(data, key)
    if value > 0:
        return value
    raise BboxAnnotationError(message=f"field must be positive: {key}")


def _required_float(data: dict[str, JsonValue], key: str) -> float:
    value = data.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            return numeric_value
    raise BboxAnnotationError(message=f"missing numeric field: {key}")


def _required_str(data: dict[str, JsonValue], key: str) -> str:
    value = data.get(key)
    if isinstance(value, str) and len(value) > 0:
        return value
    raise BboxAnnotationError(message=f"missing string field: {key}")


def _required_bbox(data: dict[str, JsonValue]) -> tuple[float, float, float, float]:
    raw_bbox = data.get("bbox")
    if not isinstance(raw_bbox, list) or len(raw_bbox) != BBOX_COORDINATE_COUNT:
        raise BboxAnnotationError(message="bbox must contain four numbers")
    values: list[float] = []
    for value in raw_bbox:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise BboxAnnotationError(message="bbox values must be numeric")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise BboxAnnotationError(message="bbox values must be finite")
        values.append(numeric_value)
    return (values[0], values[1], values[2], values[3])
