import json
from pathlib import Path
from typing import cast

import pytest

from firesight_vision.bbox_annotations import (
    REQUIRED_HUMAN_REVIEW_STATUS,
    validate_coco_bbox_file,
)


def test_validate_coco_bbox_file_accepts_minimal_valid_dataset(
    tmp_path: Path,
) -> None:
    # Given: a tiny COCO-style bbox dataset with one bounded annotation.
    dataset = _write_dataset(tmp_path, _valid_payload())

    # When: the annotation file is validated.
    summary = validate_coco_bbox_file(dataset)

    # Then: image/category/annotation counts are returned.
    assert summary.image_count == 1
    assert summary.annotation_count == 1
    assert summary.category_counts == {
        "door": 0,
        "person": 1,
        "exit": 0,
        "obstacle": 0,
    }


def test_validate_coco_bbox_file_rejects_out_of_bounds_bbox(
    tmp_path: Path,
) -> None:
    # Given: an annotation whose width exceeds the image boundary.
    payload = _valid_payload()
    _first_annotation(payload)["bbox"] = [90, 12, 30, 20]
    dataset = _write_dataset(tmp_path, payload)

    # When/Then: validation fails with a clear bounds message.
    with pytest.raises(ValueError, match="out of image bounds"):
        _ = validate_coco_bbox_file(dataset)


def test_validate_coco_bbox_file_requires_review_status(tmp_path: Path) -> None:
    # Given: a COCO file without explicit human review guardrail metadata.
    payload = _valid_payload()
    payload["info"] = {"annotation_status": "draft_single_agent"}
    dataset = _write_dataset(tmp_path, payload)

    # When/Then: validation rejects it before benchmark use.
    with pytest.raises(ValueError, match="human_review_status"):
        _ = validate_coco_bbox_file(dataset)


def test_validate_coco_bbox_file_rejects_unapproved_categories(tmp_path: Path) -> None:
    # Given: a COCO file that introduces a class outside the eval contract.
    payload = _valid_payload()
    payload["categories"] = [{"id": 1, "name": "fire"}]
    dataset = _write_dataset(tmp_path, payload)

    # When/Then: strict category policy rejects it.
    with pytest.raises(ValueError, match="categories must match"):
        _ = validate_coco_bbox_file(dataset)


def test_validate_coco_bbox_file_rejects_nonzero_iscrowd(tmp_path: Path) -> None:
    # Given: a bbox annotation marked as crowd, which this subset does not use.
    payload = _valid_payload()
    _first_annotation(payload)["iscrowd"] = 1
    dataset = _write_dataset(tmp_path, payload)

    # When/Then: the annotation contract rejects it.
    with pytest.raises(ValueError, match="iscrowd"):
        _ = validate_coco_bbox_file(dataset)


def _valid_payload() -> dict[str, object]:
    return {
        "info": {
            "annotation_status": "draft_single_agent",
            "human_review_status": REQUIRED_HUMAN_REVIEW_STATUS,
        },
        "licenses": [],
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 80}],
        "categories": [
            {"id": 1, "name": "door"},
            {"id": 2, "name": "person"},
            {"id": 3, "name": "exit"},
            {"id": 4, "name": "obstacle"},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 2,
                "bbox": [10, 12, 30, 20],
                "area": 600,
                "iscrowd": 0,
            },
        ],
    }


def _write_dataset(tmp_path: Path, payload: dict[str, object]) -> Path:
    dataset = tmp_path / "dataset.json"
    _ = dataset.write_text(json.dumps(payload), encoding="utf-8")
    return dataset


def _first_annotation(payload: dict[str, object]) -> dict[str, object]:
    annotations = payload["annotations"]
    assert isinstance(annotations, list)
    annotation = cast("dict[str, object]", annotations[0])
    assert isinstance(annotation, dict)
    return annotation
