from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from firesight_vision.bbox_annotations import (
    REQUIRED_ANNOTATION_STATUS,
    REQUIRED_HUMAN_REVIEW_STATUS,
)
from firesight_vision.bbox_build import (
    CATEGORIES,
    DraftBox,
    SelectedImage,
)

if TYPE_CHECKING:
    from firesight_vision.bbox_yolo_source import YoloPersonProvenance
    from firesight_vision.image_level_eval_types import JsonValue


@dataclass(slots=True)
class _AnnotationState:
    next_id: int = 1
    annotations: list[dict[str, JsonValue]] = field(default_factory=list)


def write_coco(
    path: Path,
    selected: tuple[SelectedImage, ...],
    boxes: tuple[DraftBox, ...],
) -> None:
    state = _AnnotationState()
    for box in boxes:
        category_id = CATEGORIES.index(box.category) + 1
        x, y, width, height = box.bbox
        state.annotations.append(
            {
                "id": state.next_id,
                "image_id": box.image_id,
                "category_id": category_id,
                "bbox": [x, y, width, height],
                "area": round(width * height, 3),
                "iscrowd": 0,
                "attributes": {
                    "source": box.source,
                    "human_review_status": REQUIRED_HUMAN_REVIEW_STATUS,
                },
            },
        )
        state.next_id += 1
    payload = {
        "info": {
            "annotation_status": REQUIRED_ANNOTATION_STATUS,
            "human_review_status": REQUIRED_HUMAN_REVIEW_STATUS,
            "source": "fire360_photo_inclusive_v0_60",
        },
        "licenses": [],
        "images": [_image_payload(image) for image in selected],
        "categories": [
            {"id": index, "name": name}
            for index, name in enumerate(CATEGORIES, start=1)
        ],
        "annotations": state.annotations,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_manifest(
    path: Path,
    selected: tuple[SelectedImage, ...],
    boxes: tuple[DraftBox, ...],
) -> None:
    labels_by_image: dict[int, set[str]] = {image.id: set() for image in selected}
    for box in boxes:
        labels_by_image[box.image_id].add(box.category)
    payload = {
        "annotation_status": REQUIRED_ANNOTATION_STATUS,
        "human_review_status": REQUIRED_HUMAN_REVIEW_STATUS,
        "classes": list(CATEGORIES),
        "items": [
            {
                "image": image.file_name,
                "source_type": image.source_type,
                "bbox_labels": sorted(labels_by_image[image.id]),
            }
            for image in selected
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_yolo_person_provenance(
    path: Path,
    provenance: tuple[YoloPersonProvenance, ...],
) -> None:
    payload = {
        "annotation_status": REQUIRED_ANNOTATION_STATUS,
        "human_review_status": REQUIRED_HUMAN_REVIEW_STATUS,
        "items": [
            {
                "image_id": item.image_id,
                "image": item.file_name,
                "status": item.status,
                "person_box_count": item.person_box_count,
            }
            for item in provenance
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_yolo_labels(
    out_dir: Path,
    selected: tuple[SelectedImage, ...],
    boxes: tuple[DraftBox, ...],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    boxes_by_image: dict[int, list[DraftBox]] = {image.id: [] for image in selected}
    for box in boxes:
        boxes_by_image[box.image_id].append(box)
    images_by_id = {image.id: image for image in selected}
    for image_id, image_boxes in boxes_by_image.items():
        image = images_by_id[image_id]
        rows = [_to_yolo_row(image, box) for box in image_boxes]
        body = "\n".join(rows) + ("\n" if rows else "")
        _ = (out_dir / f"{Path(image.file_name).stem}.txt").write_text(
            body,
            encoding="utf-8",
        )


def write_yolo_dataset(
    out_dir: Path,
    image_root: Path,
    selected: tuple[SelectedImage, ...],
    boxes: tuple[DraftBox, ...],
) -> None:
    image_dir = out_dir / "images" / "all"
    label_dir = out_dir / "labels" / "all"
    image_dir.mkdir(parents=True, exist_ok=True)
    for image in selected:
        _ = shutil.copy2(image_root / image.file_name, image_dir / image.file_name)
    write_yolo_labels(label_dir, selected, boxes)
    _ = (out_dir / "data.yaml").write_text(_yolo_data_yaml(), encoding="utf-8")


def summary_payload(
    selected: tuple[SelectedImage, ...],
    boxes: tuple[DraftBox, ...],
    coco_path: Path,
) -> dict[str, JsonValue]:
    counts: dict[str, int] = dict.fromkeys(CATEGORIES, 0)
    for box in boxes:
        counts[box.category] += 1
    json_counts: dict[str, JsonValue] = dict(counts)
    payload: dict[str, JsonValue] = {
        "status": REQUIRED_ANNOTATION_STATUS,
        "human_review_status": REQUIRED_HUMAN_REVIEW_STATUS,
        "coco": coco_path.as_posix(),
        "image_count": len(selected),
        "annotation_count": len(boxes),
        "category_counts": json_counts,
        "note": (
            "Draft bbox annotations require human review before training or "
            "reporting model quality."
        ),
    }
    return payload


def _image_payload(image: SelectedImage) -> dict[str, JsonValue]:
    return {
        "id": image.id,
        "file_name": image.file_name,
        "width": image.width,
        "height": image.height,
        "source_type": image.source_type,
    }


def _to_yolo_row(image: SelectedImage, box: DraftBox) -> str:
    x, y, width, height = box.bbox
    class_id = CATEGORIES.index(box.category)
    return (
        f"{class_id} {(x + width / 2) / image.width:.6f} "
        f"{(y + height / 2) / image.height:.6f} "
        f"{width / image.width:.6f} {height / image.height:.6f}"
    )


def _yolo_data_yaml() -> str:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CATEGORIES))
    return (
        "path: .\n"
        "train: images/all\n"
        "val: images/all\n"
        "test: images/all\n"
        f"names:\n{names}\n"
    )
