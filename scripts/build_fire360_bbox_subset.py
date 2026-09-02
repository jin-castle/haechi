# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["Pillow>=10.4,<11"]
# ///
# ----- How to run -----
# uv run python scripts/build_fire360_bbox_subset.py
#   --image-root data/fire360_photo_inclusive/all_images
#   --seed-manifest data/eval/fire360_indoor_image_level_v0.json
#   --yolo-label-dir
#     .omo/evidence/firesight-smoke-vision/fire360-photo-inclusive/yolo11n-mixed/labels
#   --out-coco data/eval/fire360_indoor_bbox_v0_60_coco.json
#   --out-manifest data/eval/fire360_indoor_bbox_v0_60_manifest.json
#   --out-dir outputs/fire360_bbox_60

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from firesight_vision.bbox_build import build_boxes, select_images
from firesight_vision.bbox_export import (
    summary_payload,
    write_coco,
    write_manifest,
    write_yolo_dataset,
    write_yolo_labels,
    write_yolo_person_provenance,
)
from firesight_vision.bbox_render import OverlaySheetRequest, write_overlay_sheet
from firesight_vision.bbox_yolo_source import (
    build_yolo_person_provenance,
    person_boxes_from_yolo,
)
from firesight_vision.yolo_dataset_validation import validate_yolo_dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

    from firesight_vision.image_level_eval_types import JsonValue


class BuildArgNamespace(argparse.Namespace):
    image_root: Path = Path()
    seed_manifest: Path = Path()
    yolo_label_dir: Path = Path()
    out_coco: Path = Path()
    out_manifest: Path = Path()
    out_dir: Path = Path()


def main(argv: Sequence[str] | None = None) -> int:
    namespace = _parse_args(argv)
    selected = select_images(namespace.image_root, namespace.seed_manifest)
    boxes = person_boxes_from_yolo(selected, namespace.yolo_label_dir) + build_boxes(
        selected,
    )
    provenance = build_yolo_person_provenance(selected, namespace.yolo_label_dir)
    write_coco(namespace.out_coco, selected, boxes)
    write_manifest(namespace.out_manifest, selected, boxes)
    write_yolo_labels(namespace.out_dir / "yolo_labels", selected, boxes)
    yolo_dataset_dir = namespace.out_dir / "yolo_dataset"
    write_yolo_dataset(yolo_dataset_dir, namespace.image_root, selected, boxes)
    yolo_validation = validate_yolo_dataset(yolo_dataset_dir, len(selected))
    write_yolo_person_provenance(
        namespace.out_dir / "yolo_person_provenance.json",
        provenance,
    )
    write_overlay_sheet(
        OverlaySheetRequest(
            namespace.out_dir / "bbox_overlay_contact_sheet.png",
            namespace.image_root,
            selected,
            boxes,
        ),
    )
    summary = summary_payload(selected, boxes, namespace.out_coco)
    validation_payload: dict[str, JsonValue] = {
        "image_count": yolo_validation.image_count,
        "label_count": yolo_validation.label_count,
        "row_count": yolo_validation.row_count,
    }
    summary["yolo_dataset_validation"] = validation_payload
    _ = (namespace.out_dir / "yolo_dataset_validation.json").write_text(
        json.dumps(validation_payload, indent=2),
        encoding="utf-8",
    )
    _ = (namespace.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


def _parse_args(argv: Sequence[str] | None) -> BuildArgNamespace:
    parser = argparse.ArgumentParser()
    namespace = BuildArgNamespace()
    _ = parser.add_argument("--image-root", required=True, type=Path)
    _ = parser.add_argument("--seed-manifest", required=True, type=Path)
    _ = parser.add_argument("--yolo-label-dir", required=True, type=Path)
    _ = parser.add_argument("--out-coco", required=True, type=Path)
    _ = parser.add_argument("--out-manifest", required=True, type=Path)
    _ = parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args(argv, namespace=namespace)


if __name__ == "__main__":
    raise SystemExit(main())
