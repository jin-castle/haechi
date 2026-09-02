from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from firesight_vision.bbox_build import DraftBox, SelectedImage

COLORS = {
    "door": (0, 255, 0),
    "person": (255, 40, 40),
    "exit": (0, 180, 255),
    "obstacle": (255, 190, 0),
}


@dataclass(frozen=True, slots=True)
class OverlaySheetRequest:
    path: Path
    image_root: Path
    selected: tuple[SelectedImage, ...]
    boxes: tuple[DraftBox, ...]


@dataclass(frozen=True, slots=True)
class TileSpec:
    width: int = 220
    height: int = 132


@dataclass(frozen=True, slots=True)
class TileOverlayRequest:
    path: Path
    image: SelectedImage
    boxes: tuple[DraftBox, ...]
    spec: TileSpec


@dataclass(frozen=True, slots=True)
class DrawTransform:
    scale: float
    offset_x: int
    offset_y: int


def write_overlay_sheet(request: OverlaySheetRequest) -> None:
    boxes_by_image: dict[int, list[DraftBox]] = {
        image.id: [] for image in request.selected
    }
    for box in request.boxes:
        boxes_by_image[box.image_id].append(box)
    cols, caption_h, pad = 5, 44, 12
    spec = TileSpec()
    rows = (len(request.selected) + cols - 1) // cols
    sheet = Image.new(
        "RGB",
        (cols * (spec.width + pad) + pad, rows * (spec.height + caption_h + pad) + pad),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw_text = cast("Callable[..., None]", draw.text)
    font = ImageFont.load_default()
    for index, image in enumerate(request.selected):
        x0 = pad + (index % cols) * (spec.width + pad)
        y0 = pad + (index // cols) * (spec.height + caption_h + pad)
        tile = _overlay_image(
            TileOverlayRequest(
                request.image_root / image.file_name,
                image,
                tuple(boxes_by_image[image.id]),
                spec,
            ),
        )
        sheet.paste(tile, (x0, y0))
        labels = sorted({box.category for box in boxes_by_image[image.id]})
        draw_text(
            (x0, y0 + spec.height + 4),
            f"{index + 1:02d} {image.file_name[:28]}",
            fill=(20, 20, 20),
            font=font,
        )
        draw_text(
            (x0, y0 + spec.height + 20),
            ",".join(labels) if labels else "-",
            fill=(20, 20, 20),
            font=font,
        )
    request.path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(request.path)


def _overlay_image(request: TileOverlayRequest) -> Image.Image:
    base = Image.open(request.path).convert("RGB")
    scale = min(
        request.spec.width / request.image.width,
        request.spec.height / request.image.height,
    )
    resized = base.resize(
        (round(request.image.width * scale), round(request.image.height * scale)),
    )
    tile = Image.new("RGB", (request.spec.width, request.spec.height), (15, 16, 18))
    transform = DrawTransform(
        scale,
        (request.spec.width - resized.width) // 2,
        (request.spec.height - resized.height) // 2,
    )
    tile.paste(resized, (transform.offset_x, transform.offset_y))
    draw = ImageDraw.Draw(tile)
    for box in request.boxes:
        _draw_box(draw, box, transform)
    return tile


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: DraftBox,
    transform: DrawTransform,
) -> None:
    x, y, box_w, box_h = box.bbox
    left = transform.offset_x + x * transform.scale
    top = transform.offset_y + y * transform.scale
    right = left + box_w * transform.scale
    bottom = top + box_h * transform.scale
    draw.rectangle((left, top, right, bottom), outline=COLORS[box.category], width=2)
    draw_text = cast("Callable[..., None]", draw.text)
    draw_text((left + 2, top + 2), box.category, fill=COLORS[box.category])
