from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class ComparisonArgNamespace(argparse.Namespace):
    input_path: Path = Path()
    baseline_path: Path = Path()
    teed_path: Path = Path()
    out_path: Path = Path()
    indices: ClassVar[list[int]] = [0, 3, 5, 9]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an original/classical/TEED comparison contact sheet.",
    )
    _ = parser.add_argument("--input", dest="input_path", type=Path, required=True)
    _ = parser.add_argument(
        "--baseline",
        dest="baseline_path",
        type=Path,
        required=True,
    )
    _ = parser.add_argument("--teed", dest="teed_path", type=Path, required=True)
    _ = parser.add_argument("--out", dest="out_path", type=Path, required=True)
    _ = parser.add_argument("--indices", type=int, nargs="+", default=[0, 3, 5, 9])
    args = ComparisonArgNamespace()
    _ = parser.parse_args(argv, namespace=args)

    input_frames = sorted(args.input_path.glob("*.jpg"))
    baseline_frames = sorted(args.baseline_path.glob("*_hud_edges.png"))
    teed_frames = sorted(args.teed_path.glob("*_teed_edges.png"))
    if not input_frames or not baseline_frames or not teed_frames:
        parser.error("input, baseline, and TEED frame directories must be non-empty")

    panel_width = 480
    panel_height = 270
    label_height = 28
    row_height = panel_height + label_height
    rows: list[tuple[int, Path, Path, Path]] = []
    frame_count = min(len(input_frames), len(baseline_frames), len(teed_frames))
    for index in args.indices:
        if index < 0 or index >= frame_count:
            parser.error(f"frame index out of range: {index}")
        rows.append(
            (index, input_frames[index], baseline_frames[index], teed_frames[index]),
        )

    sheet = Image.new(
        "RGB",
        (panel_width * 3, row_height * len(rows)),
        (18, 18, 18),
    )
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(sheet)
    draw_text = cast("Callable[..., None]", draw.text)
    font = ImageFont.load_default()
    for row_index, (index, input_path, baseline_path, teed_path) in enumerate(rows):
        top = row_index * row_height
        for column, (label, image_path) in enumerate(
            (
                ("original", input_path),
                ("classical dense-smoke", baseline_path),
                ("TEED threshold=0.75", teed_path),
            ),
        ):
            left = column * panel_width
            draw_text(
                (left + 8, top + 7),
                f"frame {index:02d} | {label}",
                fill="white",
                font=font,
            )
            with Image.open(image_path) as image:
                preview = image.convert("RGB").resize((panel_width, panel_height))
            sheet.paste(preview, (left, top + label_height))

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
