from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from PIL import Image, ImageDraw, ImageFont, ImageOps

from firesight_vision.hud_edges import FIRE360_OSD_IGNORED_REGIONS
from firesight_vision.pidinet import PidinetError, PidinetPredictor

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class ComparisonArgNamespace(argparse.Namespace):
    input_csv: Path = Path(
        ".omo/evidence/firesight-smoke-vision/model-evaluation-30/evaluation.csv",
    )
    checkpoint_path: Path = Path("models/pidinet/table5_pidinet.pth")
    output_dir: Path = Path()
    pidinet_threshold: float = 0.30
    pidinet_edge_width: int = 1
    background_scale: float = 0.24
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class PidinetComparisonConfig:
    input_csv: Path
    checkpoint_path: Path
    output_dir: Path
    pidinet_threshold: float = 0.30
    pidinet_edge_width: int = 1
    background_scale: float = 0.24
    device: str = "cpu"


class PidinetComparisonError(Exception):
    pass


PROTOCOL_NAME: Final = "fire360_model_comparison_v1"
EXPECTED_FRAME_COUNT: Final = 30
PANEL_SIZE: Final[tuple[int, int]] = (320, 180)
LABEL_HEIGHT: Final[int] = 30
TITLE_HEIGHT: Final[int] = 32
ROWS_PER_PAGE: Final[int] = 5
REQUIRED_INPUT_FIELDS: Final[tuple[str, ...]] = (
    "evaluation_id",
    "video_key",
    "input_file",
    "osd_profile",
    "ignore_fire360_osd",
    "source_frame",
    "original_image",
    "classical_image",
    "teed_image",
)
CSV_FIELDS: Final[tuple[str, ...]] = (
    "evaluation_id",
    "video_key",
    "input_file",
    "osd_profile",
    "ignore_fire360_osd",
    "source_frame",
    "original_image",
    "classical_image",
    "teed_image",
    "pidinet_image",
    "pidinet_probability",
    "pidinet_edge_ratio",
    "teed_score_1_to_5",
    "pidinet_score_1_to_5",
    "preferred_method",
    "teed_noise_flag",
    "teed_breaks_flag",
    "pidinet_noise_flag",
    "pidinet_breaks_flag",
    "frame_flag",
    "notes",
)
GUIDE_TEXT: Final = (
    "# FireSight TEED versus PiDiNet comparison\n"
    "\n"
    "This directory keeps the same deterministic 30 frames as the existing\n"
    "`model-evaluation-30` artifact and adds PiDiNet output. Inspect all four "
    "panels\n"
    "in each contact sheet row: original, Classical, TEED, and PiDiNet.\n"
    "\n"
    "For each row, score TEED and PiDiNet from 1 to 5 for usable outer-contour\n"
    "visibility. A 5 is clear and coherent, 3 is mixed, and 1 is unusable or\n"
    "dominated by false edges. Set `preferred_method` to `teed`, `pidinet`, "
    "`tie`,\n"
    "or `neither`. Set the noise and breaks flags when a result is distracting "
    "or\n"
    "an important contour disappears. Use `frame_flag` and `notes` for follow-up\n"
    "cases.\n"
    "\n"
    "The PiDiNet threshold is a visualization control, not a smoke-domain "
    "accuracy\n"
    "claim. Human labels are still required before declaring a final winner.\n"
)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _parse_args(argv)
        payload = run_pidinet_comparison(config)
    except (PidinetComparisonError, PidinetError) as error:
        _ = sys.stderr.write(f"{error}\n")
        return 2
    _ = sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _parse_args(argv: Sequence[str] | None) -> PidinetComparisonConfig:
    namespace = ComparisonArgNamespace()
    parser = argparse.ArgumentParser(
        description="Add PiDiNet outputs to the fixed FireSight 30-frame comparison.",
    )
    _ = parser.add_argument(
        "--input-csv",
        default=namespace.input_csv,
        type=Path,
    )
    _ = parser.add_argument(
        "--checkpoint",
        dest="checkpoint_path",
        default=namespace.checkpoint_path,
        type=Path,
    )
    _ = parser.add_argument("--out", dest="output_dir", required=True, type=Path)
    _ = parser.add_argument("--pidinet-threshold", default=0.30, type=float)
    _ = parser.add_argument("--pidinet-edge-width", default=1, type=int)
    _ = parser.add_argument("--background-scale", default=0.24, type=float)
    _ = parser.add_argument("--device", default="cpu")
    _ = parser.parse_args(argv, namespace=namespace)
    return PidinetComparisonConfig(
        input_csv=namespace.input_csv,
        checkpoint_path=namespace.checkpoint_path,
        output_dir=namespace.output_dir,
        pidinet_threshold=namespace.pidinet_threshold,
        pidinet_edge_width=namespace.pidinet_edge_width,
        background_scale=namespace.background_scale,
        device=namespace.device,
    )


def run_pidinet_comparison(config: PidinetComparisonConfig) -> dict[str, object]:
    rows = _load_input_rows(config.input_csv)
    _validate_config(config, rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    predictor = PidinetPredictor(config.checkpoint_path, device=config.device)
    comparison_rows = [
        _write_frame_artifacts(row, config, predictor) for row in rows
    ]
    csv_path = config.output_dir / "comparison.csv"
    _write_csv(csv_path, comparison_rows)
    _write_contact_sheets(config.output_dir, comparison_rows)
    guide_path = config.output_dir / "evaluation_guide.md"
    _ = guide_path.write_text(GUIDE_TEXT, encoding="utf-8")
    manifest = {
        "protocol": PROTOCOL_NAME,
        "claim": (
            "Fixed visual comparison only; no smoke-domain accuracy claim until "
            "human ratings are completed."
        ),
        "model_quality": "NOT_ESTABLISHED_WITHOUT_LABELS",
        "input_evaluation_csv": _workspace_relative(config.input_csv),
        "parameters": {
            "pidinet_threshold": config.pidinet_threshold,
            "pidinet_edge_width": config.pidinet_edge_width,
            "background_scale": config.background_scale,
            "device": predictor.device,
            "architecture": "PiDiNet table5, carv4, SA, dilation",
            "checkpoint_path": _workspace_relative(config.checkpoint_path),
            "checkpoint_sha256": _sha256(config.checkpoint_path),
            "upstream_repository": "https://github.com/hellozhuo/pidinet",
        },
        "artifacts": {
            "comparison_csv": _workspace_relative(csv_path),
            "evaluation_guide": _workspace_relative(guide_path),
            "contact_sheet_dir": _workspace_relative(
                config.output_dir / "contact_sheets",
            ),
        },
        "human_evaluation": {
            "status": "PENDING",
            "rows": len(comparison_rows),
            "unrated_rows": len(comparison_rows),
            "teed_mean_score": None,
            "pidinet_mean_score": None,
            "pidinet_win_rate": None,
            "flagged_frames": None,
        },
        "frames": comparison_rows,
    }
    manifest_path = config.output_dir / "comparison_manifest.json"
    _ = manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary: dict[str, object] = {
        "protocol": PROTOCOL_NAME,
        "output_dir": config.output_dir.as_posix(),
        "comparison_csv": csv_path.as_posix(),
        "comparison_manifest": manifest_path.as_posix(),
        "videos": len({row["video_key"] for row in comparison_rows}),
        "frames": len(comparison_rows),
        "model_quality": "NOT_ESTABLISHED_WITHOUT_LABELS",
        "human_evaluation_status": "PENDING",
    }
    _ = (config.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_input_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise PidinetComparisonError(f"input evaluation CSV is missing: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            missing = [field for field in REQUIRED_INPUT_FIELDS if field not in fields]
            if missing:
                raise PidinetComparisonError(
                    f"input evaluation CSV is missing fields: {', '.join(missing)}",
                )
            rows = [
                {key: value or "" for key, value in row.items() if key is not None}
                for row in reader
            ]
    except OSError as error:
        message = f"unable to read input evaluation CSV: {path}"
        raise PidinetComparisonError(message) from error
    if len(rows) != EXPECTED_FRAME_COUNT:
        raise PidinetComparisonError(
            f"expected {EXPECTED_FRAME_COUNT} input rows, got {len(rows)}",
        )
    return rows


def _validate_config(
    config: PidinetComparisonConfig,
    rows: list[dict[str, str]],
) -> None:
    if not 0.0 <= config.pidinet_threshold <= 1.0:
        raise PidinetComparisonError("PiDiNet threshold must be between 0 and 1")
    if config.pidinet_edge_width < 1 or config.pidinet_edge_width > 15:
        raise PidinetComparisonError("PiDiNet edge width must be between 1 and 15")
    if not 0.0 < config.background_scale <= 1.0:
        raise PidinetComparisonError("background scale must be in (0, 1]")
    for row in rows:
        for field in ("original_image", "classical_image", "teed_image"):
            path = _workspace_absolute(row[field])
            if not path.is_file():
                raise PidinetComparisonError(f"input artifact is missing: {path}")


def _write_frame_artifacts(
    row: dict[str, str],
    config: PidinetComparisonConfig,
    predictor: PidinetPredictor,
) -> dict[str, str]:
    frame_dir = config.output_dir / "frames" / row["video_key"]
    frame_dir.mkdir(parents=True, exist_ok=True)
    base_name = row["evaluation_id"]
    pidinet_path = frame_dir / f"{base_name}_pidinet.png"
    probability_path = frame_dir / f"{base_name}_pidinet_probability.png"
    ignored_regions = (
        FIRE360_OSD_IGNORED_REGIONS
        if row["ignore_fire360_osd"].lower() == "true"
        else ()
    )
    try:
        with Image.open(_workspace_absolute(row["original_image"])) as source:
            result = predictor.process(
                source,
                threshold=config.pidinet_threshold,
                background_scale=config.background_scale,
                edge_width=config.pidinet_edge_width,
                ignored_regions=ignored_regions,
            )
    except OSError as error:
        raise PidinetComparisonError(
            f"unable to read original evaluation image: {row['evaluation_id']}"
        ) from error
    result.overlay.save(pidinet_path)
    result.probability.save(probability_path)
    return {
        "evaluation_id": row["evaluation_id"],
        "video_key": row["video_key"],
        "input_file": row["input_file"],
        "osd_profile": row["osd_profile"],
        "ignore_fire360_osd": row["ignore_fire360_osd"],
        "source_frame": row["source_frame"],
        "original_image": row["original_image"],
        "classical_image": row["classical_image"],
        "teed_image": row["teed_image"],
        "pidinet_image": _workspace_relative(pidinet_path),
        "pidinet_probability": _workspace_relative(probability_path),
        "pidinet_edge_ratio": f"{result.edge_ratio:.8f}",
        "teed_score_1_to_5": row.get("teed_score_1_to_5", ""),
        "pidinet_score_1_to_5": "",
        "preferred_method": "",
        "teed_noise_flag": row.get("teed_noise_flag", ""),
        "teed_breaks_flag": row.get("teed_breaks_flag", ""),
        "pidinet_noise_flag": "",
        "pidinet_breaks_flag": "",
        "frame_flag": row.get("frame_flag", ""),
        "notes": row.get("notes", ""),
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_FIELDS)
        for row in rows:
            writer.writerow([row[field] for field in CSV_FIELDS])


def _write_contact_sheets(output_dir: Path, rows: list[dict[str, str]]) -> None:
    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    by_video: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_video.setdefault(row["video_key"], []).append(row)
    for video_key, video_rows in by_video.items():
        _write_contact_sheet(contact_dir / f"{video_key}_comparison.png", video_rows)
    for page_index in range(0, len(rows), ROWS_PER_PAGE):
        _write_contact_sheet(
            contact_dir / f"all_30_page_{page_index // ROWS_PER_PAGE + 1:02d}.png",
            rows[page_index : page_index + ROWS_PER_PAGE],
        )


def _write_contact_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    width = PANEL_SIZE[0] * 4
    height = TITLE_HEIGHT + (PANEL_SIZE[1] + LABEL_HEIGHT) * len(rows)
    sheet = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    draw_text = cast("Callable[..., None]", draw.text)
    font = ImageFont.load_default()
    title = " | ".join(
        sorted({f"{row['video_key']} ({row['osd_profile']})" for row in rows}),
    )
    draw_text(
        (8, 8),
        f"FireSight TEED vs PiDiNet | {title}",
        fill="white",
        font=font,
    )
    columns = (
        ("original", "original_image"),
        ("Classical", "classical_image"),
        ("TEED", "teed_image"),
        ("PiDiNet", "pidinet_image"),
    )
    for row_index, row in enumerate(rows):
        top = TITLE_HEIGHT + row_index * (PANEL_SIZE[1] + LABEL_HEIGHT)
        for column_index, (label, field) in enumerate(columns):
            left = column_index * PANEL_SIZE[0]
            draw_text(
                (left + 6, top + 6),
                f"{row['evaluation_id']} | frame {row['source_frame']} | {label}",
                fill="white",
                font=font,
            )
            with Image.open(_workspace_absolute(row[field])) as image:
                preview = ImageOps.contain(image.convert("RGB"), PANEL_SIZE)
            paste_left = left + (PANEL_SIZE[0] - preview.width) // 2
            paste_top = top + LABEL_HEIGHT + (PANEL_SIZE[1] - preview.height) // 2
            sheet.paste(preview, (paste_left, paste_top))
    sheet.save(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace_relative(path: Path) -> str:
    workspace = Path.cwd().resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(workspace).as_posix()
    except ValueError:
        return resolved.as_posix()


def _workspace_absolute(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else Path.cwd() / path


if __name__ == "__main__":
    raise SystemExit(main())
