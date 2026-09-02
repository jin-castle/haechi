from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict, cast

from PIL import Image, ImageDraw, ImageFont, ImageOps

from firesight_vision.hud_edges import (
    FIRE360_OSD_IGNORED_REGIONS,
    HudEdgeProfile,
    build_hud_edge_overlay,
)
from firesight_vision.teed import TeedError, TeedPredictor

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class EvaluationArgNamespace(argparse.Namespace):
    profile_path: Path = Path("data/fire360/video_profiles.json")
    multiclip_root: Path = Path("data/fire360/frames_multiclip")
    fallback_root: Path = Path("data/fire360/frames")
    output_dir: Path = Path()
    checkpoint_path: Path = Path()
    analysis_width: int = 640
    classical_threshold: int = 144
    classical_edge_width: int = 3
    teed_threshold: float = 0.75
    teed_edge_width: int = 1
    background_scale: float = 0.24
    device: str = "cpu"


class ProfilePayload(TypedDict):
    input_file: str
    osd_profile: str
    ignore_fire360_osd: bool
    reason: str


class EvaluationRow(TypedDict):
    evaluation_id: str
    video_key: str
    input_file: str
    osd_profile: str
    ignore_fire360_osd: str
    source_frame: str
    candidate_index: str
    candidate_count: str
    source_image: str
    original_image: str
    classical_image: str
    teed_image: str
    teed_probability: str
    classical_score_1_to_5: str
    teed_score_1_to_5: str
    preferred_method: str
    classical_noise_flag: str
    classical_breaks_flag: str
    teed_noise_flag: str
    teed_breaks_flag: str
    frame_flag: str
    notes: str


@dataclass(frozen=True, slots=True)
class ModelEvaluationConfig:
    profile_path: Path
    multiclip_root: Path
    fallback_root: Path
    output_dir: Path
    checkpoint_path: Path
    analysis_width: int = 640
    classical_threshold: int = 144
    classical_edge_width: int = 3
    teed_threshold: float = 0.75
    teed_edge_width: int = 1
    background_scale: float = 0.24
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class CandidateFrame:
    path: Path
    source_frame: int


@dataclass(frozen=True, slots=True)
class SelectedFrame:
    evaluation_id: str
    video_key: str
    profile: ProfilePayload
    candidate: CandidateFrame
    candidate_index: int
    candidate_count: int
    selected_rank: int


class ModelEvaluationError(Exception):
    pass


PROTOCOL_NAME: Final = "fire360_model_evaluation_v1"
PROFILE_PROTOCOL: Final = "fire360_video_profile_v1"
EXPECTED_VIDEO_COUNT: Final = 7
BASE_FRAMES_PER_VIDEO: Final = 4
EXTRA_FRAME_VIDEO_KEYS: Final = frozenset({"ifsi_video_4", "ifsi_video_8"})
SOURCE_FRAME_PATTERN: Final = re.compile(r"frame_(\d+)\.(?:jpg|jpeg|png)$")
CSV_FIELDS: Final[tuple[str, ...]] = (
    "evaluation_id",
    "video_key",
    "input_file",
    "osd_profile",
    "ignore_fire360_osd",
    "source_frame",
    "candidate_index",
    "candidate_count",
    "source_image",
    "original_image",
    "classical_image",
    "teed_image",
    "teed_probability",
    "classical_score_1_to_5",
    "teed_score_1_to_5",
    "preferred_method",
    "classical_noise_flag",
    "classical_breaks_flag",
    "teed_noise_flag",
    "teed_breaks_flag",
    "frame_flag",
    "notes",
)
PANEL_SIZE: Final[tuple[int, int]] = (400, 225)
LABEL_HEIGHT: Final[int] = 34
TITLE_HEIGHT: Final[int] = 34
ROWS_PER_PAGE: Final[int] = 6
EVALUATION_GUIDE: Final[str] = (
    "# FireSight 30-frame model evaluation\n"
    "\n"
    "`evaluation.csv` is the fixed human-rating table. The 30 rows are selected\n"
    "deterministically from the existing source-aware candidate frames and should\n"
    "not be reordered, removed, or replaced between Classical and TEED reviews.\n"
    "\n"
    "For every row, inspect the original, Classical, and TEED panels together.\n"
    "Score each method from 1 to 5 for **usable outer-contour visibility**:\n"
    "\n"
    "- `5`: contour is clear, coherent, and useful for navigation or scene reading.\n"
    "- `4`: mostly useful; small gaps or extra edges do not change the reading.\n"
    "- `3`: mixed; some useful structure but important ambiguity remains.\n"
    "- `2`: weak or fragmented; only a small part of the contour is useful.\n"
    "- `1`: unusable or dominated by false edges.\n"
    "\n"
    "Use `preferred_method` as `classical`, `teed`, `tie`, or `neither`.\n"
    "Set a method's `*_noise_flag` when background, smoke texture, or fixed OSD\n"
    "creates distracting false edges. Set `*_breaks_flag` when an important\n"
    "contour is visibly interrupted or disappears. Use `frame_flag` for a frame\n"
    "that deserves follow-up, and put the reason in `notes`.\n"
    "\n"
    "Do not score edge density or brightness by itself. The current model-quality\n"
    "claim remains `NOT_ESTABLISHED_WITHOUT_LABELS` until these fields are filled\n"
    "and summarized.\n"
)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _parse_args(argv)
        payload = run_model_evaluation(config)
    except (ModelEvaluationError, TeedError) as error:
        _ = sys.stderr.write(f"{error}\n")
        return 2

    _ = sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _parse_args(argv: Sequence[str] | None) -> ModelEvaluationConfig:
    namespace = EvaluationArgNamespace()
    parser = argparse.ArgumentParser(
        description=(
            "Build a fixed 30-frame original/Classical/TEED Fire360 evaluation."
        ),
    )
    _ = parser.add_argument(
        "--profiles",
        dest="profile_path",
        default=namespace.profile_path,
        type=Path,
    )
    _ = parser.add_argument(
        "--multiclip-root",
        default=namespace.multiclip_root,
        type=Path,
        help="primary root containing the existing 9-10 frame video samples",
    )
    _ = parser.add_argument(
        "--fallback-root",
        default=namespace.fallback_root,
        type=Path,
        help="fallback root used for candidate videos absent from multiclip",
    )
    _ = parser.add_argument("--out", dest="output_dir", required=True, type=Path)
    _ = parser.add_argument(
        "--checkpoint",
        dest="checkpoint_path",
        required=True,
        type=Path,
    )
    _ = parser.add_argument("--analysis-width", default=640, type=int)
    _ = parser.add_argument("--classical-threshold", default=144, type=int)
    _ = parser.add_argument("--classical-edge-width", default=3, type=int)
    _ = parser.add_argument("--teed-threshold", default=0.75, type=float)
    _ = parser.add_argument("--teed-edge-width", default=1, type=int)
    _ = parser.add_argument("--background-scale", default=0.24, type=float)
    _ = parser.add_argument("--device", default="cpu")
    _ = parser.parse_args(argv, namespace=namespace)
    return ModelEvaluationConfig(
        profile_path=namespace.profile_path,
        multiclip_root=namespace.multiclip_root,
        fallback_root=namespace.fallback_root,
        output_dir=namespace.output_dir,
        checkpoint_path=namespace.checkpoint_path,
        analysis_width=namespace.analysis_width,
        classical_threshold=namespace.classical_threshold,
        classical_edge_width=namespace.classical_edge_width,
        teed_threshold=namespace.teed_threshold,
        teed_edge_width=namespace.teed_edge_width,
        background_scale=namespace.background_scale,
        device=namespace.device,
    )


def run_model_evaluation(config: ModelEvaluationConfig) -> dict[str, object]:
    _validate_config(config)
    profiles = _load_profiles(config.profile_path)
    if len(profiles) != EXPECTED_VIDEO_COUNT:
        raise ModelEvaluationError(
            f"expected {EXPECTED_VIDEO_COUNT} profiles, got {len(profiles)}",
        )

    candidates_by_key = {
        video_key(profile["input_file"]): _load_candidates(
            profile["input_file"],
            config,
        )
        for profile in profiles
    }
    selected = select_frames(profiles, candidates_by_key)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    predictor = TeedPredictor(config.checkpoint_path, device=config.device)

    rows: list[EvaluationRow] = []
    selected_by_video: dict[str, list[EvaluationRow]] = {}
    for frame in selected:
        row = _write_frame_artifacts(frame, config, predictor)
        rows.append(row)
        selected_by_video.setdefault(frame.video_key, []).append(row)

    csv_path = config.output_dir / "evaluation.csv"
    _write_csv(csv_path, rows)
    _write_contact_sheets(config.output_dir, rows)
    guide_path = config.output_dir / "evaluation_guide.md"
    _ = guide_path.write_text(EVALUATION_GUIDE, encoding="utf-8")

    manifest = {
        "protocol": PROTOCOL_NAME,
        "profile_protocol": PROFILE_PROTOCOL,
        "claim": (
            "Fixed visual comparison artifact only; no smoke-domain accuracy "
            "claim until human ratings are completed."
        ),
        "model_quality": "NOT_ESTABLISHED_WITHOUT_LABELS",
        "selection": {
            "rule": (
                "Four evenly spaced candidate frames per video; one additional "
                "frame for each of the two longest source clips (IFSI Video 4/8)."
            ),
            "base_frames_per_video": BASE_FRAMES_PER_VIDEO,
            "extra_video_keys": sorted(EXTRA_FRAME_VIDEO_KEYS),
            "total_frames": len(selected),
            "candidate_roots": [
                config.multiclip_root.as_posix(),
                config.fallback_root.as_posix(),
            ],
        },
        "parameters": {
            "analysis_width": config.analysis_width,
            "classical_threshold": config.classical_threshold,
            "classical_edge_width": config.classical_edge_width,
            "classical_profile": HudEdgeProfile.DENSE_SMOKE.value,
            "teed_threshold": config.teed_threshold,
            "teed_edge_width": config.teed_edge_width,
            "background_scale": config.background_scale,
            "checkpoint_path": config.checkpoint_path.as_posix(),
            "device": predictor.device,
        },
        "artifacts": {
            "evaluation_csv": _workspace_relative(csv_path),
            "evaluation_guide": _workspace_relative(guide_path),
            "contact_sheet_dir": _workspace_relative(
                config.output_dir / "contact_sheets",
            ),
        },
        "human_evaluation": {
            "status": "PENDING",
            "rows": len(rows),
            "unrated_rows": len(rows),
            "classical_mean_score": None,
            "teed_mean_score": None,
            "teed_win_rate": None,
            "flagged_frames": None,
        },
        "frames": rows,
    }
    manifest_path = config.output_dir / "evaluation_manifest.json"
    _ = manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary: dict[str, object] = {
        "protocol": PROTOCOL_NAME,
        "output_dir": config.output_dir.as_posix(),
        "evaluation_csv": csv_path.as_posix(),
        "evaluation_manifest": manifest_path.as_posix(),
        "videos": len(selected_by_video),
        "frames": len(rows),
        "model_quality": "NOT_ESTABLISHED_WITHOUT_LABELS",
        "human_evaluation_status": "PENDING",
    }
    _ = (config.output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _validate_config(config: ModelEvaluationConfig) -> None:
    if config.analysis_width < 64:
        raise ModelEvaluationError("analysis width must be at least 64")
    if config.classical_threshold < 0 or config.classical_threshold > 255:
        raise ModelEvaluationError("classical threshold must be between 0 and 255")
    if config.classical_edge_width < 1 or config.classical_edge_width > 15:
        raise ModelEvaluationError("classical edge width must be between 1 and 15")
    if (
        not math.isfinite(config.teed_threshold)
        or not 0.0 <= config.teed_threshold <= 1.0
    ):
        raise ModelEvaluationError("TEED threshold must be between 0 and 1")
    if config.teed_edge_width < 1 or config.teed_edge_width > 15:
        raise ModelEvaluationError("TEED edge width must be between 1 and 15")
    if (
        not math.isfinite(config.background_scale)
        or not 0.0 < config.background_scale <= 1.0
    ):
        raise ModelEvaluationError("background scale must be in (0, 1]")


def _load_profiles(profile_path: Path) -> list[ProfilePayload]:
    if not profile_path.is_file():
        raise ModelEvaluationError(f"profile file is missing: {profile_path}")
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelEvaluationError(
            f"unable to read profile file: {profile_path}"
        ) from error
    if not isinstance(payload, dict) or payload.get("protocol") != PROFILE_PROTOCOL:
        raise ModelEvaluationError(f"unsupported profile protocol: {profile_path}")
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list):
        raise ModelEvaluationError(f"profile file has no profiles: {profile_path}")

    profiles: list[ProfilePayload] = []
    for raw_profile in raw_profiles:
        if not isinstance(raw_profile, dict):
            raise ModelEvaluationError(f"invalid profile entry: {profile_path}")
        required_fields = (
            "input_file",
            "osd_profile",
            "ignore_fire360_osd",
            "reason",
        )
        if any(field not in raw_profile for field in required_fields):
            raise ModelEvaluationError(f"incomplete profile entry: {profile_path}")
        profiles.append(
            ProfilePayload(
                input_file=str(raw_profile["input_file"]),
                osd_profile=str(raw_profile["osd_profile"]),
                ignore_fire360_osd=bool(raw_profile["ignore_fire360_osd"]),
                reason=str(raw_profile["reason"]),
            ),
        )
    return profiles


def video_key(input_file: str) -> str:
    keys = {
        "02814 (2).MTS": "clip_02814",
        "02815 (2).MTS": "clip_02815",
        "02824 (2).MTS": "clip_02824",
        "GOPR8356 (2).MP4": "gopr8356",
        "IFSI Video 4 (2).mp4": "ifsi_video_4",
        "IFSI Video 8 (2).mp4": "ifsi_video_8",
        "sample_3.MP4": "sample_3",
    }
    try:
        return keys[input_file]
    except KeyError as error:
        raise ModelEvaluationError(
            f"unsupported Fire360 input profile: {input_file}"
        ) from error


def _load_candidates(
    input_file: str, config: ModelEvaluationConfig
) -> tuple[CandidateFrame, ...]:
    key = video_key(input_file)
    primary = config.multiclip_root / key
    fallback = config.fallback_root / key
    candidate_dir = primary if primary.is_dir() else fallback
    if not candidate_dir.is_dir():
        message = (
            f"candidate frame directory is missing for {input_file}: "
            f"{primary} or {fallback}"
        )
        raise ModelEvaluationError(message)
    paths = sorted(
        {
            path
            for suffix in ("*.jpg", "*.jpeg", "*.png")
            for path in candidate_dir.glob(suffix)
        },
        key=lambda path: (_source_frame_from_path(path), path.name),
    )
    if len(paths) < BASE_FRAMES_PER_VIDEO:
        raise ModelEvaluationError(
            f"not enough candidate frames for {input_file}: {len(paths)}",
        )
    return tuple(
        CandidateFrame(path=path, source_frame=_source_frame_from_path(path))
        for path in paths
    )


def _source_frame_from_path(path: Path) -> int:
    match = SOURCE_FRAME_PATTERN.search(path.name)
    if match is None:
        raise ModelEvaluationError(f"candidate filename has no source frame: {path}")
    return int(match.group(1))


def select_frames(
    profiles: list[ProfilePayload],
    candidates_by_key: dict[str, tuple[CandidateFrame, ...]],
) -> tuple[SelectedFrame, ...]:
    selected: list[SelectedFrame] = []
    for profile in profiles:
        key = video_key(profile["input_file"])
        candidates = candidates_by_key[key]
        requested = BASE_FRAMES_PER_VIDEO + int(key in EXTRA_FRAME_VIDEO_KEYS)
        indices = evenly_spaced_indices(len(candidates), requested)
        for rank, candidate_index in enumerate(indices, start=1):
            selected.append(
                SelectedFrame(
                    evaluation_id=f"F{len(selected) + 1:03d}",
                    video_key=key,
                    profile=profile,
                    candidate=candidates[candidate_index],
                    candidate_index=candidate_index,
                    candidate_count=len(candidates),
                    selected_rank=rank,
                ),
            )
    if len(selected) != 30:
        raise ModelEvaluationError(
            f"selection produced {len(selected)} frames, expected 30"
        )
    return tuple(selected)


def evenly_spaced_indices(available: int, requested: int) -> tuple[int, ...]:
    if available < 1 or requested < 1 or requested > available:
        raise ModelEvaluationError(
            f"invalid frame selection: available={available}, requested={requested}",
        )
    if requested == 1:
        return (0,)
    return tuple(
        math.floor(index * (available - 1) / (requested - 1) + 0.5)
        for index in range(requested)
    )


def _write_frame_artifacts(
    frame: SelectedFrame,
    config: ModelEvaluationConfig,
    predictor: TeedPredictor,
) -> EvaluationRow:
    frame_dir = config.output_dir / "frames" / frame.video_key
    frame_dir.mkdir(parents=True, exist_ok=True)
    base_name = frame.evaluation_id
    original_path = frame_dir / f"{base_name}_original.png"
    classical_path = frame_dir / f"{base_name}_classical.png"
    teed_path = frame_dir / f"{base_name}_teed.png"
    probability_path = frame_dir / f"{base_name}_teed_probability.png"
    ignored_regions = (
        FIRE360_OSD_IGNORED_REGIONS if frame.profile["ignore_fire360_osd"] else ()
    )

    try:
        with Image.open(frame.candidate.path) as source:
            image = _resize_for_analysis(source.convert("RGB"), config.analysis_width)
    except OSError as error:
        raise ModelEvaluationError(
            f"unable to read candidate frame: {frame.candidate.path}"
        ) from error

    classical = build_hud_edge_overlay(
        image,
        threshold=config.classical_threshold,
        background_scale=config.background_scale,
        edge_width=config.classical_edge_width,
        profile=HudEdgeProfile.DENSE_SMOKE,
        ignored_regions=ignored_regions,
    )
    teed = predictor.process(
        image,
        threshold=config.teed_threshold,
        background_scale=config.background_scale,
        edge_width=config.teed_edge_width,
        ignored_regions=ignored_regions,
    )
    image.save(original_path)
    classical.overlay.save(classical_path)
    teed.overlay.save(teed_path)
    teed.probability.save(probability_path)

    return {
        "evaluation_id": frame.evaluation_id,
        "video_key": frame.video_key,
        "input_file": frame.profile["input_file"],
        "osd_profile": frame.profile["osd_profile"],
        "ignore_fire360_osd": str(frame.profile["ignore_fire360_osd"]).lower(),
        "source_frame": str(frame.candidate.source_frame),
        "candidate_index": str(frame.candidate_index),
        "candidate_count": str(frame.candidate_count),
        "source_image": _workspace_relative(frame.candidate.path),
        "original_image": _workspace_relative(original_path),
        "classical_image": _workspace_relative(classical_path),
        "teed_image": _workspace_relative(teed_path),
        "teed_probability": _workspace_relative(probability_path),
        "classical_score_1_to_5": "",
        "teed_score_1_to_5": "",
        "preferred_method": "",
        "classical_noise_flag": "",
        "classical_breaks_flag": "",
        "teed_noise_flag": "",
        "teed_breaks_flag": "",
        "frame_flag": "",
        "notes": "",
    }


def _resize_for_analysis(image: Image.Image, target_width: int) -> Image.Image:
    if image.width <= target_width:
        return image.copy()
    target_height = round(image.height * target_width / image.width)
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _write_csv(path: Path, rows: list[EvaluationRow]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_FIELDS)
        for row in rows:
            row_values = cast("dict[str, str]", dict(row))
            writer.writerow([row_values[field] for field in CSV_FIELDS])


def _write_contact_sheets(output_dir: Path, rows: list[EvaluationRow]) -> None:
    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    by_video: dict[str, list[EvaluationRow]] = {}
    for row in rows:
        by_video.setdefault(row["video_key"], []).append(row)
    for video_key, video_rows in by_video.items():
        _write_contact_sheet(contact_dir / f"{video_key}_comparison.png", video_rows)
    for page_index in range(0, len(rows), ROWS_PER_PAGE):
        page_rows = rows[page_index : page_index + ROWS_PER_PAGE]
        _write_contact_sheet(
            contact_dir / f"all_30_page_{page_index // ROWS_PER_PAGE + 1:02d}.png",
            page_rows,
        )


def _write_contact_sheet(path: Path, rows: list[EvaluationRow]) -> None:
    width = PANEL_SIZE[0] * 3
    height = TITLE_HEIGHT + (PANEL_SIZE[1] + LABEL_HEIGHT) * len(rows)
    sheet = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    draw_text = cast("Callable[..., None]", draw.text)
    font = ImageFont.load_default()
    title = " | ".join(
        sorted({f"{row['video_key']} ({row['osd_profile']})" for row in rows}),
    )
    draw_text((8, 8), f"FireSight evaluation | {title}", fill="white", font=font)
    columns = (
        ("original", "original_image"),
        ("Classical", "classical_image"),
        ("TEED", "teed_image"),
    )
    for row_index, row in enumerate(rows):
        top = TITLE_HEIGHT + row_index * (PANEL_SIZE[1] + LABEL_HEIGHT)
        row_values = cast("dict[str, str]", dict(row))
        for column_index, (label, field) in enumerate(columns):
            left = column_index * PANEL_SIZE[0]
            draw_text(
                (left + 8, top + 7),
                f"{row['evaluation_id']} | frame {row['source_frame']} | {label}",
                fill="white",
                font=font,
            )
            image = Image.open(_workspace_absolute(row_values[field]))
            try:
                preview = ImageOps.contain(image.convert("RGB"), PANEL_SIZE)
            finally:
                image.close()
            paste_left = left + (PANEL_SIZE[0] - preview.width) // 2
            paste_top = top + LABEL_HEIGHT + (PANEL_SIZE[1] - preview.height) // 2
            sheet.paste(preview, (paste_left, paste_top))
    sheet.save(path)


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
