from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import Sequence


class ComparisonArgNamespace(argparse.Namespace):
    profile_path: Path = Path("data/fire360/video_profiles.json")
    teed_root: Path = Path(
        ".omo/evidence/firesight-smoke-vision/long-videos-source-aware",
    )
    canny_root: Path = Path(
        ".omo/evidence/firesight-smoke-vision/long-videos-canny",
    )
    output_dir: Path = Path(
        ".omo/evidence/firesight-smoke-vision/long-videos-comparison",
    )


@dataclass(frozen=True, slots=True)
class LongVideoComparisonConfig:
    profile_path: Path
    teed_root: Path
    canny_root: Path
    output_dir: Path


class ProfilePayload(TypedDict):
    input_file: str
    ignore_fire360_osd: bool


class _Capture(Protocol):
    def get(self, property_id: int) -> float: ...

    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, object]: ...

    def release(self) -> None: ...


class _Writer(Protocol):
    def isOpened(self) -> bool: ...

    def release(self) -> None: ...

    def write(self, frame: object) -> None: ...


class _Cv2(Protocol):
    CAP_PROP_FRAME_COUNT: int
    CAP_PROP_FPS: int
    CAP_PROP_FRAME_HEIGHT: int
    CAP_PROP_FRAME_WIDTH: int
    FONT_HERSHEY_SIMPLEX: int
    LINE_AA: int

    def VideoCapture(self, path: str) -> _Capture: ...

    def VideoWriter(
        self,
        path: str,
        fourcc: int,
        fps: float,
        size: tuple[int, int],
    ) -> _Writer: ...

    def VideoWriter_fourcc(self, *chars: str) -> int: ...

    def hconcat(self, frames: list[object]) -> object: ...

    def imwrite(self, path: str, image: object) -> bool: ...

    def putText(
        self,
        image: object,
        text: str,
        origin: tuple[int, int],
        font_face: int,
        font_scale: float,
        color: tuple[int, int, int],
        thickness: int,
        line_type: int,
    ) -> None: ...

    def resize(self, image: object, size: tuple[int, int]) -> object: ...


class LongVideoComparisonError(Exception):
    pass


PROTOCOL_NAME: Final = "firesight_long_video_model_comparison_v1"
TEED_SUFFIX: Final = "_teed_long"
TEED_OSD_SUFFIX: Final = "_teed_long_osd_ignored"
OUTPUT_FPS: Final = 5.0
PANEL_LABEL_Y: Final = 32


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _parse_args(argv)
        payload = build_long_video_comparison(config)
    except LongVideoComparisonError as error:
        _ = sys.stderr.write(f"{error}\n")
        return 2
    _ = sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _parse_args(argv: Sequence[str] | None) -> LongVideoComparisonConfig:
    namespace = ComparisonArgNamespace()
    parser = argparse.ArgumentParser(
        description="Compose long-video Canny and TEED side-by-side comparisons.",
    )
    _ = parser.add_argument(
        "--profiles",
        dest="profile_path",
        default=namespace.profile_path,
        type=Path,
    )
    _ = parser.add_argument(
        "--teed-root",
        default=namespace.teed_root,
        type=Path,
    )
    _ = parser.add_argument(
        "--canny-root",
        default=namespace.canny_root,
        type=Path,
    )
    _ = parser.add_argument(
        "--out",
        dest="output_dir",
        default=namespace.output_dir,
        type=Path,
    )
    _ = parser.parse_args(argv, namespace=namespace)
    return LongVideoComparisonConfig(
        profile_path=namespace.profile_path,
        teed_root=namespace.teed_root,
        canny_root=namespace.canny_root,
        output_dir=namespace.output_dir,
    )


def build_long_video_comparison(
    config: LongVideoComparisonConfig,
) -> dict[str, object]:
    cv2 = _load_video_modules()
    profiles = _load_profiles(config.profile_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results = [_compose_profile(profile, config, cv2) for profile in profiles]
    payload: dict[str, object] = {
        "protocol": PROTOCOL_NAME,
        "claim": (
            "Long-video visual comparison at 1280x720 and 5 FPS on PC CPU; "
            "this is not a Jetson measurement or a quality label."
        ),
        "profiles": results,
        "profile_path": config.profile_path.as_posix(),
        "teed_root": config.teed_root.as_posix(),
        "canny_root": config.canny_root.as_posix(),
        "output_dir": config.output_dir.as_posix(),
    }
    summary_path = config.output_dir / "long_video_comparison_summary.json"
    _ = summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _compose_profile(
    profile: ProfilePayload,
    config: LongVideoComparisonConfig,
    cv2: _Cv2,
) -> dict[str, object]:
    input_file = profile["input_file"]
    input_stem = Path(input_file).stem
    teed_suffix = TEED_OSD_SUFFIX if profile["ignore_fire360_osd"] else TEED_SUFFIX
    teed_path = config.teed_root / f"{input_stem}{teed_suffix}.mp4"
    canny_path = config.canny_root / input_file / "overlay.mp4"
    if not teed_path.is_file():
        message = f"missing TEED long video: {teed_path}"
        raise LongVideoComparisonError(message)
    if not canny_path.is_file():
        message = f"missing Canny long video: {canny_path}"
        raise LongVideoComparisonError(message)

    output_stem = Path(input_file).stem
    output_path = config.output_dir / f"{output_stem}_canny_vs_teed.mp4"
    preview_path = config.output_dir / f"{output_stem}_canny_vs_teed_preview.png"
    frames, fps, width, height = _compose_videos(
        canny_path,
        teed_path,
        output_path,
        preview_path,
        cv2,
    )
    return {
        "input_file": input_file,
        "ignore_fire360_osd": profile["ignore_fire360_osd"],
        "canny_video": canny_path.as_posix(),
        "teed_video": teed_path.as_posix(),
        "comparison_video": output_path.as_posix(),
        "representative_preview": preview_path.as_posix(),
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": frames / fps,
    }


def _compose_videos(
    canny_path: Path,
    teed_path: Path,
    output_path: Path,
    preview_path: Path,
    cv2: _Cv2,
) -> tuple[int, float, int, int]:
    canny_capture = cv2.VideoCapture(str(canny_path))
    teed_capture = cv2.VideoCapture(str(teed_path))
    if not canny_capture.isOpened() or not teed_capture.isOpened():
        canny_capture.release()
        teed_capture.release()
        message = f"unable to open comparison inputs: {canny_path}, {teed_path}"
        raise LongVideoComparisonError(message)

    canny_width = round(canny_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    canny_height = round(canny_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    teed_width = round(teed_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    teed_height = round(teed_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = canny_width or teed_width
    height = canny_height or teed_height
    if width <= 0 or height <= 0:
        canny_capture.release()
        teed_capture.release()
        message = f"invalid comparison dimensions: {canny_path}, {teed_path}"
        raise LongVideoComparisonError(message)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        OUTPUT_FPS,
        (width * 2, height),
    )
    if not writer.isOpened():
        canny_capture.release()
        teed_capture.release()
        message = f"unable to create comparison video: {output_path}"
        raise LongVideoComparisonError(message)

    frame_totals = [
        total
        for total in (
            round(canny_capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            round(teed_capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        if total > 0
    ]
    preview_index = min(frame_totals) // 2 if frame_totals else 0
    frame_count = 0
    preview_frame: object | None = None
    representative_frame: object | None = None
    try:
        while True:
            canny_ok, canny_frame = canny_capture.read()
            teed_ok, teed_frame = teed_capture.read()
            if not canny_ok or not teed_ok:
                break
            if canny_width != width or canny_height != height:
                canny_frame = cv2.resize(canny_frame, (width, height))
            if teed_width != width or teed_height != height:
                teed_frame = cv2.resize(teed_frame, (width, height))
            panel = cv2.hconcat([canny_frame, teed_frame])
            _label_panel(panel, width, cv2)
            writer.write(panel)
            preview_frame = panel
            if frame_count == preview_index:
                representative_frame = panel
            frame_count += 1
    finally:
        writer.release()
        canny_capture.release()
        teed_capture.release()

    if frame_count == 0 or preview_frame is None:
        message = f"comparison produced no frames: {output_path}"
        raise LongVideoComparisonError(message)
    preview_frame = _select_preview_frame(representative_frame, preview_frame)
    _write_bgr_png(preview_frame, preview_path, cv2)
    return frame_count, OUTPUT_FPS, width * 2, height


def _select_preview_frame(
    representative_frame: object | None,
    fallback_frame: object,
) -> object:
    if representative_frame is not None:
        return representative_frame
    return fallback_frame


def _label_panel(panel: object, width: int, cv2: _Cv2) -> None:
    for text, origin in (
        ("Canny classical", (12, PANEL_LABEL_Y)),
        ("TEED learned", (width + 12, PANEL_LABEL_Y)),
    ):
        cv2.putText(
            panel,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _write_bgr_png(
    frame: object,
    path: Path,
    cv2: _Cv2,
) -> None:
    if not cv2.imwrite(str(path), frame):
        message = f"unable to write comparison preview: {path}"
        raise LongVideoComparisonError(message)


def _load_video_modules() -> _Cv2:
    try:
        cv2 = cast("_Cv2", cast("object", importlib.import_module("cv2")))
    except ImportError as error:
        message = (
            "long-video comparison needs OpenCV; run with "
            "`uv run --with opencv-python-headless ...`."
        )
        raise LongVideoComparisonError(message) from error
    return cv2


def _load_profiles(path: Path) -> list[ProfilePayload]:
    if not path.is_file():
        message = f"missing profile file: {path}"
        raise LongVideoComparisonError(message)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        message = f"unable to read profile file: {path}"
        raise LongVideoComparisonError(message) from error
    if not isinstance(raw, dict):
        message = f"invalid profile file: {path}"
        raise LongVideoComparisonError(message)
    raw_profiles = raw.get("profiles")
    if not isinstance(raw_profiles, list):
        message = f"invalid profile file: {path}"
        raise LongVideoComparisonError(message)
    profiles: list[ProfilePayload] = []
    for raw_entry in raw_profiles:
        if not isinstance(raw_entry, dict):
            message = f"invalid profile entry: {path}"
            raise LongVideoComparisonError(message)
        entry = cast("dict[str, object]", raw_entry)
        input_file = entry.get("input_file")
        ignore_osd = entry.get("ignore_fire360_osd")
        if not isinstance(input_file, str) or not isinstance(ignore_osd, bool):
            message = f"incomplete profile entry: {path}"
            raise LongVideoComparisonError(message)
        profiles.append(
            ProfilePayload(
                input_file=input_file,
                ignore_fire360_osd=ignore_osd,
            ),
        )
    return profiles


if __name__ == "__main__":
    raise SystemExit(main())
