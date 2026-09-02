from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict

from firesight_vision.hud_edges import FIRE360_OSD_IGNORED_REGIONS
from firesight_vision.teed import TeedError
from scripts.run_teed_video_replay import (
    TeedVideoReplayConfig,
    TeedVideoReplayError,
    TeedVideoSummaryPayload,
    run_teed_video_replay,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class ProfileRunnerArgNamespace(argparse.Namespace):
    input_root: Path = Path("data/fire360/raw")
    profile_path: Path = Path("data/fire360/video_profiles.json")
    output_dir: Path = Path()
    checkpoint_path: Path = Path()
    inference_fps: float = 5.0
    output_width: int = 1280
    threshold: float = 0.75
    background_scale: float = 0.24
    edge_width: int = 1
    device: str = "auto"


class ProfilePayload(TypedDict):
    input_file: str
    osd_profile: str
    ignore_fire360_osd: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ProfiledReplayConfig:
    input_root: Path
    profile_path: Path
    output_dir: Path
    checkpoint_path: Path
    inference_fps: float = 5.0
    output_width: int = 1280
    threshold: float = 0.75
    background_scale: float = 0.24
    edge_width: int = 1
    device: str = "auto"


class ProfiledReplayError(Exception):
    pass


PROTOCOL_NAME: Final = "fire360_profiled_teed_replay_v1"
PROFILE_PROTOCOL: Final = "fire360_video_profile_v1"


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _parse_args(argv)
        payload = run_profiled_replays(config)
    except (ProfiledReplayError, TeedError, TeedVideoReplayError) as error:
        _ = sys.stderr.write(f"{error}\n")
        return 2

    _ = sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _parse_args(argv: Sequence[str] | None) -> ProfiledReplayConfig:
    namespace = ProfileRunnerArgNamespace()
    parser = argparse.ArgumentParser(
        description=(
            "Run source-aware TEED replay using explicit Fire360 video profiles."
        ),
    )
    _ = parser.add_argument(
        "--input-root",
        default=namespace.input_root,
        type=Path,
        help="directory containing the raw videos named by the profile file",
    )
    _ = parser.add_argument(
        "--profiles",
        dest="profile_path",
        default=namespace.profile_path,
        type=Path,
    )
    _ = parser.add_argument("--out", dest="output_dir", required=True, type=Path)
    _ = parser.add_argument(
        "--checkpoint",
        dest="checkpoint_path",
        required=True,
        type=Path,
    )
    _ = parser.add_argument("--inference-fps", default=5.0, type=float)
    _ = parser.add_argument("--output-width", default=1280, type=int)
    _ = parser.add_argument("--threshold", default=0.75, type=float)
    _ = parser.add_argument("--background-scale", default=0.24, type=float)
    _ = parser.add_argument("--edge-width", default=1, type=int)
    _ = parser.add_argument("--device", default="auto")
    _ = parser.parse_args(argv, namespace=namespace)
    return ProfiledReplayConfig(
        input_root=namespace.input_root,
        profile_path=namespace.profile_path,
        output_dir=namespace.output_dir,
        checkpoint_path=namespace.checkpoint_path,
        inference_fps=namespace.inference_fps,
        output_width=namespace.output_width,
        threshold=namespace.threshold,
        background_scale=namespace.background_scale,
        edge_width=namespace.edge_width,
        device=namespace.device,
    )


def run_profiled_replays(config: ProfiledReplayConfig) -> dict[str, object]:
    profiles = _load_profiles(config.profile_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[TeedVideoSummaryPayload] = []
    profile_results: list[dict[str, object]] = []
    seen_inputs: set[str] = set()
    for profile in profiles:
        input_file = profile["input_file"]
        if input_file in seen_inputs:
            raise ProfiledReplayError(f"duplicate profile input: {input_file}")
        seen_inputs.add(input_file)
        input_path = config.input_root / input_file
        if not input_path.is_file():
            raise ProfiledReplayError(f"profile input is missing: {input_path}")
        replay_config = TeedVideoReplayConfig(
            input_path=input_path,
            output_dir=config.output_dir,
            checkpoint_path=config.checkpoint_path,
            inference_fps=config.inference_fps,
            output_width=config.output_width,
            threshold=config.threshold,
            background_scale=config.background_scale,
            edge_width=config.edge_width,
            device=config.device,
            ignored_regions=(
                FIRE360_OSD_IGNORED_REGIONS
                if profile["ignore_fire360_osd"]
                else ()
            ),
        )
        replay_summaries = run_teed_video_replay(replay_config)
        if len(replay_summaries) != 1:
            raise ProfiledReplayError(
                f"profile input produced {len(replay_summaries)} outputs: {input_path}",
            )
        summary = replay_summaries[0]
        summaries.append(summary)
        profile_results.append(
            {
                **profile,
                "output_file": summary["output_file"],
                "ignored_regions": summary["ignored_regions"],
            },
        )
    payload: dict[str, object] = {
        "protocol": PROTOCOL_NAME,
        "profile_protocol": PROFILE_PROTOCOL,
        "profile_file": config.profile_path.as_posix(),
        "input_root": config.input_root.as_posix(),
        "output_dir": config.output_dir.as_posix(),
        "profiles": profile_results,
        "summaries": summaries,
    }
    _ = (config.output_dir / "profiled_replay_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _load_profiles(profile_path: Path) -> list[ProfilePayload]:
    if not profile_path.is_file():
        raise ProfiledReplayError(f"profile file is missing: {profile_path}")
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProfiledReplayError(
            f"unable to read profile file: {profile_path}",
        ) from error
    if not isinstance(payload, dict) or payload.get("protocol") != PROFILE_PROTOCOL:
        raise ProfiledReplayError(f"unsupported profile protocol: {profile_path}")
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list) or len(raw_profiles) == 0:
        raise ProfiledReplayError(f"profile file has no profiles: {profile_path}")
    profiles: list[ProfilePayload] = []
    for raw_profile in raw_profiles:
        if not isinstance(raw_profile, dict):
            raise ProfiledReplayError(f"invalid profile entry: {profile_path}")
        required_fields = (
            "input_file",
            "osd_profile",
            "ignore_fire360_osd",
            "reason",
        )
        if any(field not in raw_profile for field in required_fields):
            raise ProfiledReplayError(f"incomplete profile entry: {profile_path}")
        profiles.append(
            ProfilePayload(
                input_file=str(raw_profile["input_file"]),
                osd_profile=str(raw_profile["osd_profile"]),
                ignore_fire360_osd=bool(raw_profile["ignore_fire360_osd"]),
                reason=str(raw_profile["reason"]),
            ),
        )
    return profiles


if __name__ == "__main__":
    raise SystemExit(main())
