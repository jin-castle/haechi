from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from firesight_vision.hud_edges import FIRE360_OSD_IGNORED_REGIONS
from firesight_vision.teed import TeedError
from firesight_vision.teed_replay import TeedReplayConfig, run_teed_replay

if TYPE_CHECKING:
    from collections.abc import Sequence


class TeedReplayArgNamespace(argparse.Namespace):
    input_path: Path = Path()
    out_dir: Path = Path()
    checkpoint_path: Path = Path()
    fps: float = 15.0
    max_frames: int | None = None
    threshold: float = 0.35
    background_scale: float = 0.24
    edge_width: int = 3
    device: str = "auto"
    realtime: bool = True
    ignore_fire360_osd: bool = False


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _parse_args(argv)
        payload = run_teed_replay(config)
    except TeedError as error:
        _ = sys.stderr.write(f"{error}\n")
        return 2

    _ = sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _parse_args(argv: Sequence[str] | None) -> TeedReplayConfig:
    namespace = TeedReplayArgNamespace()
    parser = argparse.ArgumentParser(
        description="Replay image frames through the pretrained TEED edge model.",
    )
    _ = parser.add_argument("--input", dest="input_path", required=True, type=Path)
    _ = parser.add_argument("--out", dest="out_dir", required=True, type=Path)
    _ = parser.add_argument(
        "--checkpoint",
        dest="checkpoint_path",
        required=True,
        type=Path,
    )
    _ = parser.add_argument("--fps", default=15.0, type=float)
    _ = parser.add_argument("--max-frames", default=None, type=int)
    _ = parser.add_argument(
        "--threshold",
        default=0.35,
        type=float,
        help="TEED probability cutoff in [0, 1]",
    )
    _ = parser.add_argument("--background-scale", default=0.24, type=float)
    _ = parser.add_argument("--edge-width", default=3, type=int)
    _ = parser.add_argument("--device", default="auto")
    _ = parser.add_argument(
        "--no-realtime",
        dest="realtime",
        action="store_false",
        help="disable target-FPS pacing and run as fast as possible",
    )
    _ = parser.add_argument(
        "--ignore-fire360-osd",
        action="store_true",
        help="ignore the fixed left badge and right colorbar regions",
    )
    _ = parser.parse_args(argv, namespace=namespace)
    return TeedReplayConfig(
        input_path=namespace.input_path,
        out_dir=namespace.out_dir,
        checkpoint_path=namespace.checkpoint_path,
        fps=namespace.fps,
        max_frames=namespace.max_frames,
        threshold=namespace.threshold,
        background_scale=namespace.background_scale,
        edge_width=namespace.edge_width,
        device=namespace.device,
        realtime=namespace.realtime,
        ignored_regions=(
            FIRE360_OSD_IGNORED_REGIONS if namespace.ignore_fire360_osd else ()
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
