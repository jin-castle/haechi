from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from firesight_vision.depth_anything import (
    DEFAULT_DEPTH_MODEL_ID,
    DEFAULT_DEPTH_MODEL_REVISION,
    DepthAnythingError,
)
from firesight_vision.depth_edge_fusion import DepthEdgeFusionError
from firesight_vision.hud_edges import FIRE360_OSD_IGNORED_REGIONS
from firesight_vision.teed import TeedError
from firesight_vision.teed_depth_replay import (
    TeedDepthReplayConfig,
    TeedDepthReplayError,
    run_teed_depth_replay,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class TeedDepthReplayArgNamespace(argparse.Namespace):
    input_path: Path = Path()
    out_dir: Path = Path()
    checkpoint_path: Path = Path()
    depth_model_id: str = DEFAULT_DEPTH_MODEL_ID
    depth_model_revision: str = DEFAULT_DEPTH_MODEL_REVISION
    fps: float = 5.0
    max_frames: int | None = None
    threshold: float = 0.75
    background_scale: float = 0.24
    edge_width: int = 1
    device: str = "auto"
    realtime: bool = False
    preview_max_depth_m: float = 6.0
    ignore_fire360_osd: bool = False


def main(argv: Sequence[str] | None = None) -> int:
    try:
        payload = run_teed_depth_replay(_parse_args(argv))
    except (
        DepthAnythingError,
        DepthEdgeFusionError,
        TeedDepthReplayError,
        TeedError,
    ) as error:
        _ = sys.stderr.write(f"{error}\n")
        return 2
    _ = sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _parse_args(argv: Sequence[str] | None) -> TeedDepthReplayConfig:
    namespace = TeedDepthReplayArgNamespace()
    parser = argparse.ArgumentParser(
        description="Build an offline TEED plus metric-depth HUD proof of concept.",
    )
    _ = parser.add_argument("--input", dest="input_path", required=True, type=Path)
    _ = parser.add_argument("--out", dest="out_dir", required=True, type=Path)
    _ = parser.add_argument(
        "--checkpoint",
        dest="checkpoint_path",
        required=True,
        type=Path,
    )
    _ = parser.add_argument(
        "--depth-model",
        dest="depth_model_id",
        default=DEFAULT_DEPTH_MODEL_ID,
    )
    _ = parser.add_argument(
        "--depth-revision",
        dest="depth_model_revision",
        default=DEFAULT_DEPTH_MODEL_REVISION,
    )
    _ = parser.add_argument("--fps", default=5.0, type=float)
    _ = parser.add_argument("--max-frames", default=None, type=int)
    _ = parser.add_argument("--threshold", default=0.75, type=float)
    _ = parser.add_argument("--background-scale", default=0.24, type=float)
    _ = parser.add_argument("--edge-width", default=1, type=int)
    _ = parser.add_argument("--device", default="auto")
    _ = parser.add_argument("--preview-max-depth-m", default=6.0, type=float)
    _ = parser.add_argument(
        "--realtime",
        action="store_true",
        help="pace frames at --fps; offline mode is the default",
    )
    _ = parser.add_argument(
        "--ignore-fire360-osd",
        action="store_true",
        help="exclude the fixed Fire360 badge and colorbar from TEED edges",
    )
    _ = parser.parse_args(argv, namespace=namespace)
    return TeedDepthReplayConfig(
        input_path=namespace.input_path,
        out_dir=namespace.out_dir,
        checkpoint_path=namespace.checkpoint_path,
        depth_model_id=namespace.depth_model_id,
        depth_model_revision=namespace.depth_model_revision,
        fps=namespace.fps,
        max_frames=namespace.max_frames,
        threshold=namespace.threshold,
        background_scale=namespace.background_scale,
        edge_width=namespace.edge_width,
        device=namespace.device,
        realtime=namespace.realtime,
        preview_max_depth_m=namespace.preview_max_depth_m,
        ignored_regions=(
            FIRE360_OSD_IGNORED_REGIONS if namespace.ignore_fire360_osd else ()
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
