from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from firesight_vision.rosbag_depth_extract import (
    RosbagDepthExtractConfig,
    RosbagDepthExtractError,
    extract_rosbag_depth_pairs,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class RosbagDepthExtractNamespace(argparse.Namespace):
    bag_path: Path = Path()
    out_dir: Path = Path()
    max_pairs: int = 3
    start_seconds: float = 0.0
    interval_seconds: float = 10.0
    max_sync_delta_ms: float = 50.0
    preview_max_depth_m: float = 6.0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = extract_rosbag_depth_pairs(_parse_args(argv))
    except RosbagDepthExtractError as error:
        _ = sys.stderr.write(f"{error}\n")
        return 2
    _ = sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


def _parse_args(argv: Sequence[str] | None) -> RosbagDepthExtractConfig:
    namespace = RosbagDepthExtractNamespace()
    parser = argparse.ArgumentParser(
        description="Extract synchronized thermal, RGB, and aligned-depth PNGs.",
    )
    _ = parser.add_argument("--bag", dest="bag_path", required=True, type=Path)
    _ = parser.add_argument("--out", dest="out_dir", required=True, type=Path)
    _ = parser.add_argument("--max-pairs", default=3, type=int)
    _ = parser.add_argument("--start-seconds", default=0.0, type=float)
    _ = parser.add_argument("--interval-seconds", default=10.0, type=float)
    _ = parser.add_argument("--max-sync-delta-ms", default=50.0, type=float)
    _ = parser.add_argument("--preview-max-depth-m", default=6.0, type=float)
    _ = parser.parse_args(argv, namespace=namespace)
    return RosbagDepthExtractConfig(
        bag_path=namespace.bag_path,
        out_dir=namespace.out_dir,
        max_pairs=namespace.max_pairs,
        start_seconds=namespace.start_seconds,
        interval_seconds=namespace.interval_seconds,
        max_sync_delta_ms=namespace.max_sync_delta_ms,
        preview_max_depth_m=namespace.preview_max_depth_m,
    )


if __name__ == "__main__":
    raise SystemExit(main())
