# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["Pillow>=10.4,<11"]
# ///
# ----- How to run -----
# python scripts/run_hud_edge_baseline.py
#   --input data/input/baseline/c-thru-reference.png
#   tests/fixtures/images/sample_room.jpg
#   --out .omo/evidence/firesight-smoke-vision/baseline/hud-edge

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from firesight_vision.hud_edges import (
    FIRE360_OSD_IGNORED_REGIONS,
    HudEdgeError,
    HudEdgeProfile,
    HudEdgeRequest,
    run_hud_edge_baseline,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class HudEdgeArgNamespace(argparse.Namespace):
    input_paths: list[Path] | None = None
    out_dir: Path = Path()
    threshold: int | None = None
    background_scale: float = 0.24
    edge_width: int | None = None
    profile: str = HudEdgeProfile.STANDARD.value
    ignore_fire360_osd: bool = False


def main(argv: Sequence[str] | None = None) -> int:
    try:
        requests = _parse_args(argv)
        payload = run_hud_edge_baseline(requests)
    except HudEdgeError as error:
        _ = sys.stderr.write(f"{error}\n")
        return 2

    _ = sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _parse_args(argv: Sequence[str] | None) -> tuple[HudEdgeRequest, ...]:
    namespace = HudEdgeArgNamespace()
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--input", dest="input_paths", nargs="+", type=Path)
    _ = parser.add_argument("--out", dest="out_dir", required=True, type=Path)
    _ = parser.add_argument("--threshold", default=None, type=int)
    _ = parser.add_argument("--background-scale", default=0.24, type=float)
    _ = parser.add_argument("--edge-width", default=None, type=int)
    _ = parser.add_argument(
        "--profile",
        choices=[profile.value for profile in HudEdgeProfile],
        default=HudEdgeProfile.STANDARD.value,
    )
    _ = parser.add_argument(
        "--ignore-fire360-osd",
        action="store_true",
        help="ignore the fixed left badge and right colorbar regions",
    )
    _ = parser.parse_args(argv, namespace=namespace)
    if namespace.input_paths is None or len(namespace.input_paths) == 0:
        raise HudEdgeError(message="at least one --input image is required")
    profile = HudEdgeProfile(namespace.profile)
    threshold = namespace.threshold
    if threshold is None:
        threshold = _default_threshold(profile)
    edge_width = namespace.edge_width
    if edge_width is None:
        edge_width = _default_edge_width(profile)
    return tuple(
        HudEdgeRequest(
            input_path=input_path,
            out_dir=namespace.out_dir,
            threshold=threshold,
            background_scale=namespace.background_scale,
            edge_width=edge_width,
            profile=profile,
            ignored_regions=(
                FIRE360_OSD_IGNORED_REGIONS
                if namespace.ignore_fire360_osd
                else ()
            ),
        )
        for input_path in namespace.input_paths
    )


def _default_threshold(profile: HudEdgeProfile) -> int:
    match profile:
        case HudEdgeProfile.STANDARD:
            return 36
        case HudEdgeProfile.DENSE_SMOKE:
            return 144
        case HudEdgeProfile.FIRE_LINE:
            return 192


def _default_edge_width(profile: HudEdgeProfile) -> int:
    match profile:
        case HudEdgeProfile.STANDARD:
            return 1
        case HudEdgeProfile.DENSE_SMOKE:
            return 3
        case HudEdgeProfile.FIRE_LINE:
            return 5


if __name__ == "__main__":
    raise SystemExit(main())
