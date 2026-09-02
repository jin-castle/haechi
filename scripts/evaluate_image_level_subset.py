# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = []
# ///
# ----- How to run -----
# python scripts/evaluate_image_level_subset.py
#   --manifest data/eval/fire360_indoor_image_level_v0.json
#   --predictions .omo/evidence/.../yolo11n_image_level_predictions.json
#   .omo/evidence/.../owlvit_image_level_predictions.json
#   --out .omo/evidence/.../image_level_metrics.json

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from firesight_vision.image_level_eval import EvalError, write_evaluation

if TYPE_CHECKING:
    from collections.abc import Sequence


class EvalArgNamespace(argparse.Namespace):
    manifest: Path = Path()
    predictions: list[Path] | None = None
    out: Path = Path()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        namespace = _parse_args(argv)
        payload = write_evaluation(
            namespace.manifest,
            tuple(namespace.predictions or ()),
            namespace.out,
        )
    except EvalError as error:
        print(error, file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parse_args(argv: Sequence[str] | None) -> EvalArgNamespace:
    namespace = EvalArgNamespace()
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--manifest", required=True, type=Path)
    _ = parser.add_argument("--predictions", nargs="+", required=True, type=Path)
    _ = parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv, namespace=namespace)


if __name__ == "__main__":
    raise SystemExit(main())
