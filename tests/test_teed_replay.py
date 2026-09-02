from __future__ import annotations

from pathlib import Path

import pytest

from firesight_vision.teed import TeedError
from firesight_vision.teed_replay import TeedReplayConfig, run_teed_replay


def test_teed_replay_requires_existing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(TeedError, match="missing TEED checkpoint"):
        _ = run_teed_replay(
            TeedReplayConfig(
                input_path=Path("tests/fixtures/images/sample_room.jpg"),
                out_dir=tmp_path / "teed",
                checkpoint_path=tmp_path / "missing.pth",
                realtime=False,
            ),
        )


def test_teed_replay_rejects_non_positive_fps(tmp_path: Path) -> None:
    with pytest.raises(TeedError, match="fps"):
        _ = run_teed_replay(
            TeedReplayConfig(
                input_path=Path("tests/fixtures/images/sample_room.jpg"),
                out_dir=tmp_path / "teed",
                checkpoint_path=tmp_path / "missing.pth",
                fps=0.0,
                realtime=False,
            ),
        )
