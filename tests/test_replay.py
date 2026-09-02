from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from PIL import Image

from firesight_vision.hud_edges import HudEdgeProfile
from firesight_vision.replay import (
    ReplayConfig,
    ReplayError,
    collect_replay_frames,
    run_edge_replay,
)


def test_collect_replay_frames_sorts_images_and_ignores_other_files(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "frames"
    input_dir.mkdir()
    fixture = Path("tests/fixtures/images/sample_room.jpg")
    _ = shutil.copyfile(fixture, input_dir / "frame_002.jpg")
    _ = shutil.copyfile(fixture, input_dir / "frame_001.jpg")
    _ = (input_dir / "README.txt").write_text("not a frame", encoding="utf-8")

    frames = collect_replay_frames(input_dir)

    assert [path.name for path in frames] == ["frame_001.jpg", "frame_002.jpg"]


def test_collect_replay_frames_rejects_empty_directory(tmp_path: Path) -> None:
    input_dir = tmp_path / "empty"
    input_dir.mkdir()

    with pytest.raises(ReplayError, match="no supported replay images"):
        _ = collect_replay_frames(input_dir)


def test_run_edge_replay_writes_metrics_and_frame_artifacts(tmp_path: Path) -> None:
    input_dir = tmp_path / "frames"
    input_dir.mkdir()
    fixture = Path("tests/fixtures/images/sample_room.jpg")
    for index in range(3):
        _ = shutil.copyfile(fixture, input_dir / f"frame_{index:03d}.jpg")
    out_dir = tmp_path / "replay"

    summary = run_edge_replay(
        ReplayConfig(
            input_path=input_dir,
            out_dir=out_dir,
            max_frames=2,
            profile=HudEdgeProfile.STANDARD,
            realtime=False,
        ),
    )

    assert summary["protocol"] == "pillow_hud_edge_replay_v1"
    assert summary["frames_available"] == 3
    assert summary["frames_requested"] == 2
    assert summary["frames_processed"] == 2
    assert summary["realtime_requested"] is False
    assert summary["pacing_enabled"] is False
    assert summary["effective_fps"] > 0.0
    assert summary["latency_ms"]["p95"] > 0.0
    assert summary["edge_ratio"]["mean"] > 0.0
    assert (out_dir / "replay_summary.json").exists()
    assert len((out_dir / "replay_metrics.jsonl").read_text("utf-8").splitlines()) == 2
    for frame in summary["frame_outputs"]:
        overlay_path = out_dir / frame["overlay_file"]
        mask_path = out_dir / frame["mask_file"]
        assert overlay_path.exists()
        assert mask_path.exists()
        with Image.open(overlay_path) as overlay, Image.open(mask_path) as mask:
            assert overlay.mode == "RGB"
            assert mask.mode == "L"

    persisted = cast(
        "dict[str, object]",
        json.loads((out_dir / "replay_summary.json").read_text("utf-8")),
    )
    assert persisted["frames_processed"] == 2


def test_run_edge_replay_rejects_non_positive_fps(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="fps"):
        _ = run_edge_replay(
            ReplayConfig(
                input_path=Path("tests/fixtures/images/sample_room.jpg"),
                out_dir=tmp_path / "replay",
                fps=0.0,
                realtime=False,
            ),
        )


def test_run_edge_replay_cli_outputs_summary(tmp_path: Path) -> None:
    out_dir = tmp_path / "replay"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_edge_replay.py",
            "--input",
            "tests/fixtures/images/sample_room.jpg",
            "--out",
            str(out_dir),
            "--profile",
            "standard",
            "--max-frames",
            "1",
            "--no-realtime",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    payload = cast("dict[str, object]", json.loads(result.stdout))
    assert payload["frames_processed"] == 1
    assert (out_dir / "replay_summary.json").exists()
