from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from PIL import Image

import firesight_vision.teed_depth_replay as replay_module
from firesight_vision.depth_anything import DepthAnythingResult
from firesight_vision.teed_depth_replay import (
    TeedDepthReplayConfig,
    TeedDepthReplayError,
    run_teed_depth_replay,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _FakeTeedPredictor:
    def __init__(self, _checkpoint_path: Path, device: str = "auto") -> None:
        self.device: str = "cpu" if device == "auto" else device

    def process(
        self,
        image: Image.Image,
        *,
        threshold: float,
        background_scale: float,
        edge_width: int,
        ignored_regions: tuple[object, ...],
    ) -> SimpleNamespace:
        del threshold, background_scale, edge_width, ignored_regions
        mask = Image.new("L", image.size, 0)
        for y in range(image.height):
            mask.putpixel((image.width // 2, y), 255)
        return SimpleNamespace(
            overlay=image.copy(),
            probability=mask.copy(),
            mask=mask,
            edge_ratio=image.height / (image.width * image.height),
        )


class _FakeDepthPredictor:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        device: str,
        preview_max_depth_m: float,
    ) -> None:
        del preview_max_depth_m
        self.model_id: str = model_id
        self.revision: str = revision
        self.device: str = "cpu" if device == "auto" else device
        self.model_max_depth_m: float = 20.0

    def process(self, image: Image.Image) -> DepthAnythingResult:
        depth_m = np.full((image.height, image.width), 0.75, dtype=np.float32)
        valid_mask = Image.new("L", image.size, 255)
        return DepthAnythingResult(
            depth_m=depth_m,
            preview=Image.new("RGB", image.size, (10, 200, 100)),
            valid_mask=valid_mask,
            valid_pixels=image.width * image.height,
            total_pixels=image.width * image.height,
            valid_ratio=1.0,
            min_depth_m=0.75,
            median_depth_m=0.75,
            p95_depth_m=0.75,
            max_depth_m=0.75,
        )


def test_teed_depth_replay_writes_combined_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(replay_module, "TeedPredictor", _FakeTeedPredictor)
    monkeypatch.setattr(replay_module, "DepthAnythingPredictor", _FakeDepthPredictor)
    checkpoint = tmp_path / "teed.pth"
    _ = checkpoint.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    payload = run_teed_depth_replay(
        TeedDepthReplayConfig(
            input_path=Path("tests/fixtures/images/sample_room.jpg"),
            out_dir=out_dir,
            checkpoint_path=checkpoint,
        ),
    )

    assert payload["frames_processed"] == 1
    assert payload["depth_model_max_m"] == 20.0
    assert (out_dir / "replay_summary.json").exists()
    assert (out_dir / "replay_metrics.jsonl").exists()
    frames = cast("list[dict[str, object]]", payload["frame_outputs"])
    frame = frames[0]
    for key in (
        "teed_overlay_file",
        "depth_m_file",
        "depth_preview_file",
        "depth_overlay_file",
        "contact_sheet_file",
    ):
        file_name = cast("str", frame[key])
        assert (out_dir / file_name).exists()
    depth_file = cast("str", frame["depth_m_file"])
    depth_m = cast(
        "NDArray[np.float32]",
        np.load(out_dir / depth_file, allow_pickle=False),
    )
    assert depth_m.shape == (32, 48)
    assert frame["edge_depth_model_valid_ratio"] == 1.0
    assert frame["edge_depth_banded_ratio"] == 1.0
    assert frame["mid_edge_ratio"] == 1.0


def test_teed_depth_replay_rejects_non_positive_fps(tmp_path: Path) -> None:
    with pytest.raises(TeedDepthReplayError, match="fps"):
        _ = run_teed_depth_replay(
            TeedDepthReplayConfig(
                input_path=Path("tests/fixtures/images/sample_room.jpg"),
                out_dir=tmp_path / "out",
                checkpoint_path=tmp_path / "missing.pth",
                fps=0.0,
            ),
        )
