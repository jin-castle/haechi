from pathlib import Path

import pytest
from scripts.build_fire360_model_evaluation import (
    CandidateFrame,
    ModelEvaluationError,
    ProfilePayload,
    evenly_spaced_indices,
    select_frames,
    video_key,
)


def test_evenly_spaced_indices_include_candidate_endpoints() -> None:
    assert evenly_spaced_indices(10, 4) == (0, 3, 6, 9)
    assert evenly_spaced_indices(10, 5) == (0, 2, 5, 7, 9)


def test_evenly_spaced_indices_reject_invalid_request() -> None:
    with pytest.raises(ModelEvaluationError, match="invalid frame selection"):
        _ = evenly_spaced_indices(3, 4)


def test_select_frames_produces_fixed_30_frame_distribution() -> None:
    input_files = (
        "02814 (2).MTS",
        "02815 (2).MTS",
        "02824 (2).MTS",
        "GOPR8356 (2).MP4",
        "IFSI Video 4 (2).mp4",
        "IFSI Video 8 (2).mp4",
        "sample_3.MP4",
    )
    profiles = [
        ProfilePayload(
            input_file=input_file,
            osd_profile="none",
            ignore_fire360_osd=False,
            reason="test",
        )
        for input_file in input_files
    ]
    candidates = {
        video_key(input_file): tuple(
            CandidateFrame(
                path=Path(f"{video_key(input_file)}_frame_{index:06d}.jpg"),
                source_frame=index,
            )
            for index in range(10)
        )
        for input_file in input_files
    }

    selected = select_frames(profiles, candidates)

    assert len(selected) == 30
    assert [frame.evaluation_id for frame in selected] == [
        f"F{index:03d}" for index in range(1, 31)
    ]
    assert {
        key: sum(frame.video_key == key for frame in selected) for key in candidates
    } == {
        "clip_02814": 4,
        "clip_02815": 4,
        "clip_02824": 4,
        "gopr8356": 4,
        "ifsi_video_4": 5,
        "ifsi_video_8": 5,
        "sample_3": 4,
    }
