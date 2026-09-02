from firesight_vision.bbox_build import SelectedImage, build_boxes


def test_video_obstacle_heuristic_stays_localized() -> None:
    selected = (
        SelectedImage(
            id=1,
            file_name="video__clip_02814_frame_000028.jpg",
            width=1280,
            height=720,
            source_type="video_frame",
        ),
        SelectedImage(
            id=2,
            file_name="video__ifsi_video_8_frame_000120.jpg",
            width=1280,
            height=720,
            source_type="video_frame",
        ),
    )

    boxes = [box for box in build_boxes(selected) if box.category == "obstacle"]

    assert len(boxes) == 2
    assert all(box.bbox[2] / 1280 <= 0.75 for box in boxes)
