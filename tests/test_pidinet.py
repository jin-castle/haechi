from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from firesight_vision.pidinet import PidinetPredictor

pytest.importorskip("numpy")
pytest.importorskip("torch")


def test_pidinet_official_checkpoint_preserves_frame_shape() -> None:
    checkpoint = Path("models/pidinet/table5_pidinet.pth")
    if not checkpoint.is_file():
        pytest.skip("PiDiNet checkpoint is not present")
    image = Image.new("RGB", (64, 48), (40, 40, 40))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 55, 39), outline=(230, 230, 230), width=2)

    predictor = PidinetPredictor(checkpoint, device="cpu")
    result = predictor.process(image, threshold=0.30, edge_width=1)

    assert predictor.device == "cpu"
    assert result.probability.size == image.size
    assert result.mask.size == image.size
    assert result.overlay.size == image.size
    assert result.total_pixels == 64 * 48
    assert 0.0 <= result.edge_ratio <= 1.0
