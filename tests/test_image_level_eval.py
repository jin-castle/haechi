import json
import subprocess
import sys
from pathlib import Path

from firesight_vision.image_level_eval import JsonValue, evaluate_manifest


def test_evaluate_manifest_reports_precision_recall(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    predictions = tmp_path / "predictions.json"

    manifest_payload: dict[str, JsonValue] = {
        "classes": ["fire", "smoke", "door", "person", "exit", "obstacle"],
        "items": [
            {"image": "a.jpg", "labels": ["fire", "smoke"]},
            {"image": "b.jpg", "labels": ["person", "obstacle"]},
            {"image": "c.jpg", "labels": ["door", "exit"]},
        ],
    }
    prediction_payload: dict[str, JsonValue] = {
        "model": "toy",
        "items": [
            {"image": "a.jpg", "labels": ["fire"]},
            {"image": "b.jpg", "labels": ["person", "smoke"]},
            {"image": "c.jpg", "labels": ["door", "exit"]},
        ],
    }
    _ = manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    _ = predictions.write_text(json.dumps(prediction_payload), encoding="utf-8")

    result = evaluate_manifest(manifest, (predictions,))

    toy = result["models"]["toy"]["per_class"]
    assert toy["fire"]["precision"] == 1.0
    assert toy["fire"]["recall"] == 1.0
    assert toy["smoke"]["tp"] == 0
    assert toy["smoke"]["fp"] == 1
    assert toy["smoke"]["fn"] == 1
    assert toy["exit"]["f1"] == 1.0
    assert result["models"]["toy"]["macro_precision"] == 0.6666666666666666
    assert result["models"]["toy"]["macro_recall"] == 0.6666666666666666
    assert result["models"]["toy"]["macro_f1"] == 0.6666666666666666


def test_evaluate_image_level_subset_cli_rejects_unknown_prediction_label(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    predictions = tmp_path / "predictions.json"
    out = tmp_path / "metrics.json"

    manifest_payload: dict[str, JsonValue] = {
        "classes": ["fire"],
        "items": [{"image": "a.jpg", "labels": []}],
    }
    _ = manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    bad_prediction: dict[str, JsonValue] = {
        "model": "bad",
        "items": [{"image": "a.jpg", "labels": ["smoke"]}],
    }
    _ = predictions.write_text(json.dumps(bad_prediction), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_image_level_subset.py",
            "--manifest",
            str(manifest),
            "--predictions",
            str(predictions),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unknown prediction labels" in result.stderr
