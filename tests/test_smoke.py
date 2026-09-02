import subprocess
import sys
from pathlib import Path
from shutil import which

from PIL import Image

from firesight_vision.smoke import (
    SmokeSeverity,
    SmokeTransformRequest,
    apply_synthetic_smoke,
    smoke_parameters,
)


def test_smoke_transform_is_deterministic_for_same_seed() -> None:
    with Image.open("tests/fixtures/images/sample_room.jpg") as source_image:
        first = apply_synthetic_smoke(
            SmokeTransformRequest(
                image=source_image,
                severity=SmokeSeverity.MEDIUM,
                seed=42,
            ),
        )
        second = apply_synthetic_smoke(
            SmokeTransformRequest(
                image=source_image,
                severity=SmokeSeverity.MEDIUM,
                seed=42,
            ),
        )

    assert first.tobytes() == second.tobytes()
    with Image.open("tests/fixtures/images/sample_room.jpg") as clean_image:
        clean_bytes = clean_image.convert("RGB").tobytes()
    assert first.tobytes() != clean_bytes


def test_smoke_parameters_record_visibility_degradation() -> None:
    clean = smoke_parameters(SmokeSeverity.CLEAN)
    dense = smoke_parameters(SmokeSeverity.DENSE)

    assert clean.alpha == 0.0
    assert dense.alpha > clean.alpha
    assert dense.transmission < clean.transmission
    assert dense.contrast < clean.contrast


def test_generate_smoke_aug_writes_images_and_metadata(tmp_path: Path) -> None:
    out_dir = tmp_path / "run-a"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_smoke_aug.py",
            "--input",
            "tests/fixtures/images/sample_room.jpg",
            "--out",
            str(out_dir),
            "--severity",
            "clean",
            "mild",
            "medium",
            "dense",
            "--seed",
            "42",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    metadata = (out_dir / "metadata.json").read_text(encoding="utf-8")
    assert "Synthetic visibility degradation" in metadata
    for severity in ("clean", "mild", "medium", "dense"):
        assert f'"severity": "{severity}"' in metadata
    assert len(list(out_dir.glob("*.jpg"))) == 4


def test_generate_smoke_aug_entrypoint_matches_script(tmp_path: Path) -> None:
    script_out_dir = tmp_path / "script"
    entrypoint_out_dir = tmp_path / "entrypoint"
    uv_executable = which("uv")
    assert uv_executable is not None
    shared_args = [
        "--input",
        "tests/fixtures/images/sample_room.jpg",
        "--severity",
        "clean",
        "mild",
        "medium",
        "dense",
        "--seed",
        "42",
    ]

    script_result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_smoke_aug.py",
            "--out",
            str(script_out_dir),
            *shared_args,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    entrypoint_result = subprocess.run(
        [
            uv_executable,
            "run",
            "--frozen",
            "generate-smoke-aug",
            "--out",
            str(entrypoint_out_dir),
            *shared_args,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert script_result.returncode == 0
    assert entrypoint_result.returncode == 0
    assert script_result.stdout == entrypoint_result.stdout
    assert (script_out_dir / "metadata.json").read_text(encoding="utf-8") == (
        entrypoint_out_dir / "metadata.json"
    ).read_text(encoding="utf-8")


def test_check_smoke_determinism_reports_matching_runs(tmp_path: Path) -> None:
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    for out_dir in (run_a, run_b):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_smoke_aug.py",
                "--input",
                "tests/fixtures/images/sample_room.jpg",
                "--out",
                str(out_dir),
                "--severity",
                "clean",
                "mild",
                "medium",
                "dense",
                "--seed",
                "42",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_smoke_determinism.py",
            "--a",
            str(run_a),
            "--b",
            str(run_b),
            "--expect-same",
            "true",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert '"deterministic": true' in result.stdout


def test_check_smoke_determinism_rejects_metadata_file_mismatch(
    tmp_path: Path,
) -> None:
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    for out_dir in (run_a, run_b):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_smoke_aug.py",
                "--input",
                "tests/fixtures/images/sample_room.jpg",
                "--out",
                str(out_dir),
                "--severity",
                "clean",
                "mild",
                "--seed",
                "42",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    tampered_image = run_b / "sample_room_mild_seed42.jpg"
    _ = tampered_image.write_bytes(tampered_image.read_bytes() + b"tamper")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_smoke_determinism.py",
            "--a",
            str(run_a),
            "--b",
            str(run_b),
            "--expect-same",
            "true",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "metadata/file checksum mismatch" in result.stderr


def test_generate_smoke_aug_rejects_unknown_severity(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_smoke_aug.py",
            "--input",
            "tests/fixtures/images/sample_room.jpg",
            "--out",
            str(tmp_path / "bad"),
            "--severity",
            "extreme",
            "--seed",
            "42",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "allowed severities" in result.stderr
