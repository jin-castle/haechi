import subprocess
import sys
from pathlib import Path


def test_validate_schema_accepts_valid_multimodal_frame(tmp_path: Path) -> None:
    evidence_path = tmp_path / "schema-ok.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_schema.py",
            "schemas/frame.schema.json",
            "docs/sensor_schema.md",
            "--sample",
            "tests/fixtures/schema/valid_frame.json",
            "--evidence",
            str(evidence_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    evidence = evidence_path.read_text(encoding="utf-8")
    assert '"status": "ok"' in evidence
    assert '"sample_status": "accepted"' in evidence


def test_validate_schema_reports_required_timestamp_for_bad_frame(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "schema-bad.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_schema.py",
            "schemas/frame.schema.json",
            "docs/sensor_schema.md",
            "--sample",
            "tests/fixtures/bad/frame_missing_timestamp.json",
            "--evidence",
            str(evidence_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "required timestamp" in result.stdout
    evidence = evidence_path.read_text(encoding="utf-8")
    assert '"status": "error"' in evidence
    assert "required timestamp" in evidence
