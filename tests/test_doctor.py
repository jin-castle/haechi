import subprocess
import sys
from pathlib import Path


def test_doctor_writes_ok_evidence_when_scaffold_exists(tmp_path: Path) -> None:
    evidence_path = tmp_path / "doctor.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/doctor.py",
            "--check",
            "scaffold",
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


def test_doctor_reports_missing_registry_from_bad_fixture(tmp_path: Path) -> None:
    evidence_path = tmp_path / "doctor-bad.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/doctor.py",
            "--check",
            "scaffold",
            "--fixture",
            "tests/fixtures/bad/scaffold_missing_registry.yaml",
            "--evidence",
            str(evidence_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "missing:data/registry" in result.stdout
    evidence = evidence_path.read_text(encoding="utf-8")
    assert "missing:data/registry" in evidence
