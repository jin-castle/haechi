import subprocess
import sys
from pathlib import Path


def test_validate_registry_writes_ok_evidence_for_candidate_ledger(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "registry-ok.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_registry.py",
            "data/registry/datasets.yaml",
            "--evidence",
            str(evidence_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    evidence = evidence_path.read_text(encoding="utf-8")
    expected_active_count = _count_active_datasets(Path("data/registry/datasets.yaml"))
    assert '"status": "ok"' in evidence
    assert f'"active_count": {expected_active_count}' in evidence


def test_validate_registry_reports_missing_license_from_bad_fixture(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "registry-bad.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_registry.py",
            "tests/fixtures/bad/datasets_missing_license.yaml",
            "--evidence",
            str(evidence_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "missing license_or_access_status" in result.stdout
    evidence = evidence_path.read_text(encoding="utf-8")
    assert '"status": "error"' in evidence
    assert "missing license_or_access_status" in evidence


def test_validate_registry_rejects_yaml_outside_documented_subset(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "nested.yaml"
    _ = registry_path.write_text(
        """datasets:
  - id: nested
    name: Nested YAML
    url: https://example.invalid/nested
    modality:
      - RGB image
    label_task: fixture
    access_method: fixture
    license_or_access_status: fixture
    access_notes: fixture
    planned_use: fixture
    active: true
""",
        encoding="utf-8",
    )
    evidence_path = tmp_path / "nested-evidence.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_registry.py",
            str(registry_path),
            "--evidence",
            str(evidence_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unsupported YAML subset" in result.stdout


def _count_active_datasets(registry_path: Path) -> int:
    return sum(
        1
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip() == "active: true"
    )
