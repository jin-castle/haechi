# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = []
# ///
# ----- How to run -----
# python scripts/doctor.py --check scaffold --evidence <path>
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence


CheckName = Literal["scaffold"]


class EvidencePayload(TypedDict):
    check: str
    status: str
    problems: list[str]


@dataclass(frozen=True, slots=True)
class DoctorRequest:
    check: CheckName
    evidence: Path
    fixture: Path | None


@dataclass(frozen=True, slots=True)
class CliError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


REQUIRED_PATHS = (
    Path("README.md"),
    Path("pyproject.toml"),
    Path("src/firesight_vision"),
    Path("configs"),
    Path("scripts"),
    Path("data/registry"),
    Path("reports"),
    Path(".omo/evidence/firesight-smoke-vision"),
)


def main(argv: Sequence[str] | None = None) -> int:
    request = _parse_args(argv)

    problems = _check_scaffold(request.fixture)
    status = "ok" if len(problems) == 0 else "error"
    payload = EvidencePayload(check=request.check, status=status, problems=problems)

    request.evidence.parent.mkdir(parents=True, exist_ok=True)
    _ = request.evidence.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(status)
    for problem in problems:
        print(problem)

    return 0 if status == "ok" else 1


def _parse_args(argv: Sequence[str] | None) -> DoctorRequest:
    namespace = DoctorArgNamespace()
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--check", required=True, choices=("scaffold",))
    _ = parser.add_argument("--evidence", required=True, type=Path)
    _ = parser.add_argument("--fixture", type=Path)
    _ = parser.parse_args(argv, namespace=namespace)

    return DoctorRequest(
        check=_parse_check(namespace.check),
        evidence=namespace.evidence,
        fixture=namespace.fixture,
    )


class DoctorArgNamespace(argparse.Namespace):
    check: str = ""
    evidence: Path = Path()
    fixture: Path | None = None


def _parse_check(value: str) -> CheckName:
    if value == "scaffold":
        return "scaffold"
    raise CliError(message=f"unsupported check {value}")


def _check_scaffold(fixture: Path | None) -> list[str]:
    missing = [path for path in REQUIRED_PATHS if not path.exists()]
    if fixture is not None:
        missing.extend(_read_fixture_missing_paths(fixture))
    return [f"missing:{path.as_posix()}" for path in missing]


def _read_fixture_missing_paths(path: Path) -> list[Path]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [
        Path(line.removeprefix("  - ").strip())
        for line in lines
        if line.startswith("  - ")
    ]


if __name__ == "__main__":
    raise SystemExit(main())
