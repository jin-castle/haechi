# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = []
# ///
# ----- How to run -----
# python scripts/validate_registry.py data/registry/datasets.yaml --evidence <path>
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence


ScalarValue = str | bool
RawDataset: TypeAlias = dict[str, ScalarValue]
Status = Literal["ok", "error"]

REQUIRED_FIELDS: Final = (
    "id",
    "name",
    "url",
    "modality",
    "label_task",
    "access_method",
    "license_or_access_status",
    "access_notes",
    "planned_use",
    "active",
)
STRING_FIELDS: Final = (
    "id",
    "name",
    "url",
    "modality",
    "label_task",
    "access_method",
    "license_or_access_status",
    "access_notes",
    "planned_use",
)
PARSER_DESCRIPTION: Final = (
    "YAML subset: top-level datasets list of flat mappings; values are plain "
    "strings or true/false booleans; nested lists/maps and quoted scalars are "
    "not supported."
)
BOOLEAN_SCALARS: Final[dict[str, bool]] = {"true": True, "false": False}


class DatasetSummary(TypedDict):
    id: str
    url: str
    modality: str
    label_task: str
    license_or_access_status: str
    active: bool


class EvidencePayload(TypedDict):
    status: Status
    registry: str
    parser: str
    dataset_count: int
    active_count: int
    problems: list[str]
    datasets: list[DatasetSummary]


class ParsedLine(TypedDict):
    key: str
    value: ScalarValue


@dataclass(frozen=True, slots=True)
class CliRequest:
    registry: Path
    evidence: Path | None


@dataclass(frozen=True, slots=True)
class RegistryError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


class ParserState:
    datasets: list[RawDataset]
    current: RawDataset | None
    saw_header: bool

    def __init__(self) -> None:
        """Initialize mutable parser state for the line-by-line YAML subset scan."""
        self.datasets = []
        self.current = None
        self.saw_header = False


@dataclass(frozen=True, slots=True)
class RegistryLine:
    path: Path
    number: int
    text: str


def main(argv: Sequence[str] | None = None) -> int:
    request = _parse_args(argv)
    try:
        datasets = _parse_dataset_yaml(request.registry)
        problems = _validate_datasets(datasets)
    except RegistryError as error:
        datasets = []
        problems = [str(error)]
    summaries = _summaries(datasets)
    status: Status = "ok" if len(problems) == 0 else "error"
    payload = EvidencePayload(
        status=status,
        registry=request.registry.as_posix(),
        parser=PARSER_DESCRIPTION,
        dataset_count=len(datasets),
        active_count=sum(1 for summary in summaries if summary["active"]),
        problems=problems,
        datasets=summaries,
    )

    if request.evidence is not None:
        request.evidence.parent.mkdir(parents=True, exist_ok=True)
        _ = request.evidence.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(status)
    for problem in problems:
        print(problem)
    return 0 if status == "ok" else 1


def _parse_args(argv: Sequence[str] | None) -> CliRequest:
    namespace = RegistryArgNamespace()
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("registry", type=Path)
    _ = parser.add_argument("--evidence", type=Path)
    _ = parser.parse_args(argv, namespace=namespace)
    return CliRequest(registry=namespace.registry, evidence=namespace.evidence)


class RegistryArgNamespace(argparse.Namespace):
    registry: Path = Path()
    evidence: Path | None = None


def _parse_dataset_yaml(path: Path) -> list[RawDataset]:
    lines = path.read_text(encoding="utf-8").splitlines()
    state = ParserState()

    for line_number, raw_line in enumerate(lines, start=1):
        _parse_dataset_line(
            RegistryLine(path=path, number=line_number, text=raw_line),
            state,
        )

    if state.current is not None:
        state.datasets.append(state.current)
    if len(state.datasets) == 0:
        raise RegistryError(message=f"{path}: missing datasets")
    return state.datasets


def _parse_dataset_line(registry_line: RegistryLine, state: ParserState) -> None:
    line = registry_line.text.rstrip()
    if len(line.strip()) == 0 or line.lstrip().startswith("#"):
        return
    if line == "datasets:":
        state.saw_header = True
        return
    if not state.saw_header:
        raise RegistryError(
            message=f"{registry_line.path}:{registry_line.number}: expected datasets:",
        )
    if line.startswith("  - "):
        _start_dataset(_registry_line_with_text(registry_line, line), state)
        return
    if line.startswith("    "):
        _continue_dataset(_registry_line_with_text(registry_line, line), state)
        return
    raise _unsupported_subset(registry_line)


def _registry_line_with_text(registry_line: RegistryLine, text: str) -> RegistryLine:
    return RegistryLine(path=registry_line.path, number=registry_line.number, text=text)


def _start_dataset(registry_line: RegistryLine, state: ParserState) -> None:
    if state.current is not None:
        state.datasets.append(state.current)
    state.current = {}
    _set_parsed_line(state.current, _parse_key_value(registry_line, 4))


def _continue_dataset(registry_line: RegistryLine, state: ParserState) -> None:
    if state.current is None:
        raise _unsupported_subset(registry_line)
    _set_parsed_line(state.current, _parse_key_value(registry_line, 4))


def _parse_key_value(registry_line: RegistryLine, prefix_size: int) -> ParsedLine:
    text = registry_line.text[prefix_size:]
    if ":" not in text:
        raise _unsupported_subset(registry_line)
    key, raw_value = text.split(":", maxsplit=1)
    value = raw_value.strip()
    if len(key) == 0 or len(value) == 0:
        raise _unsupported_subset(registry_line)
    if value.startswith(("-", "[", "{", "'", '"')):
        raise _unsupported_subset(registry_line)
    return ParsedLine(key=key, value=BOOLEAN_SCALARS.get(value, value))


def _set_parsed_line(dataset: RawDataset, parsed_line: ParsedLine) -> None:
    key = parsed_line["key"]
    if key not in REQUIRED_FIELDS:
        return
    dataset[key] = parsed_line["value"]


def _unsupported_subset(registry_line: RegistryLine) -> RegistryError:
    return RegistryError(
        message=(
            f"{registry_line.path}:{registry_line.number}: unsupported YAML subset"
        ),
    )


def _validate_datasets(datasets: list[RawDataset]) -> list[str]:
    problems: list[str] = []
    for index, dataset in enumerate(datasets, start=1):
        dataset_name = _dataset_name(dataset, index)
        for field in REQUIRED_FIELDS:
            if field not in dataset:
                _append_problem(problems, f"{dataset_name}: missing {field}")
        for field in STRING_FIELDS:
            if field in dataset and not isinstance(dataset[field], str):
                _append_problem(problems, f"{dataset_name}: {field} must be a string")
        if "active" in dataset and not isinstance(dataset["active"], bool):
            _append_problem(problems, f"{dataset_name}: active must be true or false")
        if _active(dataset):
            for problem in _validate_active_dataset(dataset_name, dataset):
                _append_problem(problems, problem)
    return problems


def _dataset_name(dataset: RawDataset, index: int) -> str:
    value = dataset.get("id")
    if isinstance(value, str) and len(value) > 0:
        return value
    return f"dataset[{index}]"


def _active(dataset: RawDataset) -> bool:
    value = dataset.get("active")
    return value is True


def _validate_active_dataset(dataset_name: str, dataset: RawDataset) -> list[str]:
    problems: list[str] = []
    for field in ("url", "modality", "label_task", "license_or_access_status"):
        value = dataset.get(field)
        if not isinstance(value, str) or len(value.strip()) == 0:
            _append_problem(problems, f"{dataset_name}: missing {field}")
    return problems


def _append_problem(problems: list[str], problem: str) -> None:
    if problem not in problems:
        problems.append(problem)


def _summaries(datasets: list[RawDataset]) -> list[DatasetSummary]:
    return [
        DatasetSummary(
            id=_string_or_empty(dataset, "id"),
            url=_string_or_empty(dataset, "url"),
            modality=_string_or_empty(dataset, "modality"),
            label_task=_string_or_empty(dataset, "label_task"),
            license_or_access_status=_string_or_empty(
                dataset,
                "license_or_access_status",
            ),
            active=_active(dataset),
        )
        for dataset in datasets
    ]


def _string_or_empty(dataset: RawDataset, field: str) -> str:
    value = dataset.get(field)
    if isinstance(value, str):
        return value
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
