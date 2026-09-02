# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = []
# ///
# ----- How to run -----
# python scripts/validate_schema.py schemas/frame.schema.json docs/sensor_schema.md
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


JsonType = Literal["array", "boolean", "integer", "null", "number", "object", "string"]
JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class ValidationRequest:
    schema_path: Path
    doc_path: Path
    sample_path: Path | None
    evidence_path: Path | None


@dataclass(frozen=True, slots=True)
class CliError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


TOP_LEVEL_DOC_FIELDS: Final = (
    "timestamp",
    "frame_id",
    "camera",
    "thermal",
    "mmwave",
    "imu_ahrs",
    "calibration",
)
JSON_TYPE_BY_VALUE: Final[dict[str, JsonType]] = {
    "array": "array",
    "boolean": "boolean",
    "integer": "integer",
    "null": "null",
    "number": "number",
    "object": "object",
    "string": "string",
}


def main(argv: Sequence[str] | None = None) -> int:
    request = _parse_args(argv)
    problems = _validate_artifacts(request)
    sample_status = "not_provided"
    if request.sample_path is not None:
        sample_problems = _validate_sample(request.schema_path, request.sample_path)
        problems.extend(sample_problems)
        sample_status = "accepted" if len(sample_problems) == 0 else "rejected"

    status = "ok" if len(problems) == 0 else "error"
    sample_path = request.sample_path.as_posix() if request.sample_path else None
    payload: dict[str, str | list[str] | None] = {
        "status": status,
        "schema_path": request.schema_path.as_posix(),
        "doc_path": request.doc_path.as_posix(),
        "sample_path": sample_path,
        "sample_status": sample_status,
        "problems": problems,
    }
    if request.evidence_path is not None:
        request.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        _ = request.evidence_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(status)
    for problem in problems:
        print(problem)
    return 0 if status == "ok" else 1


def _parse_args(argv: Sequence[str] | None) -> ValidationRequest:
    namespace = SchemaArgNamespace()
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("schema_path", type=Path)
    _ = parser.add_argument("doc_path", type=Path)
    _ = parser.add_argument("--sample", type=Path)
    _ = parser.add_argument("--evidence", type=Path)
    _ = parser.parse_args(argv, namespace=namespace)
    return ValidationRequest(
        schema_path=namespace.schema_path,
        doc_path=namespace.doc_path,
        sample_path=namespace.sample,
        evidence_path=namespace.evidence,
    )


class SchemaArgNamespace(argparse.Namespace):
    schema_path: Path = Path()
    doc_path: Path = Path()
    sample: Path | None = None
    evidence: Path | None = None


def _validate_artifacts(request: ValidationRequest) -> list[str]:
    schema = _read_json_object(request.schema_path)
    problems: list[str] = []
    required = schema.get("required")
    if schema.get("type") != "object":
        problems.append("schema root type must be object")
    if not isinstance(required, list) or "timestamp" not in required:
        problems.append("schema must require timestamp")
    if not isinstance(schema.get("properties"), dict):
        problems.append("schema must define properties")
    doc_text = request.doc_path.read_text(encoding="utf-8")
    problems.extend(
        f"missing docs field {field}"
        for field in TOP_LEVEL_DOC_FIELDS
        if f"`{field}`" not in doc_text
    )
    return problems


def _validate_sample(schema_path: Path, sample_path: Path) -> list[str]:
    schema = _read_json_object(schema_path)
    sample = _read_json_object(sample_path)
    return _validate_value(schema, sample, "$")


def _read_json_object(path: Path) -> dict[str, JsonValue]:
    text = path.read_text(encoding="utf-8")
    parsed_value = _json_value(json.loads(text))
    if not isinstance(parsed_value, dict):
        raise CliError(message=f"{path.as_posix()} must contain a JSON object")
    return parsed_value


def _json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, float) and not math.isfinite(value):
        raise CliError(message="JSON numbers must be finite")
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return {key: _json_value(item) for key, item in value.items()}


def _validate_value(
    schema: Mapping[str, JsonValue],
    value: JsonValue,
    path: str,
) -> list[str]:
    problems: list[str] = []
    expected = schema.get("type")
    if isinstance(expected, str):
        json_type = _parse_json_type(expected)
        if not _matches_type(value, json_type):
            return [f"{path} expected {json_type}"]
        if json_type == "object" and isinstance(value, dict):
            problems.extend(_validate_object(schema, value, path))
        if json_type == "array" and isinstance(value, list):
            problems.extend(_validate_array(schema, value, path))
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        problems.append(f"{path} expected one of {enum_values}")
    minimum = schema.get("minimum")
    if (
        isinstance(minimum, int | float)
        and not isinstance(minimum, bool)
        and isinstance(value, int | float)
        and not isinstance(value, bool)
        and value < minimum
    ):
        problems.append(f"{path} expected minimum {minimum}")
    return problems


def _parse_json_type(value: str) -> JsonType:
    try:
        return JSON_TYPE_BY_VALUE[value]
    except KeyError as error:
        raise CliError(message=f"unsupported JSON schema type {value}") from error


def _matches_type(value: JsonValue, expected: JsonType) -> bool:
    return {
        "array": isinstance(value, list),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "null": value is None,
        "number": isinstance(value, int | float) and not isinstance(value, bool),
        "object": isinstance(value, dict),
        "string": isinstance(value, str),
    }[expected]


def _validate_object(
    schema: Mapping[str, JsonValue],
    value: Mapping[str, JsonValue],
    path: str,
) -> list[str]:
    problems: list[str] = []
    required = schema.get("required")
    if isinstance(required, list):
        problems.extend(
            f"required {field}"
            for field in required
            if isinstance(field, str) and field not in value
        )
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for field, subschema in properties.items():
            if field in value and isinstance(subschema, dict):
                problems.extend(
                    _validate_value(subschema, value[field], f"{path}.{field}"),
                )
    additional_properties = schema.get("additionalProperties")
    if additional_properties is False and isinstance(properties, dict):
        allowed = set(properties)
        problems.extend(
            f"{path}.{field} is not allowed" for field in value if field not in allowed
        )
    return problems


def _validate_array(
    schema: Mapping[str, JsonValue],
    value: Sequence[JsonValue],
    path: str,
) -> list[str]:
    items = schema.get("items")
    if not isinstance(items, dict):
        return []
    return [
        problem
        for index, item in enumerate(value)
        for problem in _validate_value(items, item, f"{path}[{index}]")
    ]


if __name__ == "__main__":
    raise SystemExit(main())
