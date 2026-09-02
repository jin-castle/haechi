# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = []
# ///
# ----- How to run -----
# python scripts/check_smoke_determinism.py
#   --a .omo/evidence/firesight-smoke-vision/task-3/run-a
#   --b .omo/evidence/firesight-smoke-vision/task-3/run-b
#   --expect-same true
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence


Expectation = Literal[True, False]
EXPECTATION_VALUES: dict[str, Expectation] = {"true": True, "false": False}
JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class VariantChecksum(TypedDict):
    severity: str
    output_file: str
    sha256: str


class DeterminismPayload(TypedDict):
    deterministic: bool
    expected_same: bool
    a: str
    b: str
    compared: list[VariantChecksum]


@dataclass(frozen=True, slots=True)
class DeterminismRequest:
    a_dir: Path
    b_dir: Path
    expect_same: Expectation


@dataclass(frozen=True, slots=True)
class CliError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class MetadataVariant:
    severity: str
    output_file: str
    recorded_sha256: str


class DeterminismArgNamespace(argparse.Namespace):
    a_dir: Path = Path()
    b_dir: Path = Path()
    expect_same: str = ""


def main(argv: Sequence[str] | None = None) -> int:
    try:
        request = _parse_args(argv)
        payload = _compare(request)
    except CliError as error:
        print(error, file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["deterministic"] is request.expect_same else 1


def _parse_args(argv: Sequence[str] | None) -> DeterminismRequest:
    namespace = DeterminismArgNamespace()
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--a", dest="a_dir", required=True, type=Path)
    _ = parser.add_argument("--b", dest="b_dir", required=True, type=Path)
    _ = parser.add_argument("--expect-same", required=True, choices=("true", "false"))
    _ = parser.parse_args(argv, namespace=namespace)
    return DeterminismRequest(
        a_dir=namespace.a_dir,
        b_dir=namespace.b_dir,
        expect_same=_parse_expectation(namespace.expect_same),
    )


def _parse_expectation(value: str) -> Expectation:
    try:
        return EXPECTATION_VALUES[value]
    except KeyError as error:
        raise CliError(message="expect-same must be true or false") from error


def _compare(request: DeterminismRequest) -> DeterminismPayload:
    a_variants = _read_checksums(request.a_dir)
    b_variants = _read_checksums(request.b_dir)
    deterministic = a_variants == b_variants
    return DeterminismPayload(
        deterministic=deterministic,
        expected_same=request.expect_same,
        a=request.a_dir.as_posix(),
        b=request.b_dir.as_posix(),
        compared=a_variants,
    )


def _read_checksums(directory: Path) -> list[VariantChecksum]:
    metadata_variants = _read_metadata_variants(directory / "metadata.json")
    checksums: list[VariantChecksum] = []
    for variant in metadata_variants:
        output_path = directory / variant.output_file
        actual_sha256 = _sha256_file(output_path)
        if actual_sha256 != variant.recorded_sha256:
            raise CliError(
                message=(
                    "metadata/file checksum mismatch: "
                    f"{output_path.as_posix()} severity {variant.severity} "
                    f"recorded {variant.recorded_sha256} actual {actual_sha256}"
                ),
            )
        checksums.append(
            VariantChecksum(
                severity=variant.severity,
                output_file=variant.output_file,
                sha256=actual_sha256,
            ),
        )
    return checksums


def _read_metadata_variants(metadata_path: Path) -> list[MetadataVariant]:
    try:
        metadata_text = metadata_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise CliError(message=f"missing metadata: {metadata_path}") from error
    try:
        metadata = _json_value(json.loads(metadata_text))
    except (json.JSONDecodeError, CliError) as error:
        raise CliError(message=f"invalid metadata JSON: {metadata_path}") from error

    if not isinstance(metadata, dict):
        raise CliError(message=f"metadata root must be a JSON object: {metadata_path}")
    raw_variants = metadata.get("variants")
    if not isinstance(raw_variants, list) or len(raw_variants) == 0:
        raise CliError(message=f"metadata variant checksums missing: {metadata_path}")
    return [_parse_metadata_variant(metadata_path, variant) for variant in raw_variants]


def _parse_metadata_variant(
    metadata_path: Path,
    variant: JsonValue,
) -> MetadataVariant:
    if not isinstance(variant, dict):
        raise CliError(message=f"metadata variant must be an object: {metadata_path}")
    severity = _required_string(variant, "severity", metadata_path)
    output_file = _required_string(variant, "output_file", metadata_path)
    recorded_sha256 = _required_string(variant, "sha256", metadata_path)
    return MetadataVariant(
        severity=severity,
        output_file=output_file,
        recorded_sha256=recorded_sha256,
    )


def _required_string(
    metadata: dict[str, JsonValue],
    field: str,
    metadata_path: Path,
) -> str:
    value = metadata.get(field)
    if isinstance(value, str) and len(value) > 0:
        return value
    raise CliError(message=f"metadata variant missing {field}: {metadata_path}")


def _json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, float) and not math.isfinite(value):
        raise CliError(message="JSON numbers must be finite")
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return {key: _json_value(item) for key, item in value.items()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as image_file:
            while chunk := image_file.read(1024 * 1024):
                digest.update(chunk)
    except FileNotFoundError as error:
        raise CliError(message=f"missing output file: {path}") from error
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
