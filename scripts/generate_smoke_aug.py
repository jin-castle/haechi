# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["Pillow>=10.4,<11"]
# ///
# ----- How to run -----
# python scripts/generate_smoke_aug.py
#   --input tests/fixtures/images/sample_room.jpg
#   --out .omo/evidence/firesight-smoke-vision/task-3/run-a
#   --severity clean mild medium dense --seed 42

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict

from PIL import Image, ImageEnhance, ImageFilter

if TYPE_CHECKING:
    from collections.abc import Sequence


class SmokeSeverity(StrEnum):
    CLEAN = "clean"
    MILD = "mild"
    MEDIUM = "medium"
    DENSE = "dense"


@dataclass(frozen=True, slots=True)
class SmokeParameters:
    alpha: float
    transmission: float
    contrast: float


@dataclass(frozen=True, slots=True)
class SmokeTransformRequest:
    image: Image.Image
    severity: SmokeSeverity
    seed: int


@dataclass(frozen=True, slots=True)
class GenerateSmokeRequest:
    input_path: Path
    out_dir: Path
    severities: tuple[SmokeSeverity, ...]
    seed: int


@dataclass(frozen=True, slots=True)
class CliError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


class SmokeParametersPayload(TypedDict):
    alpha: float
    transmission: float
    contrast: float


class SmokeVariantPayload(TypedDict):
    severity: str
    output_file: str
    sha256: str
    parameters: SmokeParametersPayload


class SmokeMetadataPayload(TypedDict):
    protocol: str
    claim: str
    input_file: str
    input_sha256: str
    seed: int
    allowed_severities: list[str]
    variants: list[SmokeVariantPayload]


class GenerateArgNamespace(argparse.Namespace):
    input_path: Path = Path()
    out_dir: Path = Path()
    severities: list[str] | None = None
    seed: int = 0


ALLOWED_SEVERITIES: Final[tuple[SmokeSeverity, ...]] = (
    SmokeSeverity.CLEAN,
    SmokeSeverity.MILD,
    SmokeSeverity.MEDIUM,
    SmokeSeverity.DENSE,
)
PROTOCOL_NAME: Final = "deterministic_synthetic_smoke_fog_v1"
CLAIM_TEXT: Final = (
    "Synthetic visibility degradation for repeatable robustness testing; "
    "not a physically accurate smoke model."
)
SMOKE_PARAMETERS: Final[dict[SmokeSeverity, SmokeParameters]] = {
    SmokeSeverity.CLEAN: SmokeParameters(alpha=0.0, transmission=1.0, contrast=1.0),
    SmokeSeverity.MILD: SmokeParameters(alpha=0.18, transmission=0.82, contrast=0.92),
    SmokeSeverity.MEDIUM: SmokeParameters(alpha=0.34, transmission=0.66, contrast=0.82),
    SmokeSeverity.DENSE: SmokeParameters(alpha=0.55, transmission=0.45, contrast=0.68),
}
SMOKE_SEVERITY_OFFSETS: Final[dict[SmokeSeverity, int]] = {
    SmokeSeverity.CLEAN: 0,
    SmokeSeverity.MILD: 101,
    SmokeSeverity.MEDIUM: 202,
    SmokeSeverity.DENSE: 303,
}


def main(argv: Sequence[str] | None = None) -> int:
    try:
        request = _parse_args(argv)
        metadata = _generate(request)
    except CliError as error:
        _ = sys.stderr.write(f"{error}\n")
        return 2

    _ = sys.stdout.write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return 0


def _parse_args(argv: Sequence[str] | None) -> GenerateSmokeRequest:
    namespace = GenerateArgNamespace()
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--input", dest="input_path", required=True, type=Path)
    _ = parser.add_argument("--out", dest="out_dir", required=True, type=Path)
    _ = parser.add_argument("--severity", dest="severities", required=True, nargs="+")
    _ = parser.add_argument("--seed", required=True, type=int)
    _ = parser.parse_args(argv, namespace=namespace)
    if namespace.severities is None:
        raise CliError(message="missing severity")
    return GenerateSmokeRequest(
        input_path=namespace.input_path,
        out_dir=namespace.out_dir,
        severities=tuple(_parse_severity(value) for value in namespace.severities),
        seed=namespace.seed,
    )


def _parse_severity(value: str) -> SmokeSeverity:
    try:
        return SmokeSeverity(value)
    except ValueError as error:
        allowed = ", ".join(_allowed_severity_values())
        raise CliError(
            message=f"unsupported severity '{value}'; allowed severities: {allowed}",
        ) from error


def _generate(request: GenerateSmokeRequest) -> SmokeMetadataPayload:
    request.out_dir.mkdir(parents=True, exist_ok=True)
    variants: list[SmokeVariantPayload] = []
    with Image.open(request.input_path) as source_image:
        for severity in request.severities:
            output_name = (
                f"{request.input_path.stem}_{severity.value}_seed{request.seed}.jpg"
            )
            output_path = request.out_dir / output_name
            transformed = _apply_synthetic_smoke(
                SmokeTransformRequest(
                    image=source_image,
                    severity=severity,
                    seed=request.seed,
                ),
            )
            transformed.save(output_path, format="JPEG", quality=95, optimize=False)
            variants.append(
                SmokeVariantPayload(
                    severity=severity.value,
                    output_file=output_name,
                    sha256=_sha256_file(output_path),
                    parameters=_parameters_payload(_smoke_parameters(severity)),
                ),
            )

    metadata = _build_metadata(request.input_path, request.seed, variants)
    _ = (request.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _apply_synthetic_smoke(request: SmokeTransformRequest) -> Image.Image:
    image = request.image.convert("RGB")
    params = _smoke_parameters(request.severity)
    if request.severity is SmokeSeverity.CLEAN:
        return image.copy()

    mask = _fog_mask(image.size, request.seed, request.severity)
    fog_layer = Image.new("RGB", image.size, (226, 226, 219))
    fogged = Image.composite(fog_layer, image, mask)
    transmitted = Image.blend(fogged, fog_layer, 1.0 - params.transmission)
    return ImageEnhance.Contrast(transmitted).enhance(params.contrast)


def _fog_mask(
    size: tuple[int, int],
    seed: int,
    severity: SmokeSeverity,
) -> Image.Image:
    width, height = size
    noise_width = max(4, width // 8)
    noise_height = max(4, height // 8)
    values = _deterministic_noise_values(
        seed=seed + _severity_offset(severity),
        severity=severity,
        count=noise_width * noise_height,
    )
    noise = Image.frombytes("L", (noise_width, noise_height), values)
    resized = noise.resize(size, Image.Resampling.BICUBIC)
    blurred = resized.filter(ImageFilter.GaussianBlur(radius=max(width, height) / 16))
    alpha = _smoke_parameters(severity).alpha
    return blurred.point(lambda value: int(value * alpha))


def _deterministic_noise_values(
    seed: int,
    severity: SmokeSeverity,
    count: int,
) -> bytes:
    values = bytearray()
    counter = 0
    while len(values) < count:
        digest = hashlib.sha256(
            f"{seed}:{severity.value}:{counter}".encode(),
        ).digest()
        values.extend(96 + (value % 160) for value in digest)
        counter += 1
    return bytes(values[:count])


def _build_metadata(
    input_path: Path,
    seed: int,
    variants: list[SmokeVariantPayload],
) -> SmokeMetadataPayload:
    return SmokeMetadataPayload(
        protocol=PROTOCOL_NAME,
        claim=CLAIM_TEXT,
        input_file=input_path.as_posix(),
        input_sha256=_sha256_file(input_path),
        seed=seed,
        allowed_severities=list(_allowed_severity_values()),
        variants=variants,
    )


def _parameters_payload(parameters: SmokeParameters) -> SmokeParametersPayload:
    return SmokeParametersPayload(
        alpha=parameters.alpha,
        transmission=parameters.transmission,
        contrast=parameters.contrast,
    )


def _allowed_severity_values() -> tuple[str, ...]:
    return tuple(severity.value for severity in ALLOWED_SEVERITIES)


def _smoke_parameters(severity: SmokeSeverity) -> SmokeParameters:
    return SMOKE_PARAMETERS[severity]


def _severity_offset(severity: SmokeSeverity) -> int:
    return SMOKE_SEVERITY_OFFSETS[severity]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        while chunk := image_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
