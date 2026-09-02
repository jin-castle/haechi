from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final, TypedDict

from PIL import Image, ImageEnhance, ImageFilter

if TYPE_CHECKING:
    from pathlib import Path


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


def allowed_severity_values() -> tuple[str, ...]:
    return tuple(severity.value for severity in ALLOWED_SEVERITIES)


def smoke_parameters(severity: SmokeSeverity) -> SmokeParameters:
    return SMOKE_PARAMETERS[severity]


def apply_synthetic_smoke(request: SmokeTransformRequest) -> Image.Image:
    image = request.image.convert("RGB")
    params = smoke_parameters(request.severity)
    if request.severity is SmokeSeverity.CLEAN:
        return image.copy()

    mask = _fog_mask(image.size, request.seed, request.severity)
    fog_layer = Image.new("RGB", image.size, (226, 226, 219))
    fogged = Image.composite(fog_layer, image, mask)
    transmitted = Image.blend(fogged, fog_layer, 1.0 - params.transmission)
    return ImageEnhance.Contrast(transmitted).enhance(params.contrast)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        while chunk := image_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_metadata(
    input_path: Path,
    seed: int,
    variants: list[SmokeVariantPayload],
) -> SmokeMetadataPayload:
    return SmokeMetadataPayload(
        protocol=PROTOCOL_NAME,
        claim=CLAIM_TEXT,
        input_file=input_path.as_posix(),
        input_sha256=sha256_file(input_path),
        seed=seed,
        allowed_severities=list(allowed_severity_values()),
        variants=variants,
    )


def parameters_payload(parameters: SmokeParameters) -> SmokeParametersPayload:
    return SmokeParametersPayload(
        alpha=parameters.alpha,
        transmission=parameters.transmission,
        contrast=parameters.contrast,
    )


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
    alpha = smoke_parameters(severity).alpha
    return blurred.point(lambda value: int(value * alpha))


def _severity_offset(severity: SmokeSeverity) -> int:
    return SMOKE_SEVERITY_OFFSETS[severity]


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
