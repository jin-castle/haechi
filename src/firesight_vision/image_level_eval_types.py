from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class MetricPayload(TypedDict):
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float | None
    recall: float | None
    f1: float | None
    support: int


class ModelEvaluationPayload(TypedDict):
    image_count: int
    macro_precision: float | None
    macro_recall: float | None
    macro_f1: float | None
    per_class: dict[str, MetricPayload]


class EvaluationPayload(TypedDict):
    classes: list[str]
    image_count: int
    manifest: str
    models: dict[str, ModelEvaluationPayload]


@dataclass(frozen=True, slots=True)
class EvalItem:
    image: str
    labels: frozenset[str]


@dataclass(frozen=True, slots=True)
class PredictionSet:
    model: str
    items: dict[str, frozenset[str]]


@dataclass(frozen=True, slots=True)
class EvalError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message
