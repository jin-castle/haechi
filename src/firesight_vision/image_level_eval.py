from __future__ import annotations

import json
from typing import TYPE_CHECKING

from firesight_vision.image_level_eval_types import (
    EvalError,
    EvalItem,
    EvaluationPayload,
    JsonValue,
    MetricPayload,
    ModelEvaluationPayload,
    PredictionSet,
)

__all__ = ["EvalError", "JsonValue", "evaluate_manifest", "write_evaluation"]

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path


def evaluate_manifest(
    manifest_path: Path,
    prediction_paths: Sequence[Path],
) -> EvaluationPayload:
    classes, items = _read_manifest(manifest_path)
    models: dict[str, ModelEvaluationPayload] = {}
    for prediction_path in prediction_paths:
        predictions = _read_predictions(prediction_path, classes, items)
        models[predictions.model] = _evaluate_model(classes, items, predictions)
    return EvaluationPayload(
        classes=list(classes),
        image_count=len(items),
        manifest=manifest_path.as_posix(),
        models=models,
    )


def write_evaluation(
    manifest_path: Path,
    prediction_paths: Sequence[Path],
    out_path: Path,
) -> EvaluationPayload:
    payload = evaluate_manifest(manifest_path, prediction_paths)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _ = out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _read_manifest(path: Path) -> tuple[tuple[str, ...], tuple[EvalItem, ...]]:
    data = _read_json_object(path)
    classes = _required_string_list(data, "classes", path)
    if len(classes) == 0:
        raise EvalError(message="manifest classes must not be empty")
    items_data = data.get("items")
    if not isinstance(items_data, list) or len(items_data) == 0:
        raise EvalError(message="manifest items must be a non-empty list")
    class_set = set(classes)
    items = tuple(_parse_item(item, class_set, path) for item in items_data)
    _reject_duplicate_images(items)
    return tuple(classes), items


def _read_predictions(
    path: Path,
    classes: tuple[str, ...],
    manifest_items: tuple[EvalItem, ...],
) -> PredictionSet:
    data = _read_json_object(path)
    model = _required_string(data, "model", path)
    items_data = data.get("items")
    if not isinstance(items_data, list):
        raise EvalError(message=f"prediction items must be a list: {path}")
    class_set = set(classes)
    manifest_images = {item.image for item in manifest_items}
    items: dict[str, frozenset[str]] = {image: frozenset() for image in manifest_images}
    unknown_images: list[str] = []
    for item_data in items_data:
        item = _parse_item(item_data, class_set, path)
        if item.image not in manifest_images:
            unknown_images.append(item.image)
        else:
            items[item.image] = item.labels
    if len(unknown_images) > 0:
        joined = ", ".join(sorted(unknown_images))
        raise EvalError(message=f"unknown prediction images: {joined}")
    return PredictionSet(model=model, items=items)


def _evaluate_model(
    classes: tuple[str, ...],
    items: tuple[EvalItem, ...],
    predictions: PredictionSet,
) -> ModelEvaluationPayload:
    per_class = {
        class_name: _evaluate_class(class_name, items, predictions)
        for class_name in classes
    }
    return ModelEvaluationPayload(
        image_count=len(items),
        macro_precision=_mean_zero_filled(
            [metric["precision"] for metric in per_class.values()]
        ),
        macro_recall=_mean_zero_filled(
            [metric["recall"] for metric in per_class.values()]
        ),
        macro_f1=_mean_zero_filled([metric["f1"] for metric in per_class.values()]),
        per_class=per_class,
    )


def _evaluate_class(
    class_name: str,
    items: tuple[EvalItem, ...],
    predictions: PredictionSet,
) -> MetricPayload:
    tp = fp = fn = tn = 0
    for item in items:
        actual = class_name in item.labels
        predicted = class_name in predictions.items[item.image]
        if actual and predicted:
            tp += 1
        elif predicted:
            fp += 1
        elif actual:
            fn += 1
        else:
            tn += 1
    precision = _ratio_or_none(tp, tp + fp)
    recall = _ratio_or_none(tp, tp + fn)
    return MetricPayload(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=_f1_or_none(precision, recall),
        support=tp + fn,
    )


def _parse_item(
    item_data: JsonValue,
    classes: set[str],
    path: Path,
) -> EvalItem:
    if not isinstance(item_data, dict):
        raise EvalError(message=f"item must be an object: {path}")
    image = _required_string(item_data, "image", path)
    labels = frozenset(_required_string_list(item_data, "labels", path))
    unknown = sorted(labels - classes)
    if len(unknown) > 0:
        raise EvalError(message=f"unknown prediction labels: {', '.join(unknown)}")
    return EvalItem(image=image, labels=labels)


def _read_json_object(path: Path) -> dict[str, JsonValue]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise EvalError(message=f"missing JSON file: {path}") from error
    try:
        data = _json_value(json.loads(raw))
    except json.JSONDecodeError as error:
        raise EvalError(message=f"invalid JSON: {path}") from error
    if not isinstance(data, dict):
        raise EvalError(message=f"JSON root must be an object: {path}")
    return data


def _json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return {key: _json_value(item) for key, item in value.items()}


def _required_string(
    data: dict[str, JsonValue],
    key: str,
    path: Path,
) -> str:
    value = data.get(key)
    if isinstance(value, str) and len(value) > 0:
        return value
    raise EvalError(message=f"missing string field {key}: {path}")


def _required_string_list(
    data: dict[str, JsonValue],
    key: str,
    path: Path,
) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list):
        raise EvalError(message=f"missing string list field {key}: {path}")
    labels: list[str] = []
    for item in value:
        if not isinstance(item, str) or len(item) == 0:
            raise EvalError(message=f"{key} must contain non-empty strings: {path}")
        labels.append(item)
    return tuple(labels)


def _reject_duplicate_images(items: tuple[EvalItem, ...]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in items:
        if item.image in seen:
            duplicates.add(item.image)
        seen.add(item.image)
    if len(duplicates) > 0:
        raise EvalError(message=f"duplicate manifest images: {', '.join(duplicates)}")


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _f1_or_none(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def _mean_zero_filled(values: Iterable[float | None]) -> float | None:
    normalized = [0.0 if value is None else value for value in values]
    if len(normalized) == 0:
        return None
    return sum(normalized) / len(normalized)
