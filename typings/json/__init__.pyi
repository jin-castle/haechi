from typing import TypeAlias

JsonValue: TypeAlias = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)


class JSONDecodeError(ValueError): ...


def loads(s: str | bytes | bytearray) -> JsonValue: ...


def dumps(
    obj: object,
    *,
    indent: int | str | None = None,
    sort_keys: bool = False,
) -> str: ...
