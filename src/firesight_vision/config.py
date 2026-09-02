import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

TomlScalar = str | int
TomlLeafTable = Mapping[str, TomlScalar]
TomlMidTable = Mapping[str, TomlScalar | TomlLeafTable]
TomlRootTable = Mapping[str, TomlScalar | TomlMidTable]


@dataclass(frozen=True, slots=True)
class FireSightConfig:
    project_name: str
    seed: int
    evidence_root: Path


@dataclass(frozen=True, slots=True)
class ConfigLoadError(Exception):
    path: Path
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


def load_config(path: Path) -> FireSightConfig:
    with path.open("rb") as config_file:
        raw_config: TomlRootTable = tomllib.load(config_file)

    raw_tool_config = _tool_config(path, raw_config)
    project_name = _string_value(path, raw_tool_config, "project_name")
    seed = _integer_value(path, raw_tool_config, "seed")
    evidence_root = _string_value(path, raw_tool_config, "evidence_root")

    return FireSightConfig(
        project_name=project_name,
        seed=seed,
        evidence_root=Path(evidence_root),
    )


def _tool_config(path: Path, raw_config: TomlRootTable) -> TomlLeafTable:
    tool_table = _mid_table(path, raw_config, "tool")
    return _leaf_table(path, tool_table, "firesight_vision")


def _mid_table(path: Path, table: TomlRootTable, key: str) -> TomlMidTable:
    value = table.get(key)
    if isinstance(value, Mapping):
        return value
    raise ConfigLoadError(path=path, reason=f"missing key '{key}'")


def _leaf_table(path: Path, table: TomlMidTable, key: str) -> TomlLeafTable:
    value = table.get(key)
    if isinstance(value, Mapping):
        return value
    raise ConfigLoadError(path=path, reason=f"missing key '{key}'")


def _string_value(path: Path, raw_config: TomlLeafTable, key: str) -> str:
    value = raw_config.get(key)
    if isinstance(value, str):
        return value
    if value is None:
        raise ConfigLoadError(path=path, reason=f"missing key '{key}'")
    raise ConfigLoadError(path=path, reason=f"{key} must be a string")


def _integer_value(path: Path, raw_config: TomlLeafTable, key: str) -> int:
    value = raw_config.get(key)
    if isinstance(value, int):
        return value
    if value is None:
        raise ConfigLoadError(path=path, reason=f"missing key '{key}'")
    raise ConfigLoadError(path=path, reason=f"{key} must be an integer")
