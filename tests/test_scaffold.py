from pathlib import Path

import pytest

from firesight_vision import __version__
from firesight_vision.config import ConfigLoadError, load_config


def test_package_imports_and_loads_default_config() -> None:
    config = load_config(Path("pyproject.toml"))

    assert __version__ == "0.1.0"
    assert config.project_name == "firesight-smoke-vision"
    assert config.seed == 42
    assert config.evidence_root == Path(".omo/evidence/firesight-smoke-vision")


def test_load_config_reports_missing_required_key(tmp_path: Path) -> None:
    config_path = tmp_path / "pyproject.toml"
    config_text = """[tool.firesight_vision]
project_name = "firesight-smoke-vision"
evidence_root = ".omo/evidence/firesight-smoke-vision"
"""
    _ = config_path.write_text(
        config_text,
        encoding="utf-8",
    )

    with pytest.raises(ConfigLoadError, match="missing key 'seed'"):
        _ = load_config(config_path)
