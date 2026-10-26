from pathlib import Path

import pytest

from karagoz import config


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / ".env"
    monkeypatch.setenv(config.ENV_FILE_VAR, str(path))
    monkeypatch.delenv("META_API_KEY", raising=False)
    return path


def test_reads_env_file(env_file: Path) -> None:
    env_file.write_text("# comment\nMETA_API_KEY = 'abc'\n", encoding="utf-8")
    assert config.secret("META_API_KEY") == "abc"


def test_environment_wins(env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file.write_text("META_API_KEY=file\n", encoding="utf-8")
    monkeypatch.setenv("META_API_KEY", "env")
    assert config.secret("META_API_KEY") == "env"


def test_missing_names_variable_not_value(env_file: Path) -> None:
    env_file.write_text("OTHER=secret-value\n", encoding="utf-8")
    with pytest.raises(config.MissingSecretError) as err:
        config.secret("META_API_KEY")
    assert "META_API_KEY" in str(err.value)
    assert "secret-value" not in str(err.value)


def test_env_file_inside_repo_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.ENV_FILE_VAR, str(config.REPO_ROOT / ".env"))
    assert config.env_file_inside_repo()


def test_default_env_file_outside_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(config.ENV_FILE_VAR, raising=False)
    assert not config.env_file_inside_repo()
