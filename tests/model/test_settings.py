from pathlib import Path

from grinder_diagnostics_model.settings import ModelSettings


def test_data_root_defaults_to_repository_source_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GRINDER_DIAGNOSTICS_DATA_ROOT", raising=False)

    assert ModelSettings().data_root == Path("data/source")


def test_data_root_loads_from_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GRINDER_DIAGNOSTICS_DATA_ROOT", "/data/from-environment")

    assert ModelSettings().data_root == Path("/data/from-environment")


def test_data_root_loads_from_dotenv_without_mise(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GRINDER_DIAGNOSTICS_DATA_ROOT", raising=False)
    (tmp_path / ".env").write_text(
        "GRINDER_DIAGNOSTICS_DATA_ROOT=/data/from-dotenv\n",
        encoding="utf-8",
    )

    assert ModelSettings().data_root == Path("/data/from-dotenv")


def test_environment_overrides_dotenv(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "GRINDER_DIAGNOSTICS_DATA_ROOT=/data/from-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GRINDER_DIAGNOSTICS_DATA_ROOT", "/data/from-environment")

    assert ModelSettings().data_root == Path("/data/from-environment")
