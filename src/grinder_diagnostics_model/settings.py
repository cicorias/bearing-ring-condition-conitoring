from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from grinder_diagnostics_model.constants import DEFAULT_DATA_ROOT


class ModelSettings(BaseSettings):
    # Mise loads this file for project tasks; this fallback supports direct execution.
    model_config = SettingsConfigDict(
        env_prefix="GRINDER_DIAGNOSTICS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_root: Path = DEFAULT_DATA_ROOT
