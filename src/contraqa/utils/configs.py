from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ("config",)

_PROJECT_NAME = "contraqa"


class _Config(BaseSettings):
    # config options (can be overridden from .env file or environment variables.
    openai_api_key: str
    together_api_key: str
    gemini_api_key: str
    root_dir: Path = Path(__file__).parents[3].resolve()
    results_dir: Path = root_dir / "results"
    project_dir: Path = root_dir / "src" / _PROJECT_NAME
    cache_dir: Path = root_dir / "cache"
    data_dir: Path = root_dir / "data"

    # config settings
    model_config = SettingsConfigDict(
        env_file=(".env", root_dir / ".env"), env_file_encoding="utf-8"
    )


config = _Config()


if __name__ == "__main__":  # pragma: nocover
    print(config.model_dump())
