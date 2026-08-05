from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_storage_path() -> Path:
    return Path(__file__).resolve().parents[1] / "var" / "scans.json"


def default_ml_source_path() -> Path:
    return default_project_root() / "ml" / "src"


def default_ml_detection_config_path() -> Path:
    return (
        default_project_root()
        / "ml"
        / "configs"
        / "orthodontic_plaque_detection_mvp_v3_predict.toml"
    )


def default_ml_temp_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "var" / "ml-inputs"


class Settings(BaseSettings):
    """Typed application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="ORALLENS_",
        env_file=".env",
        extra="ignore",
    )

    app_name: str = "OralLens AI Backend"
    app_version: str = "0.1.0"
    environment: str = "local"
    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = "INFO"
    inference_mode: Literal["mock", "ml"] = "mock"
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, ge=1)
    storage_path: Path = Field(default_factory=default_storage_path)
    ml_source_path: Path = Field(default_factory=default_ml_source_path)
    ml_detection_config_path: Path = Field(default_factory=default_ml_detection_config_path)
    ml_temp_dir: Path = Field(default_factory=default_ml_temp_dir)
    request_id_header: str = "X-Request-ID"
    cors_allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    )
    cors_allow_credentials: bool = False
    cors_allowed_methods: tuple[str, ...] = ("GET", "POST", "OPTIONS")
    cors_allowed_headers: tuple[str, ...] = (
        "Accept",
        "Content-Type",
        "X-Request-ID",
    )
    allowed_content_types: tuple[str, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
    )
    allowed_extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    )
