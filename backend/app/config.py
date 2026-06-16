from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def default_storage_path() -> Path:
    return Path(__file__).resolve().parents[1] / "var" / "scans.json"


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
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, ge=1)
    storage_path: Path = Field(default_factory=default_storage_path)
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

