from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator
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
        / "orthodontic_plaque_detection_mvp_v4_originals_online_aug_predict.toml"
    )


def default_ml_temp_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "var" / "ml-inputs"


def default_ml_runtime_artifact_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "var" / "ml-runtime"


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
    runtime_mode: Literal["development", "production"] = "development"
    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = "INFO"
    inference_mode: Literal["mock", "ml"] = "mock"
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, ge=1)
    storage_path: Path = Field(default_factory=default_storage_path)
    ml_source_path: Path = Field(default_factory=default_ml_source_path)
    ml_detection_config_path: Path = Field(default_factory=default_ml_detection_config_path)
    ml_temp_dir: Path = Field(default_factory=default_ml_temp_dir)
    ml_runtime_artifact_dir: Path = Field(
        default_factory=default_ml_runtime_artifact_dir
    )
    ml_max_concurrent_inferences: int = Field(default=1, ge=1)
    ml_delete_checkpoint_after_load: bool = False
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

    def resolve_project_path(self, path: Path) -> Path:
        """Resolve configured project paths without relying on process CWD."""

        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = default_project_root() / candidate
        return candidate.resolve(strict=False)

    @model_validator(mode="after")
    def validate_production_runtime(self) -> "Settings":
        if self.runtime_mode != "production":
            return self

        if self.inference_mode != "ml":
            raise ValueError("Production runtime requires inference_mode='ml'.")

        required_explicit_fields = {
            "ml_detection_config_path",
            "ml_temp_dir",
            "ml_runtime_artifact_dir",
            "cors_allowed_origins",
        }
        missing_fields = sorted(required_explicit_fields - self.model_fields_set)
        if missing_fields:
            raise ValueError(
                "Production runtime requires explicit settings for: "
                + ", ".join(missing_fields)
                + "."
            )

        config_path = self.resolve_project_path(self.ml_detection_config_path)
        if config_path.is_symlink() or not config_path.is_file():
            raise ValueError(
                "Production ML detection config must be an existing regular file."
            )

        temp_dir = Path(self.ml_temp_dir)
        artifact_dir = Path(self.ml_runtime_artifact_dir)
        if not temp_dir.is_absolute() or not artifact_dir.is_absolute():
            raise ValueError("Production ephemeral directories must use absolute paths.")
        if temp_dir.resolve(strict=False) == artifact_dir.resolve(strict=False):
            raise ValueError("Production ephemeral directories must be distinct.")

        if not self.cors_allowed_origins:
            raise ValueError("Production requires at least one frontend origin.")
        for origin in self.cors_allowed_origins:
            parsed = urlsplit(origin)
            if (
                origin == "*"
                or parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "Production CORS origins must be explicit HTTP(S) origins."
                )

        return self
