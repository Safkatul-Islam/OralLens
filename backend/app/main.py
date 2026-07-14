from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.scans import router as scans_router
from app.config import Settings
from app.error_handlers import register_error_handlers
from app.logging_config import configure_logging
from app.middleware import request_context_middleware
from app.pipeline.inference import (
    InferencePipeline,
    MLDetectionInferencePipeline,
    MockInferencePipeline,
)
from app.services.scan_service import ScanService
from app.storage import JSONScanStore


def _build_inference_pipeline(settings: Settings) -> InferencePipeline:
    if settings.inference_mode == "ml":
        return MLDetectionInferencePipeline(
            config_path=settings.ml_detection_config_path,
            ml_source_path=settings.ml_source_path,
            temp_dir=settings.ml_temp_dir,
        )
    return MockInferencePipeline()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the FastAPI app with explicit dependencies for testability."""
    resolved_settings = settings or Settings()
    configure_logging(resolved_settings.log_level)
    store = JSONScanStore(resolved_settings.storage_path)
    pipeline = _build_inference_pipeline(resolved_settings)
    scan_service = ScanService(
        settings=resolved_settings,
        store=store,
        inference_pipeline=pipeline,
    )

    app = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        summary="AI screening-support API for oral image uploads.",
    )
    app.state.settings = resolved_settings
    app.state.scan_service = scan_service

    app.middleware("http")(request_context_middleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_allowed_origins),
        allow_credentials=resolved_settings.cors_allow_credentials,
        allow_methods=list(resolved_settings.cors_allowed_methods),
        allow_headers=list(resolved_settings.cors_allowed_headers),
    )
    register_error_handlers(app)

    app.include_router(health_router)
    app.include_router(scans_router)
    return app


app = create_app()
