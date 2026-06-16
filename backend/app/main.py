from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.scans import router as scans_router
from app.config import Settings
from app.pipeline.inference import MockInferencePipeline
from app.services.scan_service import ScanService
from app.storage import JSONScanStore


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the FastAPI app with explicit dependencies for testability."""
    resolved_settings = settings or Settings()
    store = JSONScanStore(resolved_settings.storage_path)
    pipeline = MockInferencePipeline()
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

    app.include_router(health_router)
    app.include_router(scans_router)
    return app


app = create_app()

