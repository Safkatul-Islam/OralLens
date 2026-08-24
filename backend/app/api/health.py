from fastapi import APIRouter, Request, Response, status

from app.config import Settings
from app.pipeline.inference import InferencePipeline
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


def _health_response(request: Request, *, state: str) -> HealthResponse:
    settings: Settings = request.app.state.settings
    return HealthResponse(
        status=state,
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Backward-compatible service liveness endpoint."""

    return _health_response(request, state="ok")


@router.get("/health/live", response_model=HealthResponse)
def liveness(request: Request) -> HealthResponse:
    """Report that the API process can serve requests."""

    return _health_response(request, state="live")


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
)
def readiness(request: Request, response: Response) -> HealthResponse:
    """Report readiness only after the selected inference pipeline initializes."""

    pipeline: InferencePipeline = request.app.state.inference_pipeline
    if not pipeline.is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return _health_response(request, state="not_ready")
    return _health_response(request, state="ready")
