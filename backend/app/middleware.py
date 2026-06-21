import re
from collections.abc import Awaitable, Callable
from time import perf_counter
from uuid import uuid4

from fastapi import Request, Response

from app.config import Settings
from app.error_handlers import unhandled_exception_handler
from app.logging_config import get_logger

logger = get_logger(__name__)
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _resolve_request_id(request: Request) -> str:
    settings: Settings = request.app.state.settings
    candidate = request.headers.get(settings.request_id_header)
    if candidate and REQUEST_ID_PATTERN.fullmatch(candidate):
        return candidate
    return str(uuid4())


async def request_context_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Attach a safe request ID and emit one completion log per request."""
    settings: Settings = request.app.state.settings
    request_id = _resolve_request_id(request)
    request.state.request_id = request_id
    started_at = perf_counter()

    try:
        response = await call_next(request)
    except Exception as exc:
        response = await unhandled_exception_handler(request, exc)

    duration_ms = round((perf_counter() - started_at) * 1000, 2)
    response.headers[settings.request_id_header] = request_id
    logger.info(
        "Request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response

