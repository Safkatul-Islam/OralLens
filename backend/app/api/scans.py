from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.schemas import ErrorResponse, ScanListResponse, ScanRecord
from app.services.scan_service import (
    ScanHistoryDisabledError,
    ScanService,
    UploadValidationError,
)

router = APIRouter(prefix="/scans", tags=["scans"])


def get_scan_service(request: Request) -> ScanService:
    return request.app.state.scan_service


@router.post(
    "",
    response_model=ScanRecord,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_413_CONTENT_TOO_LARGE: {"model": ErrorResponse},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"model": ErrorResponse},
    },
)
async def create_scan(
    file: Annotated[UploadFile, File(description="Oral image to screen.")],
    service: Annotated[ScanService, Depends(get_scan_service)],
) -> ScanRecord:
    try:
        return await service.create_scan(file)
    except UploadValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.get("", response_model=ScanListResponse)
def list_scans(
    service: Annotated[ScanService, Depends(get_scan_service)],
) -> ScanListResponse:
    try:
        return ScanListResponse(scans=service.list_scans())
    except ScanHistoryDisabledError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/{scan_id}",
    response_model=ScanRecord,
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
def get_scan(
    scan_id: str,
    service: Annotated[ScanService, Depends(get_scan_service)],
) -> ScanRecord:
    try:
        record = service.get_scan(scan_id)
    except ScanHistoryDisabledError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found.")
    return record
