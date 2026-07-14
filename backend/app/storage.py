from __future__ import annotations

import json
import threading
from pathlib import Path

from app.schemas import ScanRecord


class JSONScanStore:
    """Small JSON-backed store for local development and tests."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def list(self) -> list[ScanRecord]:
        with self._lock:
            return self._read_records()

    def get(self, scan_id: str) -> ScanRecord | None:
        with self._lock:
            for record in self._read_records():
                if record.id == scan_id:
                    return record
        return None

    def save(self, record: ScanRecord) -> ScanRecord:
        with self._lock:
            records = self._read_records()
            records.append(record)
            self._write_records(records)
        return record

    def _read_records(self) -> list[ScanRecord]:
        if not self._path.exists():
            return []

        raw_records = json.loads(self._path.read_text(encoding="utf-8"))
        return [ScanRecord.model_validate(record) for record in raw_records]

    def _write_records(self, records: list[ScanRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            record.model_dump(mode="json")
            for record in sorted(records, key=lambda item: item.created_at)
        ]
        temp_path = self._path.with_suffix(f"{self._path.suffix}.tmp")
        temp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(self._path)
