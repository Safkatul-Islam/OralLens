# OralLens AI API

## Scope

The FastAPI service validates oral-image uploads, invokes the configured inference adapter, stores scan records locally, and returns structured screening-support output. The API does not return a diagnosis or treatment recommendation.

Local base URL: `http://127.0.0.1:8000`

Interactive OpenAPI documentation is available at `/docs` while the server is running.

## Runtime modes

| Mode | Default | Behavior |
|---|---:|---|
| `mock` | yes | Deterministic hash-based placeholder for API development and tests |
| `ml` | no | Project-local v2 TorchVision plaque-candidate detector |

`backend\scripts\run-ml-server.ps1` selects `ml` mode and the v2 prediction config. Application startup without that setting uses `mock` mode.

## Endpoints

### `GET /health`

Returns service status, name, version, and environment. It verifies that the API process can respond; it is not a deep detector-readiness check.

### `POST /scans`

Creates a scan from a multipart upload.

Request:

- content type: `multipart/form-data`
- field name: `file`
- maximum payload: 5 MiB by default
- frontend and ML mode: JPEG or PNG
- mock-mode backend validation also recognizes WebP, but the ML adapter rejects it cleanly with HTTP `415`

Example:

```powershell
curl.exe -X POST http://127.0.0.1:8000/scans `
  -F "file=@C:\path\to\oral-image.jpg;type=image/jpeg"
```

Successful response: HTTP `201`

```json
{
  "id": "scan-uuid",
  "created_at": "2026-07-29T00:00:00Z",
  "original_filename": "oral-image.jpg",
  "content_type": "image/jpeg",
  "size_bytes": 123456,
  "sha256": "content-digest",
  "prediction": {
    "label": "possible_plaque",
    "display_name": "Possible plaque candidate",
    "confidence": 0.96,
    "severity": "medium",
    "is_mock": false,
    "model_name": "orthodontic-plaque-mvp",
    "prediction_count": 17,
    "detections": [
      {
        "box_xyxy": [110.0, 72.0, 184.0, 131.0],
        "label": 1,
        "score": 0.96
      }
    ]
  },
  "evidence": {
    "kind": "summary",
    "summary": "Detected 17 plaque candidate(s) with the MVP detector."
  },
  "report": {
    "title": "AI Screening Support Report",
    "summary": "The current model pipeline marked this image as 'Possible plaque candidate' with 96% confidence.",
    "limitations": ["This backend response is screening-support output, not a diagnosis."],
    "recommended_next_steps": ["Consult a licensed dental professional for real symptoms or concerns."],
    "disclaimer": "OralLens AI is a learning project for screening support. It is not a medical device and does not provide diagnosis."
  }
}
```

The example is structural. Counts and scores depend on the input; confidence is the highest returned detector score, not a calibrated probability of disease.

### `GET /scans`

Returns all locally stored scan records:

```json
{
  "scans": []
}
```

### `GET /scans/{scan_id}`

Returns one stored scan or HTTP `404` when the identifier is unknown.

## Validation and errors

The upload boundary checks:

- configured MIME allowlist
- filename extension allowlist
- maximum size while reading in chunks
- non-empty content
- JPEG, PNG, or WebP magic bytes matching the declared type
- ML adapter support for the selected content type

The original filename is used only as sanitized display metadata. Stored records include a SHA-256 digest of validated bytes.

Error responses use a stable envelope:

```json
{
  "code": "error_code",
  "detail": "Concise client-safe explanation.",
  "request_id": "request-id"
}
```

| Status | Meaning |
|---:|---|
| `400` | Empty upload or bytes do not match the declared image type |
| `404` | Scan record does not exist |
| `413` | Upload exceeds the configured limit |
| `415` | MIME type, extension, or selected inference adapter does not support the input |
| `500` | Unexpected internal failure; internal details are not returned to the client |

## CORS

Local development allows only:

- `http://127.0.0.1:5173`
- `http://localhost:5173`

Allowed methods and headers are explicit; wildcard CORS is not used. Tests cover a configured origin, POST/Content-Type preflight, and rejection of an unconfigured origin.

## Persistence

The local implementation stores records in `backend/var/scans.json` using a lock and temporary-file replacement. The directory is ignored by git.

This store is intended only for local demonstration. It has no authentication, encryption policy, retention policy, database migrations, or multi-process coordination and must not hold sensitive clinical records.

## Verification

The latest backend suite completed with `23 passed, 1 skipped`; the opt-in real backend-to-ML integration smoke completed with `1 passed`. A manual browser request through the real v2 detector returned HTTP `201` and rendered the returned boxes.

See [Backend guide](../backend/README.md), [Architecture](ARCHITECTURE.md), and [Training and evaluation](TRAINING.md).
