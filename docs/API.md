# OralLens AI API Contract

## Purpose

The backend provides a stable HTTP contract between the frontend and the oral-image screening pipeline. The current inference response is deterministic mock data until a trained model is integrated.

Default local base URL:

```text
http://127.0.0.1:8000
```

Interactive OpenAPI documentation is available at `/docs` while the API is running.

## Request IDs

Clients may send a request ID:

```text
X-Request-ID: portfolio-request-1
```

The value must contain 1-128 letters, numbers, dots, underscores, or hyphens. Invalid or missing IDs are replaced with a UUID. Every response returns the accepted/generated request ID in the same header.

## CORS

CORS uses an explicit allowlist. The default development origin is:

```text
http://localhost:5173
```

Production origins must be configured with `ORALLENS_CORS_ALLOWED_ORIGINS` as a JSON array. Wildcard origins are not enabled by default.

## Error Schema

All handled errors use this shape:

```json
{
  "code": "validation_error",
  "detail": "Request validation failed.",
  "request_id": "f490f47b-fc04-449f-9a5a-dd0c80a2c25f"
}
```

Unexpected errors return HTTP 500 with a generic message. Stack traces, local paths, and exception details are not returned to clients.

## Endpoints

### `GET /health`

Returns service status, version, and environment.

### `POST /scans`

Accepts one multipart field named `file`.

Accepted formats:

- JPEG: `.jpg` or `.jpeg`
- PNG: `.png`
- WebP: `.webp`

Validation checks:

- Declared content type
- Filename extension
- Image magic bytes
- Configured maximum upload size
- Non-empty content

Successful response: HTTP 201 with scan metadata, mock prediction, evidence summary, and responsible-AI report.

Possible errors:

- `400`: empty content or content/signature mismatch
- `413`: upload exceeds configured limit
- `415`: unsupported content type or extension
- `422`: missing or structurally invalid request
- `500`: unexpected internal error with safe public message

Raw uploaded image bytes are not persisted. The local store contains metadata and a SHA-256 hash.

### `GET /scans`

Returns stored scan metadata in creation order.

### `GET /scans/{scan_id}`

Returns one scan record or HTTP 404 when the ID does not exist.

## Configuration

Settings use the `ORALLENS_` environment prefix.

```text
ORALLENS_ENVIRONMENT=local
ORALLENS_LOG_LEVEL=INFO
ORALLENS_MAX_UPLOAD_BYTES=5242880
ORALLENS_CORS_ALLOWED_ORIGINS=["http://localhost:5173"]
ORALLENS_CORS_ALLOW_CREDENTIALS=false
```

## Logging Safety

Application logs are emitted as JSON. Request logs include method, URL path, status code, duration, and request ID. Request bodies, uploaded image bytes, authorization values, cookies, and query strings are not logged.

