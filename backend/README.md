# Backend

The backend will expose the API used by the frontend to upload images, run inference, and return structured results.

## Responsibilities

- Accept image uploads
- Validate file type and size
- Call the ML inference pipeline
- Return prediction results
- Store scan history
- Provide health checks
- Log important events
- Expose testable API behavior

## Planned API Endpoints

```text
GET  /health
POST /scans
GET  /scans
GET  /scans/{scan_id}
```

## Learning Goals

- Understand what an API does
- Learn request and response schemas
- Learn input validation
- Learn backend error handling
- Learn how a backend calls ML code
- Learn how tests protect behavior

## Status

Scaffold created. Implementation has not started.

